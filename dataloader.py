import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from unet import UNet
from torchsummary import summary
import torch.nn.functional as F
LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


# Define dataset and dataloader
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.ToTensor()
])

############################################UNet#################################################
class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
################################################################################################

class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
            
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        # print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        # self.subset = subset
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        # idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):
        #print("test"+str(index))
        img, target = Image.open(self.root+self.path_list[index]), self._labels[index]
        #print("Path for image opened using __getitem__:"+str(self.root+self.path_list[index]))
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self._labels)         

#define the neural network architecture
class OCTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = resnet18()
        self.features.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=train_transforms)
    testset = OCTDataset(args, 'test', transform=test_transforms)
    print(trainset[1][0].shape)
    print(len(trainset), len(testset))
    
    #set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device setup done")
    
    #define the hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("Hyperparameters defined and loaders ready")
    
    #initialize the model and optimizer
    # Define model
    #model = UNet(1, 4).to(device)
    model = UNet(n_classes=3, padding=True, up_mode='upsample').to(device)
    #UNet(in_channels=3, out_channels=4).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model and optimizer ready")
    
    #define the loss function
    criterion = nn.BCEWithLogitsLoss()
    
    #train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, masks) in enumerate(trainloader):
            images = images.to(device)
            print(images.size())
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            print(masks)
            print(outputs)
            print("Outputs size:"+str(outputs.shape))
            print("Masks size:"+str(masks.shape))
            # Compute loss using the resized Outputs tensor and one-hot encoded Masks tensor
            loss = F.cross_entropy(outputs, masks)
            #########################################################
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, running_loss/len(trainloader)))
    print("Model training complete")
    
    #evaluate the model on the test set
    correct = 0
    total = 0
    predicted_list = list()
    label_list = list()
    # Test the model
    model.eval()
    with torch.no_grad():
        for images, masks in testloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            label_list.append(masks)
            predicted_list.append(predicted)
            
    print("Model testing complete")
    print("Balanced accuracy:"+str(balanced_accuracy_score(np.concatenate(label_list), np.concatenate(predicted_list))))
    print("F1 Score:"+str(f1_score(np.concatenate(label_list), np.concatenate(predicted_list), average='micro')))
    #print('Accuracy on set: %d %%' % (100 * correct / total))