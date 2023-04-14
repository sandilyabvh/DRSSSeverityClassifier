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
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from pprint import pprint
import segmentation_models_pytorch as smp
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop((224,224), padding=4),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])
    

class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None, max_samples=None):
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
        max_samples = int(len(self._labels)/4) #32 #int(len(self._labels)/2)
        #print(max_samples)
        self.max_samples = max_samples
        pprint(self.annot.keys())
        
    def __getitem__(self, index):
        #print("test"+str(index))
        img, target = Image.open(self.root+self.path_list[index]), self._labels[index]
        #print("Path for image opened using __getitem__:"+str(self.root+self.path_list[index]))
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self._labels), self.max_samples)
        else:
            return len(self._labels)    

#define the neural network architecture
class OCTClassifier(torch.nn.Module):
    def __init__(self):
        super(OCTClassifier, self).__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",   # choose the encoder architecture (e.g., resnet34)
            encoder_weights="imagenet", # use pre-trained weights for the encoder
            in_channels=3,              # input channels
            classes=3,                  # number of output channels
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # Pooling layer to get the [32, 3] output shape

    def forward(self, x):
        x = self.unet(x)               # U-Net forward pass
        x = self.avg_pool(x)           # Apply the pooling layer
        x = x.view(x.size(0), -1)      # Reshape the tensor to [batch_size, 3]

        # Apply softmax activation
        #x = torch.softmax(x, dim=1)
        
        # Get the class labels
        #x = torch.argmax(x, dim=1)
        
        return x
         
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type = str, default = 'df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type = str, default = 'df_prime_test.csv')
    parser.add_argument('--data_root', type = str, default = '')
    return parser.parse_args()

def count_class_distribution(dataset):
    class_counts = np.zeros(len(np.unique(list(LABELS_Severity.values()))))
    for _, label in dataset:
        class_counts[label] += 1
    return class_counts
      
if __name__ == '__main__':
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=train_transform)
    testset = OCTDataset(args, 'test', transform=test_transform)
    #print(trainset[1][0].shape)
    #print(len(trainset), len(testset))
    #dataset distribution:    
    train_class_distribution = count_class_distribution(trainset)
    test_class_distribution = count_class_distribution(testset)
    print("Trainset class distribution:", train_class_distribution)
    print("Testset class distribution:", test_class_distribution)    

    #set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Found device')
    
    #define the hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10 #5
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Train and Test loader complete')
    
    #initialize the model and optimizer
    model = OCTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Model definition complete')
    
    #define the loss function
    criterion = nn.CrossEntropyLoss()    
    # to handle imbalanced dataset:
    class_counts = np.bincount(trainset._labels)
    print(class_counts)
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(class_weights)
    #exit()
    
    #train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            print('Training start for batch: '+str(i))
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            #print('Before model call for batch: '+str(i))
            outputs = model(inputs)
            #print('After model call for batch: '+str(i))
            #print('outputs:', outputs)
            #print('labels:', labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print('Training end for batch: '+str(i))
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
             
    #evaluate the model on the test set
    model.eval()
    train_batch_images, train_batch_labels = next(iter(trainloader))  # train batch
    test_batch_images, test_batch_labels = next(iter(testloader))  # test batch

    train_pred = model(train_batch_images) #logits  
    probabilities = torch.softmax(train_pred, dim=1)
    train_pred = torch.argmax(probabilities, dim=1)
    #train_pred_rounded = torch.round(train_pred)
    train_pred_rounded = train_pred
    #train_pred_rounded = train_pred_rounded.detach().numpy()
    train_true = train_batch_labels
    
    test_pred = model(test_batch_images) #logits
    probabilities = torch.softmax(test_pred, dim=1)
    test_pred = torch.argmax(probabilities, dim=1)
    #test_pred_rounded = torch.round(test_pred)
    test_pred_rounded = test_pred
    #test_pred_rounded = test_pred_rounded.detach().numpy()
    test_true = test_batch_labels

    print("train_true shape:", train_true.shape)
    print("test_true shape:", test_true.shape)
    print("train_true:", train_true)
    print("test_true:", test_true)
    print("train_pred_rounded shape:", train_pred_rounded.shape)
    print("test_pred_rounded shape:", test_pred_rounded.shape)
    print("train_pred_rounded:", train_pred_rounded)
    print("test_pred_rounded:", test_pred_rounded)
        
    train_balanced_accuracy = balanced_accuracy_score(train_true, train_pred_rounded)
    train_f1_score = f1_score(train_true, train_pred_rounded, average='weighted')
    
    test_balanced_accuracy = balanced_accuracy_score(test_true, test_pred_rounded)
    test_f1_score = f1_score(test_true, test_pred_rounded, average='weighted')
    
    print('\nTraining Balanced Accuracy: {:0.4f} | Training F1 Score: {:0.4f}'.format(train_balanced_accuracy, train_f1_score))
    print('Test Balanced Accuracy: {:0.4f} | Test F1 Score: {:0.4f}'.format(test_balanced_accuracy, test_f1_score))