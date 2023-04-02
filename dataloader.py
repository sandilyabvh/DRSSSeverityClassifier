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
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    normalize,
])
    

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

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self._labels)         

#define the neural network architecture
class OCTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = resnet18(pretrained=True)
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
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    print(trainset[1][0].shape)
    print(len(trainset), len(testset))
    
    #set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #define the hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 1
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #initialize the model and optimizer
    model = OCTClassifier(trainset.nb_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #define the loss function
    criterion = nn.CrossEntropyLoss()
    
    #train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
    
    #evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on set: %d %%' % (100 * correct / total))