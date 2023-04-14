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
from pprint import pprint
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import cv2
from sklearn.model_selection import GridSearchCV
from torchvision.models import resnet50

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}
                   
class ResNetGray(nn.Module):
    def __init__(self):
        super(ResNetGray, self).__init__()
        resnet = resnet50() #pretrained=True)
        # Change the first conv layer to accept 1 channel instead of 3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x
        
class ConvertToHue(transforms.Lambda):
    def __init__(self):
        super().__init__(self.convert_to_hue)

    def convert_to_hue(self, img):
        img = np.array(img)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        return hue
        
mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=(224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop((224,224), padding=4),
    ConvertToHue(),
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

    #set up the device (GPU or CPU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Found device')

    #feature extractor
    # Load the pre-trained ResNet50 CNN model
    cnn_model = ResNetGray() #resnet50()#pretrained=True)

    # Remove the last fully connected layer
    feature_extractor = nn.Sequential(*list(cnn_model.children())[:-1])
    
    #dataset distribution:    
    #train_class_distribution = count_class_distribution(trainset)
    #test_class_distribution = count_class_distribution(testset)
    #print("Trainset class distribution:", train_class_distribution)
    #print("Testset class distribution:", test_class_distribution) 
            
    # Convert the train and test datasets to NumPy arrays
    X_train, y_train = zip(*trainset)
    X_test, y_test = zip(*testset)
    # Extract features from the train and test sets using the feature extractor
    X_train_features = []
    for img in X_train:
        features = feature_extractor(torch.Tensor(img).unsqueeze(0)).detach().numpy().flatten()
        X_train_features.append(features)
    X_train_features = np.array(X_train_features)

    X_test_features = []
    for img in X_test:
        features = feature_extractor(torch.Tensor(img).unsqueeze(0)).detach().numpy().flatten()
        X_test_features.append(features)
    X_test_features = np.array(X_test_features)
    
    print("Converted the train and test datasets to NumPy arrays")
    
    # Reshape the input arrays
    X_train = X_train_features.reshape(X_train_features.shape[0], -1)
    X_test = X_test_features.reshape(X_test_features.shape[0], -1)

    # Initialize the SVC model with the RBF kernel
    #-------without grid search-----------
    #model = SVC(kernel='rbf', C=1, gamma='scale').to(device)
    # Train the model
    #model.fit(X_train, y_train)
    # Make predictions on the train and test sets
    #train_pred = model.predict(X_train)
    #test_pred = model.predict(X_test)
    
    # Training the model
    # Define the parameter grid for the grid search
    #param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 1e-3, 1e-4]}
    param_grid = {'C': [10], 'gamma': [1e-4]}
    # Initialize the GridSearchCV object
    grid = GridSearchCV(SVC(kernel='rbf'), param_grid, verbose=3, scoring='balanced_accuracy', n_jobs=-1, cv=5)
    print("parameter grid for the grid search computation complete")
    # Fit the grid search object to the data
    grid.fit(X_train, y_train)
    print("fit complete")
    # Print the best parameters found by grid search
    print("Best parameters found by grid search:", grid.best_params_)
    # Make predictions on the train and test sets using the best estimator
    train_pred = grid.best_estimator_.predict(X_train)
    test_pred = grid.best_estimator_.predict(X_test)
    print("Prediction complete")

    # Calculate the evaluation metrics
    train_balanced_accuracy = balanced_accuracy_score(y_train, train_pred)
    train_f1_score = f1_score(y_train, train_pred, average='weighted')

    test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
    test_f1_score = f1_score(y_test, test_pred, average='weighted')

    print('\nTraining Balanced Accuracy: {:0.4f} | Training F1 Score: {:0.4f}'.format(train_balanced_accuracy, train_f1_score))
    print('Test Balanced Accuracy: {:0.4f} | Test F1 Score: {:0.4f}'.format(test_balanced_accuracy, test_f1_score))