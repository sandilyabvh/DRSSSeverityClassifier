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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import cv2
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}

"""     
mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)
train_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), #can be one if not using any pretrained resnet
    transforms.Resize(size=(28, 28)),
    #transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=1), #can be one if not using any pretrained resnet
    transforms.Resize(size=(28, 28)),
    #transforms.ToTensor(),
    normalize,
])
"""

def normalize_np(image, mean, std):
    grayscale_image = np.array(image.convert('L'))
    return (grayscale_image - mean) / std

mean = 0.1706
std = 0.2112
target_size = (28, 28) #(112, 112)
train_transform = transforms.Compose([
    transforms.Resize(size=target_size),
    transforms.Lambda(lambda x: normalize_np(x, mean, std)),
])

test_transform = transforms.Compose([
    transforms.Resize(size=target_size),
    transforms.Lambda(lambda x: normalize_np(x, mean, std)),
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


if __name__ == '__main__':
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=train_transform)
    testset = OCTDataset(args, 'test', transform=test_transform)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Found device', device)

    print('len(trainset):'+str(len(trainset)))
    print('len(testset):'+str(len(trainset)))   
    # Access data and labels directly from trainset and testset
    X_train = np.array([x for x, _ in trainset])
    print('Computed X_train')
    y_train = np.array([y for _, y in trainset])
    print('Computed y_train')
    X_test = np.array([x for x, _ in testset])
    print('Computed X_test')
    y_test = np.array([y for _, y in testset])
    print('Computed y_test')
    print("Loaded the train and test datasets as NumPy arrays")

    # Reshape the input arrays
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print('shape of X_train:'+str(X_train.shape))
    print('shape of X_test:'+str(X_test.shape))

    # Initialize PCA with the desired number of components
    n_components = int(X_train.shape[1]/4)  # Choose the number of principal components to keep (should be smaller than original dimension)
    pca = PCA(n_components=n_components)

    # Fit PCA on the training data and transform both training and test data
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print('shape of X_train_pca:'+str(X_train_pca.shape))
    print('shape of X_test_pca:'+str(X_test_pca.shape))
    
    """
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print('len of iris dataset:'+str(len(iris)))
    print('shape of X:'+str(X.shape))
    # Split the dataset into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('len of X_train:'+str(len(X_train)))
    """
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    svm = SVC(kernel='linear', C=1, gamma='scale', random_state=42)
    ovr_classifier = OneVsRestClassifier(svm)
    ovr_classifier.fit(X_train_scaled, y_train)
    print("Model fit complete")

    test_pred = ovr_classifier.predict(X_test_scaled)
    print("Prediction complete")

    test_balanced_accuracy = balanced_accuracy_score(y_test, test_pred)
    test_f1_score = f1_score(y_test, test_pred, average='weighted')
    print('\nTest Balanced Accuracy: {:0.4f} | Test F1 Score: {:0.4f}'.format(test_balanced_accuracy, test_f1_score))