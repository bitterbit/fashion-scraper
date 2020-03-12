import os
import os.path
import sys
from skimage import io, transform
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchsummary import summary
from torchvision import models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

def get_transform():
    return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


"""
Layers:
- MaxPool: Down Sampling
- Dropout: randomise data against overfitting
- Conv2D: Non-linear, scan the image for features **
- Lienar: 
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
        )

        self.linear_layers = nn.Sequential(
                nn.Linear(16 * 5 * 5, 120), # in_features, out_feautres
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 24),
        ) 
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.linear_layers(x)
        return x


def train(loader, net, criterion, optimizer):
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 40:
                print(epoch+1, i+1, running_loss/50)
                running_loss = 0

    print("Finished trainning data")

def get_accuracy(net, testloader, classes):
    class_cnt = len(classes) 
    class_correct = list(0. for i in range(class_cnt))
    class_total = list(0. for i in range(class_cnt))

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            label = labels.item()
            class_correct[label] += c.item()
            class_total[label] += 1


    for i in range(class_cnt):
        correct = class_correct[i]
        total = class_total[i]
        print('Accuracy for class %s: %d %%' % (classes[i], (100 * correct / total)))


def main():
    transform = get_transform()
    trainset = datasets.ImageFolder(root='traindata-244/train', transform=transform)  
    testset = datasets.ImageFolder(root='traindata-244/test', transform=transform)  

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    #net = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.vgg16(pretrained=False).to(device)
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=2)


    summary(net, (3, 244, 244))
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.07)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(trainloader, net, criterion, optimizer) 
    get_accuracy(net, testloader, testset.classes)

      
    
if __name__ == '__main__':
    main()

