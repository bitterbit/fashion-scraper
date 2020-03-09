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


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random



"""
classify between images where the model is fully in the frame to
pictures where the model is not entirly in the frame

main.py learn --positive-dir <path> --negative-dir <path> -o <output_path>
main.py predict <path>
"""

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

class NetOne(nn.Module):
    def __init__(self):
        super(NetOne, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1), # in_channels, out_channels, kernel_size
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(6, 6, 3, stride=1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(216, 4), # in_features, out_feautres
        )
                        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_sample():
    dataloader = get_data_loader()
    # get some random training images
    dataiter = iter(dataloader)
    types, images = dataiter.next()

    # show images
    print(images)
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % [x for x in types] ))


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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.ImageFolder(root='traindata-small/train', transform=transform)  
    testset = datasets.ImageFolder(root='traindata-small/test', transform=transform)  

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


    while True:
        net = NetOne()
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(net.parameters(), lr=0.07)
        #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        train(trainloader, net, criterion, optimizer) 
        get_accuracy(net, testloader, testset.classes)

        save = input("save? [y/N]") == "y"
        if save:
            path = 'net_'+str(random.randint(0, 10000))
            print("saving", path)
            torch.save(net.state_dict(), path)

    """
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    print('GroundTruth: ', ' '.join('%5s' % testset.classes[labels[j]] for j in range(4)))
    outputs = net(images) #Predict!
    _, predicted = torch.max(outputs.data, 1)
    print([testset.classes[x] for x in predicted])
    #imshow(torchvision.utils.make_grid(images)) 
    """

def classify(model_path, folder):
    net = NetOne()
    net.load_state_dict(torch.load(path))

    images = os.listdir(folder)

    pass

if __name__ == '__main__':
    cmd = "train"
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]

    if cmd == "train":
        main()
    elif cmd == "classify" and len(sys.argv) > 2:
        model_path = sys.argv[2]
        folder = sys.argv[3]
        classify(model_path, folder)
    else:
        print("Unknown cmd", cmd)
