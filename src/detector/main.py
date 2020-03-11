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

def get_transform():
    return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



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

def imsave(path, img):
    img = img / 2 + 0.5     # unnormalize
    with open(path, "w") as f:
        torchvision.utils.save_image(img, f)

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

    
def classify(models, data_dir):
    nets = []
    for model in models:
        net = NetOne()
        net.load_state_dict(torch.load(model))
        nets.append(net)

    dataset = datasets.ImageFolder(data_dir, transform=get_transform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    with torch.no_grad():
        for images, _ in loader:
            combined = None
            for i in range(len(nets)):
                net = nets[i]
                netname = models[i]
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                netname = netname + " "*(20 - len(netname))
                print(netname, predicted)
                if combined is None:
                    combined = predicted
                else:
                    combined = np.add(combined, predicted)

            positive = combined == len(nets) 
            negative = combined < 3 
            # imshow(torchvision.utils.make_grid(images)) 
            for i in range(len(positive)):
                if positive[i]:
                    path = "out/positive/img{0}.jpg".format(str(random.randint(0, 10000)))
                    imsave(path, images[i])

            for i in range(len(negative)):
                if negative[i]:
                    path = "out/negative/img{0}.jpg".format(str(random.randint(0, 10000)))
                    imsave(path, images[i])



if __name__ == '__main__':
    cmd = "train"
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]

    if cmd == "train":
        main()
    elif cmd == "classify" and len(sys.argv) >= 2:
        models_path = ['net_2048', 'net_4377', 'net_4683', 'net_6038', 'net_6787','net_7676', 'net_832', 'net_9716', 'special/net_4377']
        folder = sys.argv[2]
        classify(models_path, folder)
    else:
        print("Unknown cmd", cmd)
