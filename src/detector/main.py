import os
import os.path
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



"""
classify between images where the model is fully in the frame to
pictures where the model is not entirly in the frame

main.py learn --positive-dir <path> --negative-dir <path> -o <output_path>
main.py predict <path>
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            # inputs = [F.interpolate(x, size=32) for x in inputs]
            print ("i", type(labels), type(inputs))
            optimizer.zero_grad()
            outputs = net(inputs)

            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 40:
                print(epoch+1, i+1, running_loss/50)
                running_loss = 0

    print("Finished trainning data")


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.ImageFolder(root='traindata-small/train', transform=transform)  
    testset = datasets.ImageFolder(root='traindata-small/test', transform=transform)  

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(trainloader, net, criterion, optimizer) 

if __name__ == '__main__':
    main()
