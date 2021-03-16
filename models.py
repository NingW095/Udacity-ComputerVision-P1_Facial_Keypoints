## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
         #maxpooling layer 
        self.pool = nn.MaxPool2d(2,2)
        #batch norm
        #self.batchnorm1 = nn.BatchNorm2d(32, momentum = 0.9)
        #Dropout
        self.drop1 = nn.Dropout(p = 0.1)
        
        #convolution layer2
        self.conv2 = nn.Conv2d(32, 64, 3)
        #batch norm
        #self.batchnorm2 = nn.BatchNorm2d(64, momentum = 0.9)
        #Dropout
        self.drop2 = nn.Dropout(p = 0.2)
        
        #convolution layer3
        self.conv3 = nn.Conv2d(64, 128, 2)
        #batch norm
        #self.batchnorm3 = nn.BatchNorm2d(128, momentum = 0.9)
        #Dropout
        self.drop3 = nn.Dropout(p = 0.3)
   
        #fully-connected layer
        self.fc1 = nn.Linear(128*26*26, 1024)
        #dropout layer
        self.fc1_drop = nn.Dropout(p = 0.4)
        
        #fully-connected layer
        #self.fc2 = nn.Linear(1024, 512)
        
        #dropout layer
        #self.fc2_drop = nn.Dropout(p = 0.4)
        
        #136 output channels
        self.fc2 = nn.Linear(1024, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        #x = self.batchnorm1(x)
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.batchnorm2(x)
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.batchnorm3(x)
        x = self.drop3(x)
        #Flatten
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)  #final output
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
