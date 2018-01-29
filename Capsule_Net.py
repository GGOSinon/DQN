import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class CapsuleNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)
        self.conv2 = nn.Conv2d(128, 128, 5)
        self.conv3 = nn.Conv2d(128, 128, 5)
        self.conv4 = nn.Conv2d(128, 10, 5)
        self.pool = nn.MaxPool2d(2,2)
    
    def forward(self, x, h=200, w=200):
        print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #x = self.pool(F.relu(self.conv5(x)))
        return x

    def predict(self, x):
        return self.forward(x)

