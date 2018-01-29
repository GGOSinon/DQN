import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from Base_Net import BaseNet
from Agent_Net import AgentNet
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        '''
        self.base = BaseNet()
        self.agent = AgentNet()
        '''
        self.conv1 = nn.Conv2d(3, 16, 8, stride = 4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride = 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 10, 4, stride = 2)
        self.bn3 = nn.BatchNorm2d(10)
        #self.conv4 = nn.Conv2d(32,10,5)
        #self.bn4 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(160, 512)
        #self.bf1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x, h=100, w=100, isnumpy=True):
        x = np.array(x)
        #print(x.shape)
        x = torch.from_numpy(x)
        x = Variable(x)
        x = x.type(torch.FloatTensor)
        x = x.view(-1,3,h,w)
        '''
        x = self.base(x)
        x = self.agent(x) 
        '''
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.bf1(self.fc1(x)))
        #x = x.view(-1, 64)
        x = self.fc2(x)
        if isnumpy: return x.data.numpy()
        else: return x
    
    def train(self, inputs, target):
        optimizer = optim.RMSprop(self.parameters()) 
        target = Variable(torch.from_numpy(target))
        target = target.type(torch.FloatTensor)
        optimizer.zero_grad()
        outputs = self(inputs, isnumpy=False)
        loss = F.smooth_l1_loss(outputs, target)
        loss.backward()
        #print(self)
        #print(self.parameters())
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        
        optimizer.step()
        return loss

    def predict(self, x):
        return self.forward(x)

