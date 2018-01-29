import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

class Fast_Net(nn.Module):

    def __init__(self):
        super(Fast_Net, self).__init__()
        self.fc0 = nn.Linear(4, 20)
        self.fc1 = nn.Linear(20, 18)
        self.fc2 = nn.Linear(18, 10)
        self.fc3 = nn.Linear(10, 3)
    
    def forward(self, x, isnumpy=True):
        x = np.array(x)
        #print(x.shape)
        x = torch.from_numpy(x)
        x = Variable(x)
        x = x.type(torch.FloatTensor)
        x = x.view(-1,4)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if isnumpy: return x.data.numpy()
        else: return x
    
    def train(self, inputs, target):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters()) 
        target = Variable(torch.from_numpy(target))
        target = target.type(torch.FloatTensor)
        optimizer.zero_grad()
        outputs = self(inputs, isnumpy=False)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        return loss

    def predict(self, x):
        return self.forward(x)

