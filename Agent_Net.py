import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class AgentNet(nn.Module):
    def __init__(self):
        super(AgentNet, self).__init__()
        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = x.view(-1, 40)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
   
