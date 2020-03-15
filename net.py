import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, insize):
        super(Net, self).__init__()
        self.l1 = nn.Linear(insize, 500)
        self.l2 = nn.Linear(500, 250)
        self.l3 = nn.Linear(250, 1)

    def forward(self, x):
        #print('input:', x)
        x_ = self.l1(x)
        #print('l1 res:', x_)
        x_ = F.relu(x_)
        #print('l1 res relu:', x_)
        x_ = self.l2(x_)
        #print('l2 res:', x_)
        x_ = F.relu(x_)
        #print('l2 res relu:', x_)
        x_ = self.l3(x_)
        ##print('l3 res:', x_)
        return x_
