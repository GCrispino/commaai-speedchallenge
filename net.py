import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, insize):
        super(Net, self).__init__()
        self.l1 = nn.Linear(insize, 2000)
        self.l2 = nn.Linear(2000, 500)
        self.l3 = nn.Linear(500, 250)
        self.l4 = nn.Linear(250, 1)

    def forward(self, x):
        #print('input:', x)
        x_ = self.l1(x)
        #print('l1 res:', x_)
        x_ = F.leaky_relu(x_)
        #print('l1 res relu:', x_)
        x_ = self.l2(x_)
        #print('l2 res:', x_)
        x_ = F.leaky_relu(x_)
        #print('l2 res relu:', x_)
        x_ = self.l3(x_)
        x_ = F.leaky_relu(x_)
        x_ = self.l4(x_)
        ##print('l3 res:', x_)
        return x_
