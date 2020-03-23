import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, insize):
        super(Net, self).__init__()
        hidden_size = 250
        self.l1 = nn.Linear(insize + hidden_size, 2000)
        self.l2 = nn.Linear(2000, 500)
        self.l3 = nn.Linear(500, 250)
        self.l4 = nn.Linear(250, 1)

    def forward(self, x, h):
        x_ = self.l1(torch.cat((x, h)))
        x_ = F.leaky_relu(x_)
        x_ = self.l2(x_)
        x_ = F.leaky_relu(x_)
        x_ = self.l3(x_)
        x_ = F.leaky_relu(x_)
        h_ = x_
        x_ = self.l4(x_)
        return x_, h_
