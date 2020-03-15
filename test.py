from sys import exit
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cv2
from net import Net
from main import get_train_data

# read train.txt file
target_lines = open('data/train.txt').readlines()
target = Variable(torch.from_numpy(np.fromiter(
    map(float, target_lines), dtype=np.float32)))

train_file = 'data/train.mp4'

video_capture = cv2.VideoCapture(train_file)
n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
imsize = width * height // 16
net = Net(imsize)
net.load_state_dict(torch.load('net.pt'))
net.eval()
n = None
a = [None, None]
i = 0
errs = []
loss = nn.MSELoss()


train_data, imsize, n_frames = get_train_data(train_file)
output = net(train_data.view(n_frames, -1).float())
pred = output.view(n_frames)
l = loss(pred, target)
print('net output:', output[:50])
print('target:', target[:50])
print('mean:', ((target - output) ** 2).mean())
print('loss:', l)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
