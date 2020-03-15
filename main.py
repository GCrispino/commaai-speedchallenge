from sys import exit
from statistics import mean
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import cv2
from net import Net

# TODO:
#   - Fazer back propagation em batches, e não em toda iteração
#   - Testar standardization dos dados


def get_train_data(train_file_path):
    n = None
    last_epoch_mean = None
    video_capture = cv2.VideoCapture(train_file_path)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    imsize = int(width * height // 16)
    a = [None, None]
    res = torch.zeros(n - 1 if n else n_frames - 1, requires_grad=True)
    outputs = []
    train_data = torch.zeros(n_frames, int(height / 4), int(width / 4))
    for i in range(n_frames):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            break
        trans_frame = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (int(width / 4), int(height / 4))
        ) / 255
        train_data[i] = torch.from_numpy(trans_frame)

        if i % 1000 == 0:
            print(f'  {i}th frame')

        # a[1] = trans_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    # last_epoch_mean = mean(outputs)
    video_capture.release()
    cv2.destroyAllWindows()

    return train_data, imsize, n_frames


if __name__ == '__main__':
    # read train.txt file
    train_video_file_path = 'data/train.mp4'
    train_text_file_path = 'data/train.txt'
    target_lines = open(train_text_file_path).readlines()

    print('getting train data...')
    train_data, imsize, n_frames = get_train_data(train_video_file_path)
    print('finished getting train data!')
    target = Variable(torch.from_numpy(np.fromiter(
        map(float, target_lines), dtype=np.float32)))

    train_dataset = torch.utils.data.TensorDataset(train_data, target)
    BATCH_SIZE = n_frames
    loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, num_workers=3)

    loader_train_data, loader_target = iter(loader).next()
    #print(loader_train_data, loader_train_data.shape)

    epochs = 100
    net = Net(imsize)
    net.train()
    opt = optim.Adam(net.parameters(), lr=0.0001)
    loss = nn.MSELoss()
    for e in range(epochs):
        print(f'epoch: {e}')

        opt.zero_grad()
        # net_res = net(train_data.view(n_frames, -1).float())
        net_res = net(train_data.view(BATCH_SIZE, -1).float())
        pred = net_res.view(BATCH_SIZE)
        # print(net_res, net_res.shape, net_res.view(n_frames).shape)
        l = loss(pred, target)
        l.backward()
        # outputs.append(l.item())
        print('net output:', net_res.unique())
        print('target:', target)
        print('loss =', l.item())
        print()
        opt.step()
    net_res = net(loader_train_data.view(BATCH_SIZE, -1).float())
    pred = net_res.view(BATCH_SIZE)
    l = loss(pred, target)
    # print(net_res, net_res.shape, net_res.view(n_frames).shape)
    # outputs.append(l.item())
    print('net output:', pred[:50])
    print('target:', target[:50])
    print('loss =', l.item())
    print()
    torch.save(net.state_dict(), 'net.pt')
