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
#   - Testar standardization dos dados


def get_train_data(train_file_path):
    n = None
    last_epoch_mean = None
    video_capture = cv2.VideoCapture(train_file_path)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    imsize = int(width * height // 16)

    try:
        train_data = torch.load('video_train_data.pyt')
    except:

        a = [None, None]
        res = torch.zeros(n - 1 if n else n_frames - 1, requires_grad=False)
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
        video_capture.release()
        cv2.destroyAllWindows()

        print('will save')
        #torch.save(train_data, 'video_train_data.pyt')

    finally:
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
    # BATCH_SIZE = n_frames
    BATCH_SIZE = 10000

    epochs = 15
    net = Net(imsize)
    net.train()
    opt = optim.Adam(net.parameters(), lr=0.0001)
    loss = nn.MSELoss()
    for e in range(epochs):
        print(f'epoch: {e}')
        loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True, num_workers=3)
        loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=n_frames,
            shuffle=False, num_workers=4)

        loader_train_data, loader_target = iter(loader).next()
        start = int(np.random.rand() * (n_frames - BATCH_SIZE)) - 1
        end = start + BATCH_SIZE
        print(start, end)

        outputs = torch.zeros(BATCH_SIZE)
        hidden = torch.zeros(250)
        for i, i_frame in enumerate(range(start, end)):
            frame = loader_train_data[i_frame]
            if (i - 1) % 1000 == 0:
                print(f'{i}th iteration')
            net_res, hidden = net(frame.view(-1).float(), hidden)
            outputs[i] = net_res

        opt.zero_grad()
        # net_res, hidden = net(train_data.view(BATCH_SIZE, -1).float(), hidden)
        # pred = net_res.view(BATCH_SIZE)
        print('Calculating loss...')
        l = loss(outputs, loader_target[start:end])
        print('Backpropagating...')
        l.backward()
        print('loss =', l.item())
        print()
        opt.step()
    # net_res, hidden = net(loader_train_data.view(
    #    BATCH_SIZE, -1).float(), hidden)
    # pred = net_res.view(BATCH_SIZE)
    # l = loss(pred, target)
    # print('net output:', pred[:50])
    # print('target:', target[:50])
    # print('loss =', l.item())
    # print()
    torch.save(net.state_dict(), 'net_rnn4.pt')
