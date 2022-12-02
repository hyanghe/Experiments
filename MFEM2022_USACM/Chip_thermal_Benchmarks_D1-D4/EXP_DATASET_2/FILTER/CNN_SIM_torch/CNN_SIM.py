import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
import operator 
from Adam import Adam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import shutil
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from functools import reduce
from timeit import default_timer
from os.path import exists
import matplotlib
matplotlib.use('agg')
cur_path = os.getcwd()
os.makedirs('./CNN_Sim', mode=0o777, exist_ok=True)
work_dir = os.getcwd() + '/CNN_Sim/'
os.makedirs(work_dir + 'ckpt', mode=0o777, exist_ok=True)
ckpt_dir = work_dir + 'ckpt/'

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


class Conv_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, BN):
        super().__init__()
        if BN:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = (3, 3), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=mid_channels)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size = (3, 3), padding='same'),
                nn.ReLU(),
            )
    def forward(self, x):
        return self.conv_block(x)

class Residual_Block(nn.Module):
    def __init__(self, mid_channels, dilation_rates):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(3, 3), padding='same',
            dilation=dilation_rates[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(3, 3), padding='same',
            dilation=dilation_rates[1]),
            nn.ReLU(),
        )
    
    def forward(self, x):
        y = self.residual_block(x)
        out = torch.add(x, y, alpha=1)
        out = F.relu(out)
        return out

class encoder(nn.Module):
    def __init__(self, in_channels, mid_channels, n_params=2):
        super().__init__()
        self.BN = False
        # self.encoder = nn.Sequential(
        #     Conv_Block(in_channels, mid_channels, self.BN),
        #     # nn.MaxPool2d(kernel_size=(2,2), stride=None, padding='same'),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0),
        #     Conv_Block(mid_channels, mid_channels, self.BN),
        #     # nn.MaxPool2d(kernel_size=(2,2), stride=None, padding='same'),
        #     nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0),
        # )

        self.conv1 = Conv_Block(in_channels, mid_channels, self.BN)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)
        self.conv2 = Conv_Block(mid_channels, mid_channels, self.BN)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=None, padding=0)

        self.param_fc1 = nn.Linear(n_params, 80*80)
        self.param_conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        ########## Add parameters ##########
        y = self.param_fc1(y)
        y = F.relu(y)
        y = y.view(-1, 1, 80, 80)
        y = self.param_conv1(y)
        # print('x, y: ', x.shape, y.shape)
        # raise
        x = torch.cat((x, y), dim=1)
        ########## Add parameters ##########
        
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)


        # return self.encoder(x)
        return x

class decoder(nn.Module):
    def __init__(self, mid_channels, out_channels):
        super().__init__()
        self.BN = False
        self.decoder = nn.Sequential(
            Conv_Block(mid_channels, mid_channels, self.BN),
            nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
            Conv_Block(mid_channels, mid_channels, self.BN),
            nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')
        )

    def forward(self, x):
        return self.decoder(x)


# class decoder(nn.Module):
#     def __init__(self, mid_channels, out_channels):
#         super().__init__()
#         self.BN = False
#         self.conv1 = Conv_Block(mid_channels, mid_channels, self.BN)
#         self.up1 = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
#         self.conv2 = Conv_Block(mid_channels, mid_channels, self.BN)
#         self.up2 = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None)
#         self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3, 3), padding='same')

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.up1(x)
#         x = self.conv2(x)
#         x = self.up2(x)
#         x = self.conv3(x)
#         return x


class processor(nn.Module):
    def __init__(self, mid_channels, dCNN_dilations):
        super().__init__()
        self.dCNN_dilations = dCNN_dilations

        self.layers = []
        for dCNN_dilation_1, dCNN_dilation_2 in self.grouped(self.dCNN_dilations[1:], 2):
            dCNN_dilation = [dCNN_dilation_1, dCNN_dilation_2]
            self.layers.append(Residual_Block(mid_channels=mid_channels, dilation_rates=dCNN_dilation))

        self.processor = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=(3,3), padding='same',
            dilation=1),
            nn.ReLU(),
            *self.layers
        )
    
    def grouped(self, iterable, n):
        "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
        return zip(*[iter(iterable)]*n)
    
    def forward(self, x):
        return self.processor(x)

class CNN_sim(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dCNN_dilations):
        super(CNN_sim, self).__init__()        
        self.encoder = encoder(in_channels, mid_channels)
        self.processor = processor(mid_channels, dCNN_dilations)
        self.decoder = decoder(mid_channels, out_channels)

    def forward(self, x, y):
        # print('x shape is: ', x.shape)
        x = self.encoder(x, y)
        # print('After encoding, x shape is: ', x.shape)
        x = self.processor(x)
        # print('After processing, x shape is: ', x.shape)
        x = self.decoder(x)
        # print('After decoding, x shape is: ', x.shape)
        return x


def mape(true, pred, T_min, T_max): 
    true, pred = np.array(true), np.array(pred)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    mask = true != 0
    return np.mean(np.abs((true - pred) / true)[mask])

def relativeL2(true, pred, T_min, T_max):
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    # return np.linalg.norm(true.flatten() - pred.numpy().flatten()) / np.linalg.norm(true.flatten()) 
    return np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 

def mae(true, pred, T_min, T_max):
    print('pred true', pred.shape, true.shape)
    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))


# Data
def get_data(batch_size):
    print('==> Preparing data..')
    val_percent = 0.1
    # batch_size = 100

    dir_data = './data/'
    train_x = np.load(dir_data + 'x_train.npy').astype(np.float32)
    train_y = np.load(dir_data + 'y_train.npy').astype(np.float32)
    parameter_train = np.loadtxt('./data/parameters_train.txt', skiprows=1)

    ##### Filter unrealistic cases #####
    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    parameter_train = parameter_train[idx_train]
    ##### Filter unrealistic cases #####


    ##### HTC filtering and scaling ####
    idx_htc7 = parameter_train[:, 0] >= 1e-6
    train_x = train_x[idx_htc7]
    parameter_train = parameter_train[idx_htc7]
    train_y = train_y[idx_htc7]
    ##### HTC filtering and scaling ####

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    ##### Parameter scaling ####
    parameter_train[:, 0] = np.log(parameter_train[:, 0]*10**6) * 0.1
    parameter_train[:, 1] = (parameter_train[:, 1] - 20.0) / (200.0 - 20.0)
    ##### Parameter scaling ####

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)
    parameter_train = torch.Tensor(parameter_train)
    

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    train_x = torch.permute(train_x, (0,3,1,2))
    train_y = torch.permute(train_y, (0,3,1,2))


    test_x = np.load(dir_data + 'x_test.npy').astype(np.float32)
    test_y = np.load(dir_data + 'y_test.npy').astype(np.float32)
    parameter_test = np.loadtxt('./data/parameters_test.txt', skiprows=1)

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]
    parameter_test = parameter_test[idx_test]

    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    idx_htc7 = parameter_test[:, 0] >= 1e-6
    test_x = test_x[idx_htc7]
    parameter_test = parameter_test[idx_htc7]
    test_y = test_y[idx_htc7]
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    
    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    ##### Parameter scaling ####
    parameter_test[:, 0] = np.log(parameter_test[:, 0]*10**6) * 0.1
    parameter_test[:, 1] = (parameter_test[:, 1] - 20.0) / (200.0 - 20.0)
    ##### Parameter scaling ####


    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)
    parameter_test = torch.Tensor(parameter_test)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    test_x = torch.permute(test_x, (0,3,1,2))
    test_y = torch.permute(test_y, (0,3,1,2))

    # print('train_x: ', train_x.shape, train_y.shape, parameter_train.shape)
    # raise
    dataset = torch.utils.data.TensorDataset(train_x, train_y, parameter_train)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, parameter_test)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)

    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # print('train_set: ', train_set)
    # print('val_set: ', val_set)
    # raise
    # 3. Create data loaders

    # loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    loader_args = dict(batch_size=batch_size)
    trainloader = DataLoader(train_set, shuffle=True, **loader_args)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    testloader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    return trainloader, valloader, testloader, T_min, T_max, test_x, test_y, parameter_test


# def test_net(model,
#               device,
#               train_loader,
#               test_loader,
#               x_normalizer,
#               y_normalizer,
#               batch_size: int = 20,
#               s: int=80
#               ):
def test_net(model,
              DEVICE_NUM,
              testloader,
              T_min,
              T_max,
              input_size,
              batch_size
              ):
    with torch.no_grad():
        true = []
        pred = []

        for x, y, param in testloader:
            x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)

            out = model(x, param).reshape(batch_size, input_size, input_size)

            out = out * (T_max - T_min) + T_min
            y = y * (T_max - T_min) + T_min
            # print('y shape: ', y.detach().cpu().numpy().shape)
            # raise
            true.extend(y.detach().cpu().numpy())
            pred.extend(out.detach().cpu().numpy())
        true = np.squeeze(np.asarray(true))
        pred = np.squeeze(np.asarray(pred))

        mae = np.mean(np.abs(true - pred))
        # print('true.shape[0]', true.shape[0])
        # raise
        # idx_ls = np.random.choice(true.shape[0], 10)
        idx_ls = np.arange(10)
        # print('idx_ls: ', idx_ls)
        # raise
        # print('true.shape', true.shape)
        # print('pred.shape', pred.shape)
        # raise
        for idx in idx_ls:
            fig = plt.figure(figsize=(15, 5))
            plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
            ax = fig.add_subplot(131)
            ax.set_title(f'Truth')
            im = ax.imshow(true[idx, :,:], origin='lower', cmap='jet')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)

            ax = fig.add_subplot(132)
            im = ax.imshow(pred[idx, :,:], cmap='jet', origin='lower')
            ax.set_title(f'Pred')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)

            ax = fig.add_subplot(133)
            im = ax.imshow(abs(pred[idx, :,:] - true[idx, :,:]), cmap='jet', origin='lower')
            ax.set_title(f'Error')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="7%", pad="2%")
            cb = fig.colorbar(im, cax=cax)
            # plt.savefig(f'./figs/{cnt}.jpg')
            plt.savefig(f'./figs/final_test_sample_{idx}.jpg')
            # plt.show()
            plt.close()
            # cnt += 1
        
        # torch.save(net.state_dict(), "./checkpoint/network.pt")

        rel_l2 = np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 
        mape_error = mape(true, pred, T_min, T_max)
        print('mae: ', mae)
        print('rel_l2: ', rel_l2)
        print('mape_error: ', mape_error)
        with open('./final_test_l2.txt', 'w') as f:
            f.write(f'mae is: {mae} \n')
            f.write(f'relative l2 is: {rel_l2} \n')
            f.write(f'mape_error is: {mape_error} \n')
################################################################
# training and evaluation
################################################################

def get_args():
    parser = argparse.ArgumentParser(description='FNO')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--mode', '-m', type=str, default='test', help='train or test')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    DEVICE_NUM = args.gpu
    batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    mode = args.mode



    # DEVICE_NUM = 0
    # batch_size = 100
    # learning_rate = 3e-4
    # epochs = 5000
    # # mode = 'train'
    # mode = 'test'
    
    device = torch.device(f'cuda:{DEVICE_NUM}')

    ntrain = 5000
    ntest = 1000
    # s = 80
    input_size = 80
    trainloader, valloader, testloader, T_min, T_max, test_x, test_y, parameter_test  = get_data(batch_size)
    
    step_size = 100
    gamma = 0.5

    
    # num_filter = 48
    in_channels = 2
    mid_channels = 48
    out_channels = 1
    dCNN_dilations = [1, 2, 4, 8, 4, 2, 1]

    model = CNN_sim(in_channels, mid_channels, out_channels, dCNN_dilations).cuda(DEVICE_NUM)
    print(count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # myloss = LpLoss(size_average=False)
    # y_normalizer.cuda(DEVICE_NUM)
    criterion = nn.MSELoss()
    min_val_l2 = 10**10


    model_path = "./checkpoint/network.pt"
    file_exists = exists(model_path)
    if file_exists:
        print('model exist')
        model.load_state_dict(torch.load(model_path, map_location=device))
        num_params = count_params(model)
        print(f'model loaded, num_params: {num_params}')
    else:
        print('model does not exist!')

    if mode == 'train':
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for x, y, param in trainloader:
                x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)

                optimizer.zero_grad()
                out = model(x, param)
                # out = model(x).reshape(batch_size, out_channels, input_size, input_size)

                # out = (out - T_min) / (T_max - T_min)
                # y = (y - T_min) / (T_max - T_min)

                loss = criterion(out.view(batch_size,-1), y.view(batch_size,-1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()

            model.eval()

            val_l2 = 0.0
            with torch.no_grad():
                for x, y, param in valloader:
                    x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)

                    out = model(x, param).reshape(batch_size, input_size, input_size)
                    # out = (out - T_min) / (T_max - T_min)

                    val_l2 += criterion(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            train_l2/= len(trainloader)*trainloader.batch_size
            val_l2 /= len(valloader)*valloader.batch_size

            t2 = default_timer()
            print(f'epoch: {ep}, time: {t2-t1:.5f}, train_l2: {train_l2}, val_l2: {val_l2}')

            with torch.no_grad():
                if val_l2 < min_val_l2:
                    idx = np.random.choice(len(test_x))
                    x_p, y_p, param_p = test_x[idx:idx+1].cuda(DEVICE_NUM),\
                                        test_y[idx:idx+1].cuda(DEVICE_NUM),\
                                        parameter_test[idx:idx+1].cuda(DEVICE_NUM)
                    pred_p = model(x_p, param_p).reshape(1, input_size, input_size)

                    pred_p = pred_p * (T_max - T_min) + T_min
                    y_p = y_p * (T_max - T_min) + T_min
                    # y = y_normalizer.decode(y)
                    pred_p = np.squeeze(pred_p.detach().cpu().numpy())
                    true_p = np.squeeze(y_p.detach().cpu().numpy())
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    fig = plt.figure(figsize=(15, 5))
                    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
                    ax = fig.add_subplot(131)
                    ax.set_title(f'Truth')
                    im = ax.imshow(true_p, origin='lower', cmap='jet')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(132)
                    im = ax.imshow(pred_p, cmap='jet', origin='lower')
                    ax.set_title(f'Pred')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(133)
                    im = ax.imshow(abs(pred_p - true_p), cmap='jet', origin='lower')
                    ax.set_title(f'Error')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)
                    plt.savefig(f'./figs/case_{idx}.jpg')
                    # plt.show()
                    plt.close()
                    min_val_l2 = val_l2
                    torch.save(model.state_dict(), "./checkpoint/network.pt")
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    # network.load_state_dict(torch.load("./checkpoint/model.pt"))
    elif mode == 'test':
        model.eval()
        test_net(model,
              DEVICE_NUM,
              testloader,
              T_min,
              T_max,
              input_size,
              batch_size
              )