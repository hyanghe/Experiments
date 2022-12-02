"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from os.path import exists
import argparse
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

DEVICE_NUM = 7
################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, n_params=2):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        # self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(4, self.width) # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)


        ## params
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        #     # nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
        #     # nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
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
        y = self.param_fc1(y)
        y = F.relu(y)
        y = y.view(-1, 1, 80, 80)
        y = self.param_conv1(y)
        y = y.permute(0, 2, 3, 1)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, y), dim=-1)
        x = torch.cat((x, grid), dim=-1)
        # print('y shape: ', y.shape)
        # print('x shape: ', x.shape)
        # raise
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
def get_dataLoader(batch_size, s):
    # TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
    # TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'

    # ntrain = 1000
    # ntest = 100

    # batch_size = 20
    # batch_size = 10
    # learning_rate = 0.001

    # epochs = 50000
    # epochs = 10
    # step_size = 100
    # gamma = 0.5

    # modes = 12
    # width = 32

    # r = 5
    # h = int(((421 - 1)/r) + 1)
    # s = h

    # s = 80
    # ntrain = 5000
    # ntest = 1000

    # ntrain = 4941
    # ntest = 789
    # ntrain = 90
    # ntest = 10
    ################################################################
    # load data and data normalization
    ################################################################
    # reader = MatReader(TRAIN_PATH)
    # x_train = reader.read_field('coeff')[:ntrain,::r,::r][:,:s,:s]
    # y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

    # reader.load_file(TEST_PATH)
    # x_test = reader.read_field('coeff')[:ntest,::r,::r][:,:s,:s]
    # y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]

    # x_train = torch.Tensor(np.load('./data/x_train.npy')[:ntrain])
    # y_train = torch.Tensor(np.load('./data/y_train.npy')[:ntrain])
    # x_test = torch.Tensor(np.load('./data/x_train.npy')[ntrain:])
    # y_test = torch.Tensor(np.load('./data/y_train.npy')[ntrain:])

    # x_train_100 = np.load('./data/x_train_100.npy')
    # y_train_100 = np.load('./data/y_train_100.npy')

    # x_train_1000 = np.load('./data/x_train.npy')
    # y_train_1000 = np.load('./data/y_train.npy')

    # x_train_all = np.concatenate((x_train_100, x_train_1000))
    # y_train_all = np.concatenate((y_train_100, y_train_1000))

    # print('x_train_shape: ', x_train.shape)
    # print('y_train_shape: ', y_train.shape)
    # raise
    # x_train = torch.Tensor(x_train_all[:ntrain])
    # y_train = torch.Tensor(y_train_all[:ntrain])
    # x_test = torch.Tensor(x_train_all[ntrain:])
    # y_test = torch.Tensor(y_train_all[ntrain:])

    # x_train = torch.Tensor(np.load('./data/x_train.npy')[:ntrain])
    # y_train = torch.Tensor(np.load('./data/y_train.npy')[:ntrain])
    # x_test = torch.Tensor(np.load('./data/x_train.npy')[ntrain:ntrain+ntest])
    # y_test = torch.Tensor(np.load('./data/y_train.npy')[ntrain:ntrain+ntest])

    # x_train = torch.Tensor(np.load('./data/x_train.npy'))
    # y_train = torch.Tensor(np.load('./data/y_train.npy'))
    # x_test = torch.Tensor(np.load('./data/x_test.npy'))
    # y_test = torch.Tensor(np.load('./data/y_test.npy'))
    # parameter_train = torch.Tensor(np.loadtxt('./data/parameters_train.txt', skiprows=1))
    # parameter_test = torch.Tensor(np.loadtxt('./data/parameters_test.txt', skiprows=1))

    x_train = np.load('./data/x_train.npy')
    y_train = np.load('./data/y_train.npy')
    x_test = np.load('./data/x_test.npy')
    y_test = np.load('./data/y_test.npy')
    parameter_train = np.loadtxt('./data/parameters_train.txt', skiprows=1)
    parameter_test = np.loadtxt('./data/parameters_test.txt', skiprows=1)


    idx_train = np.amax(y_train, axis=(1, 2)) < 300
    x_train = x_train[idx_train]
    y_train = y_train[idx_train]
    parameter_train = parameter_train[idx_train]

    ##### HTC filtering and scaling ####
    # idx_htc7 = parameter_train[:, 0] >= 1e-6
    # x_train = x_train[idx_htc7]
    # parameter_train = parameter_train[idx_htc7]
    # y_train = y_train[idx_htc7]
    ##### HTC filtering and scaling ####

    temp, power, htc, die_z = y_train, x_train, parameter_train[:, 0], parameter_train[:, 1]
    # train_y = scaling_strategy3(temp, power, htc, die_z)  
    ##### HTC filtering and scaling ####
    parameter_train[:, 0] = np.log(parameter_train[:, 0]*10**6) * 0.1
    parameter_train[:, 1] = (parameter_train[:, 1] - 20.0) / (200.0 - 20.0)


    idx_test = np.amax(y_test, axis=(1, 2)) < 300
    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    parameter_test = parameter_test[idx_test]

    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    # idx_htc7 = parameter_test[:, 0] >= 1e-6
    # x_test = x_test[idx_htc7]
    # parameter_test = parameter_test[idx_htc7]
    # y_test = y_test[idx_htc7]
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####
    ##### HTC filtering and scaling ####

    temp, power, htc, die_z = y_test, x_test, parameter_test[:, 0], parameter_test[:, 1]
    # test_y = scaling_strategy3(temp, power, htc, die_z)  
    ##### HTC filtering and scaling ####
    parameter_test[:, 0] = np.log(parameter_test[:, 0]*10**6) * 0.1
    parameter_test[:, 1] = (parameter_test[:, 1] - 20.0) / (200.0 - 20.0)
    # test_y = dataset.load_power(dir_data + 'y_test.npy')


    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    parameter_train = torch.Tensor(parameter_train)
    parameter_test = torch.Tensor(parameter_test)

    print('x_train_shape: ', x_train.shape)
    print('y_train_shape: ', y_train.shape)
    print('x_test_shape: ', x_test.shape)
    print('y_test_shape: ', y_test.shape)
    print('parameter_train shape: ', parameter_train.shape)
    # raise
    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)
    y_test = y_normalizer.encode(y_test)
    print('After normalizer:\n')
    print('x_train_shape: ', x_train.shape)
    print('y_train_shape: ', y_train.shape)
    print('x_test_shape: ', x_test.shape)
    print('y_test_shape: ', y_test.shape)
    # raise
    x_train = x_train.reshape(-1,s,s,1)
    x_test = x_test.reshape(-1,s,s,1)
    print('After reshape:\n')
    print('x_train_shape: ', x_train.shape)
    print('y_train_shape: ', y_train.shape)
    print('x_test_shape: ', x_test.shape)
    print('y_test_shape: ', y_test.shape)
    print('x_train type: ', type(x_train))
    # raise

    # 2. Split into train / validation partitions
    dataset = torch.utils.data.TensorDataset(x_train, y_train, parameter_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test, parameter_test)
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))



    n_test = len(test_dataset)
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train, parameter_train), batch_size=batch_size, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test, parameter_test), batch_size=batch_size, shuffle=False)
    loader_args = dict(batch_size=batch_size)
    trainloader = DataLoader(train_set, shuffle=True, **loader_args)
    valloader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    # testloader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)    
    testloader = DataLoader(test_dataset, shuffle=False, drop_last=True, batch_size=100)    

    return trainloader, valloader, testloader, x_normalizer, y_normalizer, x_test, y_test, parameter_test, n_train, n_val, n_test
    # return trainloader, testloader, testloader, x_normalizer, y_normalizer, x_test, y_test, n_train, n_val, n_test

def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))


def test_net(model,
              device,
              train_loader,
              test_loader,
              x_normalizer,
              y_normalizer,
              batch_size: int = 20,
              s: int=80,
              ntest: int=1000
              ):
    with torch.no_grad():
        true = []
        pred = []
        test_loss = 0

        for x, y, param in test_loader:
            x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)
            # print('x, y, param: ', x.shape, y.shape, param.shape)
            # print('model(x, param): ', model(x, param).shape)
            # raise
            out = model(x, param).reshape(-1, s, s)

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
            test_loss += loss.item()


            # print('y shape: ', y.detach().cpu().numpy().shape)
            # raise
            true.extend(y.detach().cpu().numpy())
            pred.extend(out.detach().cpu().numpy())
        print(f'Test Loss: {test_loss/ntest} for ntest: {ntest}')
        true = np.asarray(true)
        pred = np.asarray(pred)

        mae = np.mean(np.abs(true - pred))
        # idx_ls = np.random.choice(true.shape[0], 10)
        idx_ls = np.arange(10)
        # print("true:", true.shape)
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
        # print('true.max: ', true.max())
        # raise
        
        rel_l2 = np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 
        mape_error = mape(true, pred)
        print('true, pred: ', true.max(), pred.max())
        print('mae: ', mae)
        print('rel_l2: ', rel_l2)
        print('mape_error: ', mape_error)
        with open('./final_test_l2.txt', 'w') as f:
            f.write(f'mae is: {mae} \n')
            f.write(f'relative l2 is: {rel_l2} \n')
            f.write(f'mape_error is: {mape_error} \n')
    model.train()
    return test_loss/ntest

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
    device = torch.device(f'cuda:{args.gpu}')

    s = 80

    train_loader, val_loader, test_loader, x_normalizer, y_normalizer, x_test, y_test, parameter_test, ntrain, nval, ntest  = get_dataLoader(batch_size, s)

    # ntrain = 5000
    # ntest = 1000
    

    step_size = 100
    gamma = 0.5

    modes = 12
    width = 32

    model = FNO2d(modes, modes, width).cuda(DEVICE_NUM)
    print(count_params(model))

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = LpLoss(size_average=False)
    y_normalizer.cuda(DEVICE_NUM)
    # min_test_l2 = 10**10

    model_path = "./checkpoint/network.pt"
    file_exists = exists(model_path)
    if file_exists:
        print('model exist')
        model.load_state_dict(torch.load(model_path, map_location=device))
        num_params = count_params(model)
        print(f'model loaded, num_params: {num_params}')

    if mode == 'train':
        min_test_l2 = test_net(model,
              DEVICE_NUM,
              train_loader,
              val_loader,
              x_normalizer,
              y_normalizer,
              batch_size,
              s,
              nval
              )
        # min_test_l2 = 0.02

        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            train_l2 = 0
            for x, y, param in train_loader:
                x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)

                optimizer.zero_grad()
                # print('model(x, param)', model(x, param).shape)
                out = model(x, param).reshape(-1, s, s)

                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)

                loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
                loss.backward()

                optimizer.step()
                train_l2 += loss.item()

            scheduler.step()

            model.eval()

            val_l2 = 0.0
            with torch.no_grad():
                for x, y, param in val_loader:
                    x, y, param = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM), param.cuda(DEVICE_NUM)

                    out = model(x, param).reshape(-1, s, s)

                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)

                    val_loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
                    # print('val_loss: ', val_loss)
                    val_l2 += val_loss


                # if test_l2 < min_test_l2:
                #     # x, y = x_test[-1:].cuda(DEVICE_NUM), y_test[-1:].cuda(DEVICE_NUM)
                #     # pred = y_normalizer.decode(model(x))
                #     pred = out[-1]
                #     true = y[-1]
                #     # print('pred shape: ', pred.shape)
                #     # print('y shape: ', y.shape)
                #     # raise
                #     pred = pred.detach().cpu().numpy()
                #     true = true.detach().cpu().numpy()
                #     # print('pred shape: ', pred.shape)
                #     # print('true shape: ', true.shape)
                #     fig = plt.figure(figsize=(15, 5))
                #     plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
                #     ax = fig.add_subplot(131)
                #     ax.set_title(f'Truth')
                #     im = ax.imshow(true, origin='lower', cmap='jet')
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes("right", size="7%", pad="2%")
                #     cb = fig.colorbar(im, cax=cax)

                #     ax = fig.add_subplot(132)
                #     im = ax.imshow(pred, cmap='jet', origin='lower')
                #     ax.set_title(f'Pred')
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes("right", size="7%", pad="2%")
                #     cb = fig.colorbar(im, cax=cax)

                #     ax = fig.add_subplot(133)
                #     im = ax.imshow(abs(pred - true), cmap='jet', origin='lower')
                #     ax.set_title(f'Error')
                #     divider = make_axes_locatable(ax)
                #     cax = divider.append_axes("right", size="7%", pad="2%")
                #     cb = fig.colorbar(im, cax=cax)
                #     plt.savefig(f'./figs/{ep}.jpg')
                #     # plt.show()
                #     plt.close()
                #     min_test_l2 = test_l2
                #     torch.save(model.state_dict(), "./checkpoint/network.pt")



            train_l2/= ntrain
            # test_l2 /= ntest
            val_l2 /= nval
            # print('ntrain, ntest, nval: ', ntrain, ntest, nval)
            # raise
            t2 = default_timer()
            # print(ep, t2-t1, train_l2, test_l2)
            print(f'Epochs: {ep}, time: {t2-t1}, train_l2: {train_l2}, val_l2: {val_l2}')

            with torch.no_grad():
                if val_l2 < min_test_l2:

                    # x, y = x.cuda(DEVICE_NUM), y.cuda(DEVICE_NUM)
                    # out = model(x).reshape(batch_size, s, s)
                    # out = y_normalizer.decode(out)
                    idx = np.random.choice(len(x_test))
                    x, y, param = x_test[idx:idx+1].cuda(DEVICE_NUM), y_test[idx:idx+1].cuda(DEVICE_NUM), parameter_test[idx:idx+1].cuda(DEVICE_NUM)
                    # print('model(x, param) shape: ', model(x, param).shape)
                    # raise
                    # print('x,y,param: ', x.shape, y.shape, param.shape)
                    # raise
                    pred = model(x, param).reshape(1, s, s)

                    pred = y_normalizer.decode(pred)
                    y = y_normalizer.decode(y)

                    pred = pred.detach().cpu().numpy()
                    true = y.detach().cpu().numpy()
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    fig = plt.figure(figsize=(15, 5))
                    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
                    ax = fig.add_subplot(131)
                    ax.set_title(f'Truth')
                    im = ax.imshow(true[0,:,:], origin='lower', cmap='jet')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(132)
                    im = ax.imshow(pred[0,:,:], cmap='jet', origin='lower')
                    ax.set_title(f'Pred')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)

                    ax = fig.add_subplot(133)
                    im = ax.imshow(abs(pred[0,:,:] - true[0,:,:]), cmap='jet', origin='lower')
                    ax.set_title(f'Error')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="7%", pad="2%")
                    cb = fig.colorbar(im, cax=cax)
                    plt.savefig(f'./figs/{idx}.jpg')
                    # plt.show()
                    plt.close()

                    print(f'Val loss improved from {min_test_l2} to {val_l2}')

                    min_test_l2 = val_l2
                    print('Saving model...')
                    torch.save(model.state_dict(), "./checkpoint/network.pt")
                    # print('pred shape: ', pred.shape)
                    # print('true shape: ', true.shape)
                    # raise
                    # network.load_state_dict(torch.load("./checkpoint/model.pt"))
    elif mode == 'test':
        model.eval()

        test_net(model,
              DEVICE_NUM,
              train_loader,
              test_loader,
              x_normalizer,
              y_normalizer,
              batch_size,
              s,
              ntest
              )

        # test_net(model,
        #       DEVICE_NUM,
        #       train_loader,
        #       test_loader,
        #       x_normalizer,
        #       y_normalizer,
        #       batch_size,
        #       ntest
        #       )