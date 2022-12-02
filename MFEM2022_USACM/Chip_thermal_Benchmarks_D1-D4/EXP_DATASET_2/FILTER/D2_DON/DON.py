import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from tqdm import tqdm
import os
import matplotlib
from functools import reduce
import operator
import argparse
matplotlib.use('Agg') 

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


def mape(true, pred, T_min, T_max): 
    true, pred = np.array(true), np.array(pred)
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    mask = true != 0
    return np.mean(np.abs((true - pred) / true)[mask])

def relativeL2(true, pred, T_min, T_max):
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    # return np.linalg.norm(true.flatten() - pred.numpy().flatten()) / np.linalg.norm(true.flatten()) 
    return np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 

def mae_cal(true, pred, T_min, T_max):
    # print('pred true shape', pred.shape, true.shape)
    # raise
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))


class deepONet(nn.Module):
    def __init__(self, in_channels=1, mid_channels=32, in_features=2, mid_features=32, activation=nn.ReLU()):
        super().__init__()
        self.dropout = 0.00
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels = mid_channels, kernel_size = 3, padding=1),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            activation,
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            activation,
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            activation,
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            activation,
            nn.Dropout(self.dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1),
            activation,
        )

        self.trunk = nn.Sequential(
            nn.Linear(in_features=in_features, out_features = mid_features),
            # nn.BatchNorm1d(mid_channels),
            activation,
            nn.Dropout(self.dropout),

            nn.Linear(in_features=mid_features, out_features = mid_features),
            # nn.BatchNorm1d(mid_channels),
            activation,
            nn.Dropout(self.dropout),

            nn.Linear(in_features=mid_features, out_features = mid_features),
            # nn.BatchNorm1d(mid_channels),
            activation,
            nn.Dropout(self.dropout),

            nn.Linear(in_features=mid_features, out_features = int(80 / 2**(len(self.branch)//5))**2 * mid_channels),  
            activation,          
        )

        self.merge = nn.Sequential(
            nn.Linear(in_features=int(80 / 2**(len(self.branch)//5))**2 * mid_channels,\
                out_features=80**2),
            activation,
            nn.Linear(in_features=80**2,\
                out_features=80**2),
            # activation
        )

        self.outConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding='same'),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding='same'),
            )
    
    def forward(self, x1, x2):
        x1 = self.branch(x1)
        x1 = torch.flatten(x1, 1)
        x2 = self.trunk(x2)
        out = torch.mul(x1, x2)
        out = self.merge(out)
        out = torch.reshape(out, (-1, 1, 80, 80))
        # out = self.outConv(out)
        return out



def get_data():
    data_dir = './data/'
    train_x = np.load(data_dir + 'x_train.npy')
    train_y = np.load(data_dir + 'y_train.npy')
    # print('train_x, train_y: ', train_x.shape, train_y.shape)
    # raise

    parameter_train = np.loadtxt(data_dir + 'parameters_train.txt', skiprows=1)


    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    parameter_train = parameter_train[idx_train]

    ##### HTC filtering and scaling ####
    idx_htc7 = parameter_train[:, 0] >= 1e-6
    train_x = train_x[idx_htc7]
    parameter_train = parameter_train[idx_htc7]
    train_y = train_y[idx_htc7]
    # print(train_y.max())
    # print('train_x shape: ', train_x.shape)
    # raise
    ##### HTC filtering and scaling ####

    parameter_train[:, 0] = np.log(parameter_train[:, 0]*10**7) * 0.1
    parameter_train[:, 1] = (parameter_train[:, 1] - 20.0) / (200.0 - 20.0)


    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    parameter_train = torch.from_numpy(parameter_train)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)






    test_x = np.load(data_dir + 'x_test.npy')
    parameter_test = np.loadtxt(data_dir + 'parameters_test.txt', skiprows=1)
    test_y = np.load(data_dir + 'y_test.npy')

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]
    parameter_test = parameter_test[idx_test]

##### HTC filtering and scaling ####
    idx_htc7 = parameter_test[:, 0] >= 1e-6
    test_x = test_x[idx_htc7]
    parameter_test = parameter_test[idx_htc7]
    test_y = test_y[idx_htc7]

    # temp, power, htc, die_z = test_y, test_x, parameter_test[:, 0], parameter_test[:, 1]
    # test_y = scaling_strategy3(temp, power, htc, die_z)  
##### HTC filtering and scaling ####

    # print('test_x shape: ', test_x.shape)
    # raise

    parameter_test[:, 0] = np.log(parameter_test[:, 0]*10**7) * 0.1
    parameter_test[:, 1] = (parameter_test[:, 1] - 20.0) / (200.0 - 20.0)


    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    parameter_test = torch.from_numpy(parameter_test)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    # train_x = torch.permute(train_x, (0,3,1,2))
    # train_y = torch.permute(train_y, (0,3,1,2))
    train_x = train_x.permute((0,3,1,2))
    train_y = train_y.permute((0,3,1,2))

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    # test_x = torch.permute(test_x, (0,3,1,2))
    # test_y = torch.permute(test_y, (0,3,1,2))
    test_x = test_x.permute((0,3,1,2))
    test_y = test_y.permute((0,3,1,2))

    print('train_x, train_y: ', train_x.shape, train_y.shape)
    # raise
    dataset = torch.utils.data.TensorDataset(train_x, train_y, parameter_train)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y, parameter_test)
    val_percent = 0.1
    n_val = int(len(dataset) * val_percent)

    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    batch_size = 100
    loader_args = dict(batch_size=batch_size)
    train_loader , val_loader = DataLoader(train_set, shuffle=True, **loader_args),\
                                DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    return train_loader, val_loader, test_loader, T_min, T_max


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def save_checkpoint(state, filename='model.pth'):
	print('=> Saving checkpoint')
	torch.save(state, filename)

def load_checkpoint(checkpoint):
	print('=> Loading checkpoint')
	net.load_state_dict(checkpoint['state_dict'])
	# optimizer.load_state_dict(checkpoint['optimizer'])

def validate(net, dataloader, device, mx, mn):
    net.eval()
    num_val_batches = len(dataloader)
    cnt = 0
    mses = []
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, params = batch[0], batch[1], batch[2]
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        params = params.to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            mask_pred = net(image, params)
            # print('mask_pred shape: ', mask_pred.shape)
            # print('mask_true shape: ', mask_true.shape)
            # raise
            pred = mask_pred.detach().cpu().numpy()
            true = mask_true.detach().cpu().numpy()
            # print('pred shape: ', pred.shape)
            # print('true shape: ', true.shape)
            # print('mx: ', mx)
            # raise
            # mx = mx.numpy()
            # mn = mn.numpy()
            mae = np.mean(np.abs(true - pred))
            mse = np.square(true-pred).mean()
            mses.append(mse)

            pred = pred * (mx - mn) + mn
            true = true * (mx - mn) + mn

            pred = np.squeeze(pred)
            true = np.squeeze(true)


            idx = np.random.choice(mask_pred.shape[0])
            # print('pred shape: ', pred.shape)
            # print('true shape: ', true.shape)
            # raise
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
            plt.savefig(f'./figs/{idx}.jpg')
            # plt.show()
            plt.close()
            cnt += 1
            print('mae: ', mae)
    net.train()
    return sum(mses) / (len(mses)*dataloader.batch_size)

def test(net, testloader, device, T_min, T_max):
    # global best_acc
    net.eval()
    test_loss = 0
    num_test = len(testloader) * testloader.batch_size
    # correct = 0
    # total = 0
    true = []
    pred = []
    with torch.no_grad():
        for batch in tqdm(testloader, desc='Testing round', unit='batch', leave=False):
            inputs_image, targets, inputs_params = batch[0], batch[1], batch[2]
            inputs_image = inputs_image.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            inputs_params = inputs_params.to(device=device, dtype=torch.float32)

            outputs = net(inputs_image, inputs_params)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

            true.extend(targets.detach().cpu().numpy())
            pred.extend(outputs.detach().cpu().numpy())

            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(f'Test Loss: {test_loss/num_test}')


    # true = targets.detach().cpu().numpy()
    # pred = outputs.detach().cpu().numpy()
    true = np.squeeze(np.asarray(true))
    pred = np.squeeze(np.asarray(pred))

    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min

    print('true: ', true.shape)
    print('pred: ', pred.shape)
    # raise
    # idxs = np.random.choice(true.shape[0], 10)
    idxs = np.arange(10)
    for idx in idxs:
        fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
        ax = fig.add_subplot(131)
        ax.set_title(f'Truth')
        im = ax.imshow(true[idx,:,:], origin='lower', cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(132)
        im = ax.imshow(pred[idx,:,:], cmap='jet', origin='lower')
        ax.set_title(f'Pred')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(133)
        im = ax.imshow(abs(pred[idx,:,:] - true[idx,:,:]), cmap='jet', origin='lower')
        ax.set_title(f'Error')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)
        # plt.savefig(f'./figs/{cnt}.jpg')
        # plt.savefig(f'./figs/final_test_sample_{idx}.jpg')
        plt.savefig(f'./figs/final_test_sample_{idx}.jpg')
        # plt.show()
        plt.close()

    # mae = np.mean(np.abs(true - pred))
    # rel_l2 = np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 
    # print('mae: ', mae)
    # print('rel_l2: ', rel_l2)
    mae_error = mae_cal(true, pred, T_min, T_max)
    mape_error = mape(true, pred, T_min, T_max)
    rel_l2 = relativeL2(true, pred, T_min, T_max)
    print('mae: ', mae_error)
    print('rel_l2: ', rel_l2)
    print('mape_error: ', mape_error)
    with open('./final_test_l2.txt', 'w') as f:
        f.write(f'mae is: {mae_error} \n')
        f.write(f'relative l2 is: {rel_l2} \n')
        f.write(f'mape_error is: {mape_error} \n')
    net.train()
    return test_loss/num_test

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50000, help='Number of epochs')
    # parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--mode', '-m', type=str, default='test', help='train or test')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index')
    parser.add_argument("--load", '-l', type=str2bool, nargs='?',
                        const=True, default=True,dest='load_model',
                        help="Load or retrain")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    DEVICE_NUM = args.gpu
    # batch_size = args.batch_size
    learning_rate = args.lr
    epochs = args.epochs
    mode = args.mode
    load_model = args.load_model

    device = torch.device(f'cuda:{DEVICE_NUM}')

    trainloader, valloader, testloader, T_min, T_max = get_data()

    net = deepONet()

    Epochs = 50000
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    net = net.to(device=device)

    print('Num of params: ', count_params(net))

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',\
    #     factor=0.1, patience=10)
    criterion = nn.MSELoss()

    if load_model:
        print('Loading model...')
        assert os.path.exists('model.pth'), 'Error: no checkpoint directory found!'
        load_checkpoint(torch.load('model.pth', map_location=device))
        print('Model loaded...')

    if mode == 'train':
        cur_min_loss = test(net, testloader, device, T_min, T_max)
        for epoch in range(Epochs+1):
            net.train()
            losses = []
            # if epoch % 3 == 0:
            #     checkpoint = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            #     save_checkpoint(checkpoint)
            for batch in tqdm(trainloader):
                power, T, param = batch[0], batch[1], batch[2]
                power = power.to(device = device, dtype=torch.float32)
                T = T.to(device=device, dtype=torch.float32)
                param = param.to(device, dtype=torch.float32)

                pred = net(power, param)
                true = T 
                loss = criterion(pred, true)
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_loss = validate(net, valloader, device, T_max, T_min)
            if val_loss < cur_min_loss:
                print(f'Val loss improved from {cur_min_loss} to {val_loss}')
                print('Saving models..')
                checkpoint = {'state_dict': net.state_dict()}
                save_checkpoint(checkpoint)

                cur_min_loss = val_loss
                print('Model saved')
            # scheduler.step(val_loss)
            # print(f'Epoch: {epoch}, loss: {loss}, lr: {get_lr(optimizer)}')
            # print('trainloader: ', trainloader.batch_size)
            # print('len(losses): ', len(losses))
            # raise
            mean_loss = sum(losses) / len(losses) / trainloader.batch_size
            print(f'Epoch: {epoch}, Train loss: {mean_loss}, val loss: {val_loss}, lr: {get_lr(optimizer)}')
    elif mode == 'test':
        # print('==> Loading from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        # checkpoint = torch.load('./checkpoint/ckpt.pth')
        # net.load_state_dict(checkpoint['net'])
        test(net, testloader, device, T_min, T_max)