'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary
import os
import argparse
from tqdm import tqdm
# from models_params import *
from models_htc import *
from utils import progress_bar
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true',
#                     help='resume from checkpoint')
# args = parser.parse_args()

import matplotlib
matplotlib.use('Agg') 

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
best_acc = 10**10  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

def mae(true, pred, T_min, T_max):
    print('pred true', pred.shape, true.shape)
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))



# Data
def get_data():
    print('==> Preparing data..')
    val_percent = 0.1
    batch_size = 100
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # trainset = torchvision.datasets.CIFAR10(
    #     root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=128, shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(
    #     root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(
    #     testset, batch_size=100, shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer',
    #            'dog', 'frog', 'horse', 'ship', 'truck')

    dir_data = './data/'
    train_x = np.load(dir_data + 'x_train.npy').astype(np.float32)
    train_y = np.load(dir_data + 'y_train.npy').astype(np.float32)
    train_htc_top = np.load(dir_data + 'htc_top_train.npy').astype(np.float32)
    train_htc_btm = np.load(dir_data + 'htc_btm_train.npy').astype(np.float32)

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_htc_top = train_htc_top * 10**6
    train_htc_top_min = train_htc_top.min()
    train_htc_top_max = train_htc_top.max()
    train_htc_top = (train_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    train_htc_btm = train_htc_btm * 10**7
    train_htc_btm_min = train_htc_btm.min()
    train_htc_btm_max = train_htc_btm.max()
    train_htc_btm = (train_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)




    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)
    train_htc_top = np.expand_dims(train_htc_top, -1)
    train_htc_btm = np.expand_dims(train_htc_btm, -1)
    train_htc_top = train_htc_top.astype(np.float32)
    train_htc_btm = train_htc_btm.astype(np.float32)
    train_htc_top = torch.from_numpy(train_htc_top)
    train_htc_btm = torch.from_numpy(train_htc_btm)

    train_x = torch.cat((train_x, train_htc_top, train_htc_btm), dim=-1)

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    # print('train_x.dtype: ', train_x.dtype)
    # raise
    # train_x = torch.permute(train_x, (0,3,1,2))
    train_x = train_x.permute((0,3,1,2))
    # train_y = torch.permute(train_y, (0,3,1,2))
    train_y = train_y.permute((0,3,1,2))



    test_x = np.load(dir_data + 'x_test.npy').astype(np.float32)
    test_y = np.load(dir_data + 'y_test.npy').astype(np.float32)
    test_htc_top = np.load(dir_data + 'htc_top_test.npy').astype(np.float32)
    test_htc_btm = np.load(dir_data + 'htc_btm_test.npy').astype(np.float32)


    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)
    test_htc_top = test_htc_top * 10**6
    test_htc_top = (test_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    test_htc_btm = test_htc_btm * 10**7
    test_htc_btm = (test_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)


    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)
    test_htc_top = np.expand_dims(test_htc_top, -1)
    test_htc_btm = np.expand_dims(test_htc_btm, -1)
    test_htc_top = test_htc_top.astype(np.float32)
    test_htc_btm = test_htc_btm.astype(np.float32)
    test_htc_top = torch.from_numpy(test_htc_top)
    test_htc_btm = torch.from_numpy(test_htc_btm)



    test_x = torch.cat((test_x, test_htc_top, test_htc_btm), dim=-1)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    # test_x = torch.permute(test_x, (0,3,1,2))
    test_x = test_x.permute((0,3,1,2))
    # test_y = torch.permute(test_y, (0,3,1,2))
    test_y = test_y.permute((0,3,1,2))

    # print('train_x, train_y, parameter_train: ', train_x.shape, train_y.shape, parameter_train.shape)
    # raise

    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
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
    return trainloader, valloader, testloader, T_min, T_max


# Training
def train(net, epoch, trainloader, valloader, optimizer):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    # print('trainloader len:', len(trainloader))
    # print('batch_size: ', trainloader.batch_size)
    num_train = len(trainloader) * trainloader.batch_size
    # raise
    train_loss = 0
    val_loss = 0
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
    # for batch in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        # print('inputs, targets, params: ', inputs.shape, targets.shape, params.shape)
        # raise
        optimizer.zero_grad()
        # print('inputs: ', inputs.dtype)
        # # raise
        outputs = net(inputs)
        # print('outputs, targets: ', outputs.shape, targets.shape)
        # raise
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += targets.size(0)
        # correct += predicted.eq(targets).sum().item()
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        true = targets.detach().cpu().numpy()
        pred = outputs.detach().cpu().numpy()
        true = np.asarray(true)
        pred = np.asarray(pred)

        mae = np.mean(np.abs(true - pred))
        idx = np.random.choice(true.shape[0])
    print(f'Train Loss: {train_loss/num_train}')




    net.eval()
    num_val = len(valloader) * valloader.batch_size
    for batch_idx_val, (inputs_val, targets_val) in tqdm(enumerate(valloader)):
    # for batch in trainloader:
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        optimizer.zero_grad()
        outputs_val = net(inputs_val)
        loss_val = criterion(outputs_val, targets_val)

        val_loss += loss_val.item()

        true_val = targets_val.detach().cpu().numpy()
        pred_val = outputs_val.detach().cpu().numpy()
        true_val = np.asarray(true_val)
        pred_val = np.asarray(pred_val)

        mae_val = np.mean(np.abs(true_val - pred_val))
        idx_val = np.random.choice(true_val.shape[0])
    print(f'Val Loss: {val_loss/num_val}')

    # Save checkpoint.
    cur_val_loss = val_loss/num_val
    if cur_val_loss < best_acc:
        print(f'Val loss improved from {best_acc} to {cur_val_loss}')
        print('Saving models..')
        state = {
            'net': net.state_dict(),
            'loss': cur_val_loss,
            # 'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = cur_val_loss
        print('Model saved')

    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
    ax = fig.add_subplot(131)
    ax.set_title(f'Truth')
    im = ax.imshow(true[idx,0, :,:], origin='lower', cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(132)
    im = ax.imshow(pred[idx,0, :,:], cmap='jet', origin='lower')
    ax.set_title(f'Pred')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(133)
    im = ax.imshow(abs(pred[idx,0, :,:] - true[idx,0, :,:]), cmap='jet', origin='lower')
    ax.set_title(f'Error')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)
    # plt.savefig(f'./figs/{cnt}.jpg')
    # plt.savefig(f'./figs/final_test_sample_{idx}.jpg')
    plt.savefig(f'./figs/train_sample_{idx}.jpg')
    # plt.show()
    plt.close()

    net.train()

def test(net, testloader, T_min, T_max):
    # global best_acc
    net.eval()
    test_loss = 0
    num_test = len(testloader) * testloader.batch_size
    # correct = 0
    # total = 0
    true = []
    pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
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
    true = np.asarray(true)
    pred = np.asarray(pred)

    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min

    # idxs = np.random.choice(true.shape[0], 10)
    idxs = np.arange(10)
    for idx in idxs:
        fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
        ax = fig.add_subplot(131)
        ax.set_title(f'Truth')
        im = ax.imshow(true[idx,0, :,:], origin='lower', cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(132)
        im = ax.imshow(pred[idx,0, :,:], cmap='jet', origin='lower')
        ax.set_title(f'Pred')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im, cax=cax)

        ax = fig.add_subplot(133)
        im = ax.imshow(abs(pred[idx,0, :,:] - true[idx,0, :,:]), cmap='jet', origin='lower')
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
    mae_error = mae(true, pred, T_min, T_max)
    mape_error = mape(true, pred, T_min, T_max)
    rel_l2 = relativeL2(true, pred, T_min, T_max)
    print('mae: ', mae_error)
    print('rel_l2: ', rel_l2)
    print('mape_error: ', mape_error)
    with open('./final_test_l2.txt', 'w') as f:
        f.write(f'mae is: {mae} \n')
        f.write(f'relative l2 is: {rel_l2} \n')
        f.write(f'mape_error is: {mape_error} \n')
    net.train()


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
    parser = argparse.ArgumentParser(description='ResNet')
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
    device = torch.device(f'cuda:{args.gpu}')
    load_model = args.load_model

    trainloader, valloader, testloader, T_min, T_max = get_data()

    # Model
    print('==> Building model..')
    # from unet import UNet
    # net = UNet(n_channels=1, n_classes=1)
    # net = VGG('VGG19')
    net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()

    net = net.to(device)
    # summary(net, (1, 80, 80))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr,
    #                       momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.SGD(net.parameters(), lr=1e-4,
    #                     momentum=0.9, weight_decay=5e-4)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if mode == 'train':
        print('Mode: ', mode)
        resume_training = load_model
        print('resume_training: ', resume_training)
        # raise
        if resume_training:
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./checkpoint/ckpt.pth')
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['loss']
            # start_epoch = checkpoint['epoch']       
        # if args.resume:
        #     # Load checkpoint.
        #     print('==> Resuming from checkpoint..')
        #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        #     checkpoint = torch.load('./checkpoint/ckpt.pth')
        #     net.load_state_dict(checkpoint['net'])
        #     best_acc = checkpoint['acc']
        #     start_epoch = checkpoint['epoch']


        for epoch in range(start_epoch, start_epoch+epochs):
            train(net, epoch, trainloader, valloader, optimizer)
    # for epoch in range(start_epoch, start_epoch+200):
    #     train(epoch, trainloader)
            # test(net, epoch, testloader)
            # test(net, valloader)
            # scheduler.step()
    elif mode == 'test':
        print('==> Loading from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(net, testloader, T_min, T_max)
