import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
from evaluate import evaluate
# from unet import UNet
from unet import UNet_param
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib
from functools import reduce
import operator
matplotlib.use('Agg') 


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()+(2,) if p.is_complex() else p.size()))
    return c


device_list = [0, 1]

# dir_img = Path('./data_random_powermap_and_params/imgs/')
# dir_mask = Path('./data_random_powermap_and_params/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_data = './data/'
BATCH_SIZE = 20
def scaling_strategy3(temp, power, htc, die_z, total_power=50000.0):
    power_ratio = []
    for j in range(power.shape[0]):
        power_ratio.append(total_power/power[j])
    power_ratio = np.asarray(power_ratio)

    # for j in range(temp.shape[0]):
    #     temp[j] = temp[j] * power_ratio[j]

    for j in range(temp.shape[0]):
        # temp[j] = temp[j] / (10**(htc[j])+4.0)
        # print('scale: ', (die_z[j]/100.0)**(1.0/3.0)/ -((np.log10(htc[j])+2.0)**3.0))
        temp[j] = (die_z[j]/100.0)**(1.0/3.0) * temp[j] / -((np.log10(htc[j])+2.0)**3.0)
    print('temp: ', temp.shape)
    return temp



def unscaling_strategy3(temp, power, htc, die_z, total_power=50000.0):

    temp = np.squeeze(temp)
    power = np.squeeze(power).numpy()
    # print('temp shape: ', temp.shape)
    # print('power shape: ', power.shape)
    # raise
    power_ratio = []
    for j in range(power.shape[0]):
        power_ratio.append(total_power/power[j])
    power_ratio = np.asarray(power_ratio)

    for j in range(temp.shape[0]):
        temp[j] = temp[j] * -((np.log10(htc[j])+2.0)**3.0) / (die_z[j]/100.0)**(1.0/3.0)

    # for j in range(temp.shape[0]):
    #     temp[j] = temp[j] / power_ratio[j]

    return temp


def train_net(net,
              device,
              epochs: int = 5000,
              batch_size: int = BATCH_SIZE,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # print('dataset: ', dataset)
    # raise
    data_dir = './data/'
    
    # train_x = dataset.load_power(dir_data + 'x_train.npy')
    train_x = np.load(data_dir + 'x_train.npy')
    train_y = np.load(data_dir + 'y_train.npy')
    train_htc_top = np.load(data_dir + 'htc_top_train.npy')
    train_htc_btm = np.load(data_dir + 'htc_btm_train.npy')
    train_htc_top = train_htc_top * 10**6
    train_htc_top_min = train_htc_top.min()
    train_htc_top_max = train_htc_top.max()
    train_htc_top = (train_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    train_htc_btm = train_htc_btm * 10**7
    train_htc_btm_min = train_htc_btm.min()
    train_htc_btm_max = train_htc_btm.max()
    train_htc_btm = (train_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)
    print('train_x max, min: ', train_x.max(), train_x.min())
    print('train_y max, min: ', train_y.max(), train_y.min())
    print('train_htc_top min, max: ', train_htc_top.min(), train_htc_top.max())
    print('train_htc_btm min, max: ', train_htc_btm.min(), train_htc_btm.max())
    

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    # parameter_train = torch.from_numpy(parameter_train)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)


    train_htc_top = np.expand_dims(train_htc_top, -1)
    train_htc_btm = np.expand_dims(train_htc_btm, -1)
    train_htc_top = torch.from_numpy(train_htc_top)
    train_htc_btm = torch.from_numpy(train_htc_btm)

    # test_x = dataset.load_power(dir_data + 'x_test.npy')
    test_x = np.load(dir_data + 'x_test.npy')
    test_y = np.load(dir_data + 'y_test.npy')
    test_htc_top = np.load(data_dir + 'htc_top_test.npy')
    test_htc_btm = np.load(data_dir + 'htc_btm_test.npy')
    test_htc_top = test_htc_top * 10**6
    test_htc_top = (test_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    test_htc_btm = test_htc_btm * 10**7
    test_htc_btm = (test_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)


    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    # parameter_test = torch.from_numpy(parameter_test)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    test_htc_top = np.expand_dims(test_htc_top, -1)
    test_htc_btm = np.expand_dims(test_htc_btm, -1)
    test_htc_top = torch.from_numpy(test_htc_top)
    test_htc_btm = torch.from_numpy(test_htc_btm)


    train_x = torch.cat((train_x, train_htc_top, train_htc_btm), dim=-1)
    test_x = torch.cat((test_x, test_htc_top, test_htc_btm), dim=-1)

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
    # raise
    # train_x = np.load(dir_data + 'x_train.npy')
    # train_y = np.load(dir_data + 'y_train.npy')

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)

    # print('train_htc_top min, max: ', train_htc_top.min(), train_htc_top.max())
    # print('train_htc_btm min, max: ', train_htc_btm.min(), train_htc_btm.max())
    # print('train_x min, max: ', train_x.min(), train_x.max())
    # print('train_y min, max: ', train_y.min(), train_y.max())

    # print('test_htc_top min, max: ', test_htc_top.min(), test_htc_top.max())
    # print('test_htc_btm min, max: ', test_htc_btm.min(), test_htc_btm.max())
    # print('test_x min, max: ', test_x.min(), test_x.max())
    # print('test_y min, max: ', test_y.min(), test_y.max())

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
    loader_args = dict(batch_size=batch_size)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='maps') as pbar:
            for batch in train_loader:
                # print('batch: ', batch[0].shape, batch[1].shape)
                # raise
                # images = batch['image']
                # true_masks = batch['mask']
                # print('batch length: ', len(batch))
                # print('batch[0] shape: ', batch[0].shape)
                # print('batch[1] shape: ', batch[1].shape)
                # print('batch[2] shape: ', batch[2].shape)
                # raise
                images = batch[0]
                true_masks = batch[1]
                # params = batch[2]


                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                # images = images.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.float32)
                # params = params.to(device=device, dtype=torch.float32)

                images = images.to(device=net.device_ids[0], dtype=torch.float32)
                true_masks = true_masks.to(device=net.device_ids[0], dtype=torch.float32)
                # params = params.to(device=net.device_ids[0], dtype=torch.float32)

                # with torch.cuda.amp.autocast(enabled=amp):
                # masks_pred = net(images, params)
                masks_pred = net(images)
                # print('masks_pred.shape: ', masks_pred.shape)
                # print('true_masks.shape: ', true_masks.shape)
                # raise
                loss = criterion(masks_pred, true_masks)
                # print('mask_pred dtype: ', masks_pred.dtype)
                # print('true_masks dtype: ', true_masks.dtype)
                # print('loss dtype: ', loss.dtype)
                # raise
                    # loss = criterion(masks_pred, true_masks) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # grad_scaler.step(optimizer)
                # grad_scaler.update()

                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (10 * batch_size))
                division_step = 1000
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(net, val_loader, device)
                        # scheduler.step(val_score)
                        # evaluate(net, val_loader, device, T_max, T_min)
                        evaluate(net, test_loader, device, T_max, T_min)
                        
                        # logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            # 'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            # **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint.pth'))
            logging.info(f'Checkpoint {epoch} saved!')



def test_net(net,
              device,
              batch_size: int = 100,
              img_scale: float = 0.5,
              ):
    # 1. Create dataset
    # train_x = dataset.load_power(dir_data + 'x_train.npy')
    train_x = np.load(dir_data + 'x_train.npy')
    train_y = np.load(dir_data + 'y_train.npy')

    print('train_y max, min: ', train_y.max(), train_y.min())
    train_htc_top = np.load(dir_data + 'htc_top_train.npy')
    train_htc_btm = np.load(dir_data + 'htc_btm_train.npy')
    train_htc_top = train_htc_top * 10**6
    train_htc_top_min = train_htc_top.min()
    train_htc_top_max = train_htc_top.max()
    train_htc_top = (train_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    train_htc_btm = train_htc_btm * 10**7
    train_htc_btm_min = train_htc_btm.min()
    train_htc_btm_max = train_htc_btm.max()
    train_htc_btm = (train_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)
    print('train_x max, min: ', train_x.max(), train_x.min())
    print('train_y max, min: ', train_y.max(), train_y.min())
    # print('train_htc_top min, max: ', train_htc_top.min(), train_htc_top.max())
    # print('train_htc_btm min, max: ', train_htc_btm.min(), train_htc_btm.max())


    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    # parameter_train = torch.from_numpy(parameter_train)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)

    train_htc_top = np.expand_dims(train_htc_top, -1)
    train_htc_btm = np.expand_dims(train_htc_btm, -1)
    train_htc_top = torch.from_numpy(train_htc_top)
    train_htc_btm = torch.from_numpy(train_htc_btm)


    
    test_x = np.load(dir_data + 'x_test.npy')
    test_y = np.load(dir_data + 'y_test.npy')
    test_htc_top = np.load(dir_data + 'htc_top_test.npy')
    test_htc_btm = np.load(dir_data + 'htc_btm_test.npy')
    test_htc_top = test_htc_top * 10**6
    test_htc_top = (test_htc_top - train_htc_top_min) / (train_htc_top_max - train_htc_top_min)
    test_htc_btm = test_htc_btm * 10**7
    test_htc_btm = (test_htc_btm - train_htc_btm_min) / (train_htc_btm_max - train_htc_btm_min)


    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    test_htc_top = np.expand_dims(test_htc_top, -1)
    test_htc_btm = np.expand_dims(test_htc_btm, -1)
    test_htc_top = torch.from_numpy(test_htc_top)
    test_htc_btm = torch.from_numpy(test_htc_btm)


    train_x = torch.cat((train_x, train_htc_top, train_htc_btm), dim=-1)
    test_x = torch.cat((test_x, test_htc_top, test_htc_btm), dim=-1)

    # print('parameter_train: ', parameter_train.min(), parameter_train.max())
    # print('parameter_test: ', parameter_test.min(), parameter_test.max())
    # raise
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
    # raise
    # train_x = np.load(dir_data + 'x_train.npy')
    # train_y = np.load(dir_data + 'y_train.npy')

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    # print('parameter_train shape: ', parameter_train.shape)
    # raise
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * 0.1)

    n_train = len(dataset) - n_val

    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # print('train_set: ', train_set)
    # print('val_set: ', val_set)
    # raise
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    evaluate(net, test_loader, device, T_max, T_min)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5000*10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default='./checkpoints/checkpoint.pth', help='Load model from a .pth file')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--outChannel', '-c', type=int, default=1, help='Number of output channels')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train or test')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index')
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{args.gpu}')
    logging.info(f'Using device {device}')

    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=1, n_classes=args.outChannel, bilinear=args.bilinear, n_params=2)
    net = UNet_param(n_channels=2, n_classes=args.outChannel, bilinear=args.bilinear, n_params=2)
    

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.outChannel} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # if args.load:
    #     net.load_state_dict(torch.load(args.load, map_location=device))
    #     logging.info(f'Model loaded from {args.load}')

    # net.to(device=device)

    if args.load and args.mode == 'train':
        print('loading model')
        net = nn.DataParallel(net, device_ids = device_list)
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

        num_params = count_params(net)
        print(f'net loaded, num_params: {num_params}')
    elif args.load and args.mode == 'test':
        net = nn.DataParallel(net, device_ids = [device])
        net.load_state_dict(torch.load(args.load, map_location=device))
    else:
        print('torch.cuda.device_count(): ', torch.cuda.device_count())
        # raise
        if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
          # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
          # net = nn.DataParallel(net)
          net = nn.DataParallel(net, device_ids = device_list)

    if args.mode == 'train':
        net.to(f'cuda:{net.device_ids[0]}')
    elif args.mode == 'test':
        print('move net to device: ', device)
        net.to(device=device)



    mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
    mem = mem_params + mem_bufs # in bytes
    print('mem: ', mem/10**6) # in mb
    print('current memory allocated: {}'.format(torch.cuda.memory_allocated() / 1024 ** 2))
    print('max memory allocated: {}'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
    print('cached memory: {}'.format(torch.cuda.memory_cached() / 1024 ** 2))  


    if args.mode == 'train':
        try:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      learning_rate=args.lr,
                      device=device,
                      img_scale=args.scale,
                      val_percent=args.val / 100,
                      amp=args.amp)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            raise
    elif args.mode == 'test':
        print('Start evaluation')
        test_net(net, device)
        print('Evaluation done')
