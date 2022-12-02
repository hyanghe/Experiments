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

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_data = './data/'


def train_net(net,
              device,
              epochs: int = 5000,
              batch_size: int = 128,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # print('dataset: ', dataset)
    # raise

    # train_x = dataset.load_power(dir_data + 'x_train.npy')
    train_x = np.load(dir_data + 'x_train.npy')
    parameter_train = np.loadtxt(dir_data + 'parameters_train.txt', skiprows=1)
    parameter_train[:, 0] = np.log(parameter_train[:, 0]*10**7) * 0.1
    parameter_train[:, 1] = (parameter_train[:, 1] - 20.0) / (200.0 - 20.0)
    train_y = np.load(dir_data + 'y_train.npy')

##### Local shifting ####
    print('train_y shape: ', train_y.shape, train_y.max())
    train_y_local = []
    for tr_y in train_y:
        tr_y = tr_y - tr_y.min()
        train_y_local.append(tr_y)
    train_y = np.asarray(train_y_local)
    print('train_y shape: ', train_y.shape, train_y.max())
##### Local shifting ####

    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    parameter_train = parameter_train[idx_train]

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


    # test_x = dataset.load_power(dir_data + 'x_test.npy')
    test_x = np.load(dir_data + 'x_test.npy')
    parameter_test = np.loadtxt(dir_data + 'parameters_test.txt', skiprows=1)
    parameter_test[:, 0] = np.log(parameter_test[:, 0]*10**7) * 0.1
    parameter_test[:, 1] = (parameter_test[:, 1] - 20.0) / (200.0 - 20.0)
    # test_y = dataset.load_power(dir_data + 'y_test.npy')
    test_y = np.load(dir_data + 'y_test.npy')

##### Local shifting ####
    print('test_y shape: ', test_y.shape, test_y.max())
    test_y_local = []
    for te_y in test_y:
        te_y = te_y - te_y.min()
        test_y_local.append(te_y)
    test_y = np.asarray(test_y_local)
    print('test_y shape: ', test_y.shape, test_y.max())
    # raise
##### Local shifting ####

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]
    parameter_test = parameter_test[idx_test]

    test_x = (test_x - Power_max) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    parameter_test = torch.from_numpy(parameter_test)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)


    # print('parameter_train: ', parameter_train.min(), parameter_train.max())
    # print('parameter_test: ', parameter_test.min(), parameter_test.max())
    # raise
    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    train_x = torch.permute(train_x, (0,3,1,2))
    train_y = torch.permute(train_y, (0,3,1,2))

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    test_x = torch.permute(test_x, (0,3,1,2))
    test_y = torch.permute(test_y, (0,3,1,2))
    # raise
    # train_x = np.load(dir_data + 'x_train.npy')
    # train_y = np.load(dir_data + 'y_train.npy')

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    print('parameter_train shape: ', parameter_train.shape)
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
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
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
                params = batch[2]


                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                params = params.to(device=device, dtype=torch.float32)

                # with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images, params)
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
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        # histograms = {}
                        # for tag, value in net.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(net, val_loader, device)
                        # scheduler.step(val_score)
                        evaluate(net, val_loader, device, T_max, T_min)
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
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)


    train_x = dataset.load_power(dir_data + 'x_train.npy')
    train_y = dataset.load_power(dir_data + 'y_train.npy')
    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()


    test_x = dataset.load_power(dir_data + 'x_train_100.npy')
    test_y = dataset.load_power(dir_data + 'y_train_100.npy')
    # test_x = dataset.load_power(dir_data + 'x_train.npy')[:100]
    # test_y = dataset.load_power(dir_data + 'y_train.npy')[:100]
    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    test_x = torch.permute(test_x, (0,3,1,2))
    test_y = torch.permute(test_y, (0,3,1,2))

    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)
    evaluate(net, test_loader, device, T_max, T_min)



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default='./checkpoints/checkpoint.pth', help='Load model from a .pth file')
    # parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--outChannel', '-c', type=int, default=1, help='Number of output channels')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=1, n_classes=args.outChannel, bilinear=args.bilinear, n_params=2)
    net = UNet_param(n_channels=1, n_classes=args.outChannel, bilinear=args.bilinear, n_params=2)
    

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.outChannel} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

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
