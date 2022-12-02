import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
from evaluate_pwr_only import evaluate
from unet import UNet
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

device_list = [4, 5, 6, 7]
# dir_img = Path('./data/imgs/')
# dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')
dir_data = './data_random_powermap_only/'


def train_net(net,
              device,
              epochs: int = 5000,
              batch_size: int = 1,
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

    train_x = np.load(dir_data + 'x_train.npy')
    train_y = np.load(dir_data + 'y_train.npy')
##### Local shifting ####
    # print('train_y shape: ', train_y.shape, train_y.max())
    # train_y_local = []
    # for tr_y in train_y:
    #     tr_y = tr_y - tr_y.min()
    #     train_y_local.append(tr_y)
    # train_y = np.asarray(train_y_local)
    # print('train_y shape: ', train_y.shape, train_y.max())
##### Local shifting ####

##### Filter unrealistic cases #####
    idx_train = np.amax(train_y, axis=(1, 2)) < 300
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
##### Filter unrealistic cases #####

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    train_x = (train_x - Power_min) / (Power_max - Power_min)
    train_y = (train_y - T_min) / (T_max - T_min)

    train_x = np.expand_dims(train_x, -1)
    train_x = torch.from_numpy(train_x)
    train_y = np.expand_dims(train_y, -1)
    train_y = torch.from_numpy(train_y)

    print('train_x shape: ', train_x.shape)
    print('train_y shape: ', train_y.shape)
    train_x = torch.permute(train_x, (0,3,1,2))
    train_y = torch.permute(train_y, (0,3,1,2))








    test_x = np.load(dir_data + 'x_test.npy')
    test_y = np.load(dir_data + 'y_test.npy')

##### Local shifting ####
    # print('test_y shape: ', test_y.shape, test_y.max())
    # test_y_local = []
    # for te_y in test_y:
    #     te_y = te_y - te_y.min()
    #     test_y_local.append(te_y)
    # test_y = np.asarray(test_y_local)
    # print('test_y shape: ', test_y.shape, test_y.max())
    # raise
##### Local shifting ####

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]

    test_x = (test_x - Power_max) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    test_x = torch.permute(test_x, (0,3,1,2))
    test_y = torch.permute(test_y, (0,3,1,2))



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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
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

                images = train_x
                true_masks = train_y


                # assert images.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'
                images = images.to(device=net.device_ids[0], dtype=torch.float32)
                true_masks = true_masks.to(device=net.device_ids[0], dtype=torch.float32)
                # images = images.to(device=device, dtype=torch.float32)
                # true_masks = true_masks.to(device=device, dtype=torch.float32)

                # with torch.cuda.amp.autocast(enabled=amp):
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
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(net, val_loader, device)
                        # scheduler.step(val_score)
                        evaluate(net, val_loader, device, true_masks.max(), true_masks.min())
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
                            **histograms
                        })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


def mape(actual, pred): 
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual))

def test_net(net,
              device,
              batch_size: int = 100,
              img_scale: float = 0.5,
              ):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    train_x = np.load(dir_data + 'x_train.npy')
    train_y = np.load(dir_data + 'y_train.npy')

    Power_max = train_x.max()
    Power_min = train_x.min()
    T_max = train_y.max()
    T_min = train_y.min()
    print('Power_max: ', Power_max)
    print('Power_min: ', Power_min)
    print('T_max: ', T_max)
    print('T_min: ', T_min)

    test_x = np.load(dir_data + 'x_test.npy')
    test_y = np.load(dir_data + 'y_test.npy')

    # test_x = train_x[:1]
    # test_y = train_y[:1]

##### Local shifting ####
    # print('test_y shape: ', test_y.shape, test_y.max())
    # test_y_local = []
    # for te_y in test_y:
    #     te_y = te_y - te_y.min()
    #     test_y_local.append(te_y)
    # test_y = np.asarray(test_y_local)
    # print('test_y shape: ', test_y.shape, test_y.max())
    # raise
##### Local shifting ####

    idx_test = np.amax(test_y, axis=(1, 2)) < 300
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]

    test_x = (test_x - Power_min) / (Power_max - Power_min)
    test_y = (test_y - T_min) / (T_max - T_min)

    test_x = np.expand_dims(test_x, -1)
    test_x = torch.from_numpy(test_x)
    test_y = np.expand_dims(test_y, -1)
    test_y = torch.from_numpy(test_y)

    print('test_x shape: ', test_x.shape)
    print('test_y shape: ', test_y.shape)
    test_x = torch.permute(test_x, (0,3,1,2))
    test_y = torch.permute(test_y, (0,3,1,2))
    print('After permute, test_x shape: ', test_x.shape)
    print('After permute, test_y shape: ', test_y.shape)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=True, **loader_args)

    # evaluate(net, test_loader, device, torch.from_numpy(np.asarray(T_max)), torch.from_numpy(np.asarray(T_min)))


    image = test_x.to(device=device, dtype=torch.float32)
    mask_true = test_y.to(device=device, dtype=torch.float32)

    mask_pred = net(image)

    pred = mask_pred.detach().cpu().numpy()
    true = mask_true.detach().cpu().numpy()

    pred = pred * (T_max - T_min) + T_min
    true = true * (T_max - T_min) + T_min


    idx = np.random.choice(mask_pred.shape[0])

    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(left=0.06, bottom=0.05, right=0.95, top=0.85, wspace=0.2, hspace=0.3)
    ax = fig.add_subplot(131)
    ax.set_title(f'Truth')
    im = ax.imshow(true[idx, 0,:,:], origin='lower', cmap='jet')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(132)
    im = ax.imshow(pred[idx, 0,:,:], cmap='jet', origin='lower')
    ax.set_title(f'Pred')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad="2%")
    cb = fig.colorbar(im, cax=cax)

    ax = fig.add_subplot(133)
    im = ax.imshow(abs(pred[idx, 0,:,:] - true[idx, 0,:,:]), cmap='jet', origin='lower')
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

    true_all = true
    pred_all = pred

    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)
    # print('true_all, pred_all', true_all.shape, pred_all.shape)
    mae_error = np.mean(np.abs(true_all - pred_all))
    # print('pred shape: ', pred.shape)
    # raise
    rel_l2 = np.linalg.norm(true_all.flatten() - pred_all.flatten()) / np.linalg.norm(true_all.flatten()) 
    mape_error = mape(true_all, pred_all)
    print('mae: ', mae_error)
    print('rel_l2: ', rel_l2)
    print('mape_error: ', mape_error)
    mape_error = mape(true, pred)
    with open('./logs/final_test_l2.txt', 'w') as f:
        f.write(f'mae is: {mae_error} \n')
        f.write(f'relative l2 is: {rel_l2} \n')
        f.write(f'mape_error is: {mape_error} \n')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=256, help='Batch size')
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
    parser.add_argument('--mode', '-m', type=str, default='test', help='train or test')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU index')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f'cuda:{args.gpu}')
    
    # print(torch.cuda.memory_summary())
    # raise
    # torch.cuda.empty_cache() 


    # device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=1, n_classes=args.outChannel, bilinear=args.bilinear)

    # raise

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.outChannel} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load and args.mode == 'train':
        print('loading model')
        net = nn.DataParallel(net, device_ids = device_list)
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
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
    # raise

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
