import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np

def evaluate(net, dataloader, device, mx, mn):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    cnt = 0

    true_all = []
    pred_all = []
    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        # image, mask_true = batch['image'], batch['mask']
        image, mask_true, params = batch[0], batch[1], batch[2]
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        params = params.to(device=device, dtype=torch.float32)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
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
            pred = pred * (mx - mn) + mn
            true = true * (mx - mn) + mn
            mae = np.mean(np.abs(true - pred))

            true_all.extend(true)
            pred_all.extend(pred)

            idx = np.random.choice(mask_pred.shape[0])
            # print('pred shape: ', pred.shape)
            # print('true shape: ', true.shape)
            # raise
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
            plt.savefig(f'./figs/{idx}.jpg')
            # plt.show()
            plt.close()
            cnt += 1
            print('mae: ', mae)
            # torch.save(net.state_dict(), "./checkpoint/network.pt")

            # # convert to one-hot format
            # if net.n_classes == 1:
            #     mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            #     # compute the Dice score
            #     dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            # else:
            #     mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #     # compute the Dice score, ignoring background
            #     dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)
    rel_l2 = np.linalg.norm(true_all.flatten() - pred_all.flatten()) / np.linalg.norm(true_all.flatten()) 
    with open('/logs/test_l2.txt', 'w') as f:
        f.write('relative l2 is: ', rel_l2, '\n')
    net.train()

    # # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    # return dice_score / num_val_batches
