import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np


# def mape(true, pred, T_min, T_max): 
def mape(true, pred): 
    true, pred = np.array(true), np.array(pred)
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    mask = true != 0
    return np.mean(np.abs((true - pred) / true)[mask])

# def relativeL2(true, pred, T_min, T_max):
def relativeL2(true, pred):
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    # return np.linalg.norm(true.flatten() - pred.numpy().flatten()) / np.linalg.norm(true.flatten()) 
    return np.linalg.norm(true.flatten() - pred.flatten()) / np.linalg.norm(true.flatten()) 

# def mae(true, pred, T_min, T_max):
def mae_cal(true, pred):
    print('pred true', pred.shape, true.shape)
    # pred = pred * (T_max - T_min) + T_min
    # true = true * (T_max - T_min) + T_min
    return np.mean(np.abs(true - pred))

def unscaling_strategy3(temp, power, htc, die_z, total_power=50000.0):

    temp = np.squeeze(temp)
    power = np.squeeze(power)
    # power = np.squeeze(power).numpy()
    # print('temp shape: ', temp.shape)
    # print('power shape: ', power.shape)
    # raise
    # power_ratio = []
    # for j in range(power.shape[0]):
    #     # power_ratio.append(total_power/power[j])
    #     # a = total_power
    #     # b = power[j]
    #     # power_ratio.append(np.divide(a, b, out=np.zeros_like(a), where=b!=0))
    # power_ratio = np.asarray(power_ratio)

    for j in range(temp.shape[0]):
        temp[j] = temp[j] * -((np.log10(htc[j])+2.0)**3.0) / (die_z[j]/100.0)**(1.0/3.0)

    # for j in range(temp.shape[0]):
    #     temp[j] = temp[j] / power_ratio[j]

    return temp




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
        # image, mask_true, params = batch[0], batch[1], batch[2]
        image, mask_true = batch[0], batch[1]
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # params = params.to(device=device, dtype=torch.float32)
        # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        # print('image shape: ', image.shape)
        # raise
        with torch.no_grad():
            # predict the mask
            # mask_pred = net(image, params)
            mask_pred = net(image)
            # print('mask_pred shape: ', mask_pred.shape)
            # print('mask_true shape: ', mask_true.shape)
            # raise
            pred = mask_pred.detach().cpu().numpy()
            true = mask_true.detach().cpu().numpy()
            # params = params.detach().cpu().numpy()
            image = image.detach().cpu().numpy()
            # print('pred shape: ', pred.shape)
            # print('true shape: ', true.shape)
            # print('mx: ', mx)
            # raise
            # mx = mx.numpy()
            # mn = mn.numpy()
            pred = pred * (mx - mn) + mn
            true = true * (mx - mn) + mn


            # htc = params[:, 0]
            # die_z = params[:, 1]
            # for j in range(pred.shape[0]):
            #     pred[j] = (die_z[j]/100.0)**(1.0/3.0) * pred[j] / -((np.log10(htc[j])+2.0)**3.0)
            #     true[j] = (die_z[j]/100.0)**(1.0/3.0) * true[j] / -((np.log10(htc[j])+2.0)**3.0)

            mae = np.mean(np.abs(true - pred))

            true_all.extend(true)
            pred_all.extend(pred)


            for idx in range(len(image)):
                # idx = np.random.choice(mask_pred.shape[0])

                # print('pred shape: ', pred.shape)
                # print('true shape: ', true.shape)
                # raise
                true = np.squeeze(true)
                pred = np.squeeze(pred)

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
                # plt.savefig(f'./figs/{idx}_htc_{htc[idx]}_diez_{die_z[idx]}.jpg')
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
        break
    true_all = np.asarray(true_all)
    pred_all = np.asarray(pred_all)
    # rel_l2 = np.linalg.norm(true_all.flatten() - pred_all.flatten()) / np.linalg.norm(true_all.flatten())
    mae_error = mae_cal(true, pred)
    mape_error = mape(true, pred)
    rel_l2 = relativeL2(true, pred)
    print('mae: ', mae_error)
    print('rel_l2: ', rel_l2)
    print('mape_error: ', mape_error)
    with open('./final_test_l2.txt', 'w') as f:
        f.write(f'mae is: {mae} \n')
        f.write(f'relative l2 is: {rel_l2} \n')
        f.write(f'mape_error is: {mape_error} \n')

    # with open('./logs/test_l2.txt', 'w') as f:
    #     f.write(f'relative l2 is: {rel_l2} \n')
    net.train()

    # # Fixes a potential division by zero error
    # if num_val_batches == 0:
    #     return dice_score
    # return dice_score / num_val_batches
