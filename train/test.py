""" Test """
import os
import torch
from train import utils
torch.cuda.current_device()
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as Data
from train.model import train_model
import pytorch_ssim
import imageio
from vgg import Vgg16
import math

device = torch.device("cuda")

def main():

    # set default gpu device id
    torch.cuda.set_device(3)
    torch.backends.cudnn.benchmark = True
    # set seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # load data
    test_path = "train/test.mat"
    ldct, laplace, ndct = utils.load_test_data(test_path)
    # init
    l2 = nn.MSELoss().to(device)
    ssim = pytorch_ssim.SSIM()
    model = train_model()
    vgg = Vgg16()
    model.load_state_dict(torch.load('train/checkpoint.pth.tar').module.state_dict())
    model = model.to(device)
    vgg = vgg.to(device)
    torch_dataset_test = Data.TensorDataset(torch.from_numpy(ldct), torch.from_numpy(laplace), torch.from_numpy(ndct))
    test_loader = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    test(test_loader, model, ssim, l2, vgg)


def test(test_loader, model, ssim, l2, vgg):
    percep_losses = utils.AverageMeter()
    ssim_losses = utils.AverageMeter()
    mse_losses = utils.AverageMeter()
    rmse_losses = utils.AverageMeter()
    psnr_losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, L, y) in enumerate(valid_loader):
            X = X.type(torch.FloatTensor)
            L = L.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            L = L.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X, L)
            output = logits.detach().cpu().numpy()
            output = np.squeeze(output)
            output[output < 0] = 0
            output[output > 1] = 1
            #imageio.imsave('res.png', output)

            logits_vgg = vgg(torch.cat((logits, logits, logits), 1))
            y_vgg = vgg(torch.cat((y, y, y), 1))
            percep_loss = l2(logits_vgg[0], y_vgg[0])+l2(logits_vgg[1], y_vgg[1])+l2(logits_vgg[2], y_vgg[2])+l2(logits_vgg[3], y_vgg[3])
            ssim_loss = ssim(logits, y)
            mse_loss = l2(logits, y)
            rmse_loss = mse_loss**0.5
            psnr_loss = 10*math.log10(1. / mse_loss)

            percep_losses.update(percep_loss.item(), 1)
            ssim_losses.update(ssim_loss.item(),1)
            mse_losses.update(mse_loss.item(),1)
            rmse_losses.update(rmse_loss.item(),1)
            psnr_losses.update(psnr_loss,1)

    print('percep',percep_losses.avg,'+-',percep_losses.SD())
    print('ssim',ssim_losses.avg,'+-',ssim_losses.SD())
    print('mse',mse_losses.avg,'+-',mse_losses.SD())
    print('rmse',rmse_losses.avg,'+-',rmse_losses.SD())
    print('psnr',psnr_losses.avg,'+-',psnr_losses.SD())

if __name__ == "__main__":
    main()
