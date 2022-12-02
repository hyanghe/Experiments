""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet_param(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, n_params=2):
        super(UNet_param, self).__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.outChannel = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels+1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.fc1 = nn.Linear(n_params, 80*80)
        self.param_conv1 = DoubleConv(1, 1)        

    def forward(self, x, y):

        y = self.fc1(y)
        y = F.relu(y)
        y = y.view(-1, 1, 80, 80)
        y = self.param_conv1(y)
        # print('y shape: ', y.shape)
        # print('x shape: ', x.shape)
        # print('y: ', y.shape, y.min(), y.max())
        # raise
        combined = torch.cat([x, y], dim=1)
        # print('combined shape: ', combined.shape)
        # raise

        x1 = self.inc(combined)
        # x1 = self.inc(x)



        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)



        return logits
