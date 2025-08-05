#Dependencies 
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import torch.optim as optim
from torchsummary import summary


#---------------------------------------- Functions --------------------------------------------#

# Double Convolution Block with Batch Normalization (*)
def double_conv(in_channels, out_channels): 
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# DownSample function
def down_sample(in_channels, out_channels):
    conv = double_conv(in_channels, out_channels)
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    return conv, pool

# UpSample function
def up_sample(in_channels, out_channels):
    up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
    conv = double_conv(in_channels, out_channels)
    return up, conv

#------------------------------------------- UNet Class ----------------------------------------#

class Architecture(nn.Module):

    # Constructor:
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Downsampling path (encoder)
        self.down1_conv, self.down1_pool = down_sample(in_channels, 64)
        self.down2_conv, self.down2_pool = down_sample(64, 128)
        self.down3_conv, self.down3_pool = down_sample(128, 256)
        self.down4_conv, self.down4_pool = down_sample(256, 512)

        # Bottleneck layer
        self.bottleneck = double_conv(512, 1024)

        # Upsampling path (decoder)
        self.up1_up, self.up1_conv = up_sample(1024, 512)
        self.up2_up, self.up2_conv = up_sample(512, 256)
        self.up3_up, self.up3_conv = up_sample(256, 128)
        self.up4_up, self.up4_conv = up_sample(128, 64)

        # Final output layer: Converts feature maps to segmentation output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    # Forward Pass
    def forward(self, input_tensor):

        #------------------------------------ Downsampling -------------------------------------#   
        down1 = self.down1_conv(input_tensor)
        p1 = self.down1_pool(down1)

        down2 = self.down2_conv(p1)
        p2 = self.down2_pool(down2)

        down3 = self.down3_conv(p2)
        p3 = self.down3_pool(down3)

        down4 = self.down4_conv(p3)
        p4 = self.down4_pool(down4)
        

        #------------------------------------ Bottleneck ----------------------------------------#      
        bottleneck = self.bottleneck(p4)
       

        #------------------------------------- Upsampling ---------------------------------------#    
        up1 = self.up1_up(bottleneck)
        up1 = torch.cat([up1, down4], dim=1)
        up1 = self.up1_conv(up1)

        up2 = self.up2_up(up1)
        up2 = torch.cat([up2, down3], dim=1)
        up2 = self.up2_conv(up2)

        up3 = self.up3_up(up2)
        up3 = torch.cat([up3, down2], dim=1)
        up3 = self.up3_conv(up3)

        up4 = self.up4_up(up3)
        up4 = torch.cat([up4, down1], dim=1)
        up4 = self.up4_conv(up4)
       

        #------------------------------------ Output Layer --------------------------------------#
        out = self.out_conv(up4)
        return out

