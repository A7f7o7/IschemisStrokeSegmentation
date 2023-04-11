import UnetParts

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms

import numpy as np

import itk

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = UnetParts.DoubleConv(n_channels, 64)
        self.down1 = UnetParts.Down(64, 128)
        self.down2 = UnetParts.Down(128, 256)
        self.down3 = UnetParts.Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = UnetParts.Down(512, 1024 // factor)
        self.up1 = UnetParts.Up(1024, 512 // factor, bilinear)
        self.up2 = UnetParts.Up(512, 256 // factor, bilinear)
        self.up3 = UnetParts.Up(256, 128 // factor, bilinear)
        self.up4 = UnetParts.Up(128, 64, bilinear)
        self.outc = UnetParts.OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
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

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint.checkpoint(self.outc)