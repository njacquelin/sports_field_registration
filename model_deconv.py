import torch
from torch import nn
from torchvision.transforms import Resize

from random import random

class vanilla_Unet2 (nn.Module) :
    def __init__(self, final_depth = 18):
        super(vanilla_Unet2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=2, stride=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 3, padding=2, stride=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, final_depth, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z1 = self.conv1(x)
        z2 = self.conv2(z1)
        z3 = self.conv3(z2)
        z4 = self.conv4(z3)

        code = self.bottleneck(z4)
        x_bottleneck = torch.cat((code, z4), dim=1)

        x4 = self.deconv4(x_bottleneck)
        x4 = torch.cat((x4, z3), dim=1)
        x3 = self.deconv3(x4)
        x3 = torch.cat((x3, z2), dim=1)
        x2 = self.deconv2(x3)
        x2 = torch.cat((x2, z1), dim=1)
        x1 = self.deconv1(x2)

        return x1


class double_Unet (nn.Module) :
    def __init__(self, final_depth = 18):
        super(double_Unet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 3, padding=1, stride=1, dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, final_depth, 1, 1, 0),
            nn.Sigmoid(),
        )


        self.heatmap_deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.heatmap_deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128,kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.heatmap_deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.heatmap_deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z1 = self.conv1(x)
        z2 = self.conv2(z1)
        z3 = self.conv3(z2)
        z4 = self.conv4(z3)

        code = self.bottleneck(z4)
        x_bottleneck = torch.cat((code, z4), dim=1)

        x4 = self.heatmap_deconv4(x_bottleneck)
        x4 = torch.cat((x4, z3), dim=1)
        x3 = self.heatmap_deconv3(x4)
        x3 = torch.cat((x3, z2), dim=1)
        x2 = self.heatmap_deconv2(x3)
        x2 = torch.cat((x2, z1), dim=1)
        heatmap = self.heatmap_deconv1(x2)

        x4 = self.deconv4(x_bottleneck * nn.AvgPool2d(16)(heatmap))
        x4 = torch.cat((x4, z3), dim=1) * nn.AvgPool2d(8)(heatmap)
        x3 = self.deconv3(x4)
        x3 = torch.cat((x3, z2), dim=1) * nn.AvgPool2d(4)(heatmap)
        x2 = self.deconv2(x3)
        x2 = torch.cat((x2, z1), dim=1) * nn.AvgPool2d(2)(heatmap)
        x1 = self.deconv1(x2)

        return x1, heatmap