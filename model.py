import torch
from torch import nn
from torchvision.transforms import Resize

from random import random


class deeper_Unet_like (nn.Module) :
    def __init__(self):
        super(deeper_Unet_like, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.deconv5 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(64, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 24, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(24),
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.heatmap = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1, 1, 0),
            nn.Sigmoid(),
            # nn.Softmax(dim=1),
        )

    def forward(self, x, residual_connexions=False, rand_threshold=0.5):
        z1 = self.conv1(x)
        z2 = self.conv2(z1)
        z3 = self.conv3(z2)
        z4 = self.conv4(z3)
        z5 = self.conv5(z4)

        code = self.bottleneck(z5)

        x_bottleneck = torch.cat((code, z5), dim=1)
        x5 = self.deconv5(x_bottleneck)
        x5 = torch.cat((x5, z4), dim=1)
        x4 = self.deconv4(x5)
        x4 = torch.cat((x4, z3), dim=1)
        x3 = self.deconv3(x4)
        x3 = torch.cat((x3, z2), dim=1)
        x2 = self.deconv2(x3)
        x2 = torch.cat((x2, z1), dim=1)
        x1 = self.deconv1(x2)

        x_out = self.heatmap(x1)

        return x_out


class vanilla_Unet (nn.Module) :
    def __init__(self, final_depth = 18):
        super(vanilla_Unet, self).__init__()

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
            # nn.Conv2d(256, 512, 3, 1, padding=3, dilation=3),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )


        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 1, padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )

        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2), mode="bilinear", align_corners=True),
            nn.Conv2d(128, 64, 1, 1),
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