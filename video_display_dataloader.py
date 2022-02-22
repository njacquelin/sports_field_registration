import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
import cv2
from skimage import io
from cv2 import resize, GaussianBlur, findHomography, warpPerspective
import numpy as np
from random import random


class Yes_Dataloader(Dataset) :
    def __init__(self, img_path, size) :
        self.transform = self.get_transform()
        self.size = size

        yes_img = os.listdir(img_path)
        yes_img = [os.path.join(img_path, f) for f in yes_img]
        yes_img.sort()
        yes_img.sort(key=len)
        self.yes_img = yes_img

        self.len = len(self.yes_img)

    def __len__(self) :
        return self.len

    def __getitem__(self, idx) :
        img_path = self.yes_img[idx]
        img = io.imread(img_path)
        # img = self.zoom_out(img)
        img = resize(img, (self.size))
        tensor_img = self.transform(img)
        tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])
        return {'img' : img, 'tensor_img' : tensor_img, 'path': img_path}

    def zoom_out(self, img) :
        reducing_factor = 2
        h, w, _ = img.shape
        new_size = int(w / reducing_factor), int(h / reducing_factor)
        x_margin = (h - new_size[1]) // 2
        y_margin = (w - new_size[0]) // 2
        grey = np.zeros_like(img)
        img = cv2.resize(img, new_size)
        grey[x_margin : x_margin + new_size[1],
             y_margin : y_margin + new_size[0]] = img
        return grey

    def get_transform(self):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        return img_transform

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result


def get_video_dataloaders(img_path, size, batch_size=32):
    dataset = Yes_Dataloader(img_path, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader