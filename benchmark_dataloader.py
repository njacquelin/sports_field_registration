import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
import cv2
from cv2 import resize, GaussianBlur, findHomography, warpPerspective
import numpy as np
from random import random


class Dataloader(Dataset) :
    def __init__(self, img_path, homographies_path, size) :
        self.transform = self.get_transform()
        self.size = size

        img_name = os.listdir(img_path)
        img_name.sort()
        img_name.sort(key=len)
        img = [os.path.join(img_path, f) for f in img_name]
        self.yes_img = img

        homographies_name = [f.replace("jpg", "homography.npy") for f in img_name]
        homographies = [os.path.join(homographies_path, f)
                        for f in homographies_name]
        self.homographies = homographies

        self.len = len(self.yes_img) # - 2 # -2 because 3 images stacking

    def __len__(self) :
        return self.len

    def __getitem__(self, idx) :
        img_path = self.yes_img[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize(img, (self.size))
        tensor_img = self.transform(img)
        tensor_img = tensor_img.view(3, tensor_img.shape[-2], tensor_img.shape[-1])

        homography_path = self.homographies[idx]
        homography = np.load(homography_path)
        # homography = self.adapt_homography(homography)

        return {'img' : img, 'tensor_img' : tensor_img,
                'matrix': homography, 'path': img_path}


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


def get_benchmark_dataloaders(img_path, label_path, size, batch_size=32):
    dataset = Dataloader(img_path, label_path, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader