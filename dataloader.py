import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
from skimage import io
from copy import deepcopy
from sklearn.utils import shuffle
from imageio import mimread
import cv2
from cv2 import resize, GaussianBlur, findHomography,\
                warpPerspective, getRotationMatrix2D, warpAffine
from utils.homography_utils import augment_matrix
import numpy as np
from random import random


class Dataloader(Dataset) :
    def __init__(self, img_path, out_path, size, augment_data=True, lines_nb=5) :
        self.transform = self.get_transform()
        self.augment_data = augment_data
        self.size = size

        self.temperature = 1

        yes_img = os.listdir(img_path)
        yes_img = [os.path.join(img_path, f) for f in yes_img]
        yes_img.sort()
        yes_img.sort(key=len)
        self.yes_img = yes_img

        yes_out = os.listdir(out_path)
        yes_out.sort()
        yes_out.sort(key=len)
        dico = {}
        for f in yes_out :
            dico[f] = os.path.join(out_path, f)
        self.yes_out = dico

        self.len = len(self.yes_img)

        self.lines_nb = lines_nb

    def __len__(self) :
        return self.len

    def __getitem__(self, idx):
        img_path = self.yes_img[idx]
        img = io.imread(img_path)

        img_name = img_path.split('/')[-1]
        out_name = img_name.replace("jpg", "homography.npy")
        out_name = self.yes_out[out_name]
        out = np.load(out_name) / 255
        out = cv2.resize(out, (img.shape[1], img.shape[0]))

        if self.augment_data :

            if random() < (1 - self.temperature) :
                step = (1 - self.temperature) * 0.2
                img, out = augment_matrix(img, out, step=step)

            if random() < (1 - self.temperature) :
                img = self.color_shift(img)

            if random() < (1 - self.temperature):  # gaussian blurr
                if random() > 0.5 :
                    kernel = 3
                    img = GaussianBlur(img, (kernel, kernel), 0)
                else :
                    img = self.salt_pepper(img)

            if self.augment_data and random() < (1 - self.temperature) :
                img, out = self.rotation(img, out)
                # if random() > 0.5 :
                #     img, out = self.rotation(img, out)
                # else :
                #     img = np.rot90(img)
                #     out = np.rot90(out)

            if random() < (1 - self.temperature) :
                img, out = self.random_patch(img, out, patch_heatmap=False)

            if random() < (1 - self.temperature) :
                    if random() > 0.5 :
                        img, out = self.random_crop(img, out)
                    else :
                        img, out = self.zoom_out(img, out)

            if random() > 0.5 :
                img, out = self.lr_flip(img, out)

        img = resize(img, self.size)
        img = self.transform(img)

        out = resize(out, self.size)

        mask = np.where(np.max(out, axis=2) != 0, 1, 0)
        mask = transforms.ToTensor()(mask).float()[0]

        out = self.adapt_to_cross_correlation(out)
        out = self.to_tensor(out)

        return {'img': img, 'out': out, 'mask': mask}

    def salt_pepper(self, img):
        s_vs_p = 0.5
        amount = 0.04

        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        img[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        img[tuple(coords)] = 0
        return img

    def rotation(self, img, out, angle=0) :
        max_angle = 30
        angle = random() * max_angle*2 - max_angle
        angle = int(angle * (1 - self.temperature))
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        rotation_matrix = getRotationMatrix2D((cX, cY), angle, scale=1.0)
        img = warpAffine(img, rotation_matrix, (w, h))
        out = warpAffine(out, rotation_matrix, (w, h))
        return img, out

    def zoom_out(self, img, out) :
        reducing_factor = 1 + random() * 1 # [1, 2[
        h, w, _ = img.shape
        new_size = int(w / reducing_factor), int(h / reducing_factor)
        x_margin = int(random() * (h - new_size[1]))
        y_margin = int(random() * (w - new_size[0]))

        grey = np.zeros_like(img)
        img = cv2.resize(img, new_size)
        grey[x_margin : x_margin + new_size[1],
             y_margin : y_margin + new_size[0]] = img

        black = np.zeros_like(out)
        out = cv2.resize(out, new_size)
        black[x_margin : x_margin + new_size[1],
              y_margin : y_margin + new_size[0]] = out

        return grey, black

    def get_transform(self):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        return img_transform

    def random_crop(self, img, out) :
        min = 0.5 # side of the cropped image in respect to the original one
        prop = random() * (1 - min) + min

        h, w, _ = img.shape
        xmin = int(random() * w * (1 - prop))
        xmax = int(xmin + prop * w)
        ymin = int(random() * h * (1 - prop))
        ymax = int(ymin + prop * h)

        h_out, w_out = out.shape[0], out.shape[1]
        xmin_out = int(xmin * w_out / w)
        xmax_out = int(xmax * w_out / w)
        ymin_out = int(ymin * h_out / h)
        ymax_out = int(ymax * h_out / h)

        img = img[ymin:ymax, xmin:xmax]
        out = out[ymin_out:ymax_out, xmin_out:xmax_out]

        return img, out


    def color_shift(self, img) :
        brightness = int((random() * 128 - 64) * (1 - self.temperature))
        contrast = int((random() * 128 - 64) * (1 - self.temperature))
        hue = random() * 40 - 20

        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hnew = np.mod(h + hue, 180).astype(np.uint8)
        hsv = cv2.merge([hnew, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def random_patch(self, img, out, patch_heatmap=False):
        max_patch_nb = 30 * (1 - self.temperature)
        patch_nb = int(random() * max_patch_nb)
        size_max = 0.5 * (1 - self.temperature)
        size_min = 0.05
        for p in range(patch_nb) :

            prop_x = random()**2 * size_max + size_min
            prop_y = random() * (size_max - prop_x)**2 + size_min
            if random() > 0.5 : prop_x, prop_y = prop_y, prop_x

            h, w, _ = img.shape
            xmin = int(random() * w * (1 - prop_x))
            ymin = int(random() * h * (1 - prop_y))
            xmax = int(xmin + prop_x * w)
            ymax = int(ymin + prop_y * h)

            r = random()
            # r = 0.9
            if r < 0.3 : # noise
                img[ymin:ymax, xmin:xmax] = np.random.randint(0, 255, (ymax-ymin, xmax-xmin, 3))
            elif r < 0.6 : # grey shade
                img[ymin:ymax, xmin:xmax] = np.ones((ymax-ymin, xmax-xmin, 3)) * int(random() * 255)
            else :
                hsv_patch = np.ones((ymax-ymin, xmax-xmin, 3)) * [random() * 180, random() * 256, random() * 256]
                bgr_patch = cv2.cvtColor(hsv_patch.astype(np.uint8), cv2.COLOR_HSV2BGR)
                img[ymin:ymax, xmin:xmax] = bgr_patch

            if patch_heatmap :
                out[ymin:ymax, xmin:xmax] = np.zeros((xmax - xmin, ymax - ymin, out.shape[2]))

        return img, out

    def lr_flip(self, img, out) :
        img = np.flip(img, axis=1)
        out = np.flip(out, axis=1)
        markers = out[:,:,self.lines_nb:]
        flipped_markers = markers[:,:,::-1]
        out[:, :, self.lines_nb:] = flipped_markers
        return img, out

    def td_flip(self, img, out):
        img = np.flip(img, axis=0)
        out = np.flip(out, axis=0)
        lines = out[:, :, :self.lines_nb]
        flipped_lines = lines[:, :, ::-1]
        out[:, :, :self.lines_nb] = flipped_lines
        return img, out

    def to_tensor(self, x):
        # x = transforms.ToTensor()(x).float()
        x = torch.tensor(x)
        x = x.permute(2, 0, 1).long()
        return x

    def adapt_to_cross_correlation(self, array, axis=2, threshold=0.001, default_value=100):
        lines = array[:, :, :self.lines_nb]
        markers = array[:, :, self.lines_nb:]

        lines_tensor_2D = np.max(lines, axis=axis)
        lines_adapted = np.argmax(lines, axis=axis)
        lines_adapted = np.where(lines_tensor_2D < threshold, default_value, lines_adapted)
        markers_tensor_2D = np.max(markers, axis=axis)
        markers_adapted = np.argmax(markers, axis=axis)
        markers_adapted = np.where(markers_tensor_2D < threshold, default_value, markers_adapted)

        out = np.zeros((array.shape[0], array.shape[1], 2))
        out[:, :, 0] = lines_adapted
        out[:, :, 1] = markers_adapted
        return out


def get_train_test_dataloaders(img_path, out_path, size, batch_size=32, train_test_ratio=0.8,
                               augment_data=True, shuffle=True, lines_nb=11):
    dataset = Dataloader(img_path, out_path, size, augment_data, lines_nb)
    if train_test_ratio != 1 :
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset.data_augmentation = False
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_dataloader, test_dataloader
    else :
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader