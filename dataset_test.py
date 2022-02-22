from torchvision import transforms
from torch.autograd import Variable
import torch

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2

from dataloader import get_train_test_dataloaders
from utils.grid_utils import get_homography_from_points, get_landmarks_positions


def torch2np(img, inv_trans=True, float_to_uint8=True, is_binar=False) :
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                                  ])
    if inv_trans : img = invTrans(img)

    img = img.permute(1, 2, 0)

    img = img.numpy()
    if float_to_uint8 :
        if is_binar :
            img[img != 0] = 1
        img *= 255
        img = img.astype(np.uint8)
    return img


def generate_expected_out(reverse_mask, expected_output, out) :
    return out*reverse_mask + expected_output


if __name__=='__main__' :

    # img_path = '/home/nicolas/datasets/Neptune Dataset/frames/train'
    img_path = '/home/nicolas/datasets/Neptune Dataset/frames/train_all'
    # img_path = '/home/nicolas/datasets/Neptune Dataset/frames/test'
    out_path = './data_management/grids'
    size = (256, 256)

    train_dataloader = get_train_test_dataloaders(img_path, out_path, size, batch_size=1, train_test_ratio=1,
                                                  augment_data=True, shuffle=True, lines_nb=7)
    train_dataloader.dataset.temperature = 0

    for batch in train_dataloader:
        image = batch['img'][0]
        img = torch2np(image)
        img = np.ascontiguousarray(img)

        out = batch["out"][0]
        out = torch2np(out, inv_trans=False, float_to_uint8=True, is_binar=False)

        out[out != 156] = 255
        out[out == 156]=0

        # display_out = np.max(out, axis=2)
        # display_out = np.expand_dims(display_out, 2)
        # display_out = np.concatenate((display_out, display_out, display_out), axis=2)
        # display_out = out * 255
        # display_out = display_out.astype(np.uint8)
        display_out = np.expand_dims(out.astype(np.uint8), 3)
        display_out = np.concatenate((display_out[:,:,1], display_out[:,:,1], display_out[:,:,1]), axis=2)
        # display_out[out[:,:,0]==0] = 20
        # display_out[out[:,:,0]==100] = 0
        # display_out[out[:, :, 1] == 0] = 20
        # display_out[out[:, :, 1] == 100] = 0
        # display_out *= 255
        # initial_display_out = cv2.addWeighted(initial_img, 0.6, display_out, 0.4, 0)
        img_display_out = cv2.addWeighted(img, 0.7, display_out, 0.5, 0)

        fig = plt.figure(figsize=(10, 5))
        fig.add_subplot(1, 3, 1)
        plt.imshow(img)
        fig.add_subplot(1, 3, 2)
        plt.imshow(img_display_out)
        fig.add_subplot(1, 3, 3)
        plt.imshow(display_out)
        plt.show()

        # img, src_pts, dst_pts = get_landmarks_positions(img, out, lines_threshold=0.5, lines_nb=5,
        #                                                 markers_threshold=0.5, write_on_image=False)
        # H = get_homography_from_points(src_pts, dst_pts, size)
        # if H is not None :
        #     warped_img = cv2.warpPerspective(img, H, size)
        # else :
        #     warped_img = img
        # # display_out = cv2.addWeighted(img, 0.7, display_out, 0.3, 0)

        # fig = plt.figure(figsize=(10, 5))
        # fig.add_subplot(1, 2, 1)
        # plt.imshow(img)
        # fig.add_subplot(1, 2, 2)
        # plt.imshow(warped_img)
        # plt.show()