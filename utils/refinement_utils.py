from scipy.ndimage.morphology import binary_dilation

import numpy as np
import cv2
from cv2 import GaussianBlur
import sys

from random import random

import torch
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.nn.functional import grid_sample, affine_grid
from torchvision import transforms
import torchgeometry as tgm

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def unnormalize_matrix(H) :
    mean = np.array([[1., 1., 128.],
                         [1., 1., 128.],
                         [0., 0., 1.]])
    std = np.array([[2., 2., 128.],
                        [2., 2., 128.],
                        [0.01, 0.01, 1.]])

    H = H * std + mean

    return H


def normalize_matrix(H):
    mean = np.array([[1., 1., 128.],
                         [1., 1., 128.],
                         [0., 0., 1.]])
    std = np.array([[2., 2., 128.],
                        [2., 2., 128.],
                        [0.01, 0.01, 1.]])

    H = (H - mean) / std

    return H

def get_norm_matrices():
    mean = np.array([[1., 1., 128.],
                     [1., 1., 128.],
                     [0., 0., 1.]])
    std = np.array([[2., 2., 128.],
                    [2., 2., 128.],
                    [0.01, 0.01, 1.]])
    return mean, std


def get_frame1(video_path) :
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    return frame1


def get_template(template_path, swapaxes=True) :
    pool_template = np.load(template_path)
    if swapaxes :
        pool_template = np.swapaxes(pool_template, 2, 0)
        pool_template = np.swapaxes(pool_template, 0, 1)
    pool_template[:, :, 1] = np.sum(pool_template[:, :, 1:3], axis=2)
    pool_template[:, :, 2] = pool_template[:, :, 3]
    pool_template[:, :, 3] = np.sum(pool_template[:, :, 4:6], axis=2)
    pool_template[:, :, 4] = np.sum(pool_template[:, :, -2:], axis=2)
    pool_template = pool_template[:, :, :5]
    return pool_template


def get_similarity(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    fusion = -np.mean(term_0 + term_1, axis=(0, 1))
    fusion = np.mean(fusion)
    return fusion


def display(out, template, H, display_film, tensor_size=(256, 256)) :
    output = cv2.warpPerspective(out, H, tensor_size)
    print(H)
    print(np.sum(np.max(output, axis=2) * np.max(template, axis=2)) / (256*256))
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 3, 1)
    plt.imshow(np.max(template, axis=2))
    plt.title("proj")
    fig.add_subplot(1, 3, 2)
    plt.imshow(np.max(output * template, axis=2))
    plt.title("fusion")
    fig.add_subplot(1, 3, 3)
    plt.imshow(np.max(output, axis=2))
    plt.title("out")
    plt.show()
    if display_film:
        plt.pause(0.05)


def refinement(out, template) :
    display = False
    display_film = False
    # display_film = True

    mean, std = get_norm_matrices()
    step_size = 0.01

    tensor_size = (256, 256)

    H = np.identity(3)

    opti_results = 1000
    reset_patience = 5
    patience = reset_patience

    if display_film:
        plt.ion()
        plt.show()

    for count in range(200):

        if count % 50 == 0 and display :
            display(out, template, H, display_film)

        results = []
        coords = [(i, j, s) for i in range(3) for j in range(3) for s in [1, -1]]
        for i in range(3):
            for j in range(3):
                if i == 2 and j == 2: continue
                # H + step
                Hbis = np.copy(H)
                Hbis[i, j] += step_size * std[i, j]
                output = cv2.warpPerspective(out, Hbis, tensor_size)
                result = get_similarity(output, template)
                results.append(result)

                # H - step
                Hbis = np.copy(H)
                Hbis[i, j] -= step_size * std[i, j]
                output = cv2.warpPerspective(out, Hbis, tensor_size)
                result = get_similarity(output, template)
                results.append(result)

        best_result = min(results)
        if best_result < opti_results or random() < 0.1:
            best = results.index(best_result)
            i, j, s = coords[best]
            H[i, j] += s * step_size * std[i, j]

        # if np.random.rand(1) > 0.05 :
        #     stochastic_matrix = (np.random.rand(3, 3) * 2 - 1) * std
        #     H += stochastic_matrix * 0.01
        #     step_size *= 10

        opti = opti_results - best_result
        if opti < 1 / 1000:
            patience -= 1
        else:
            patience = reset_patience
            opti_results = best_result
        if patience == 0:
            std /= 2
            step_size /= 2
            # print("step reduction to", std)
            patience = reset_patience
        # print("\t", best_result, "\t", opti, "\t", patience)
        print(count)
    return cv2.warpPerspective(out, H, tensor_size)