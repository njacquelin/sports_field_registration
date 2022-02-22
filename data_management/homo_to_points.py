import pickle
import pandas as pd
import numpy as np
from random import random
import os
from cv2 import warpPerspective, findHomography, GaussianBlur, resize, imread
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')



def get_matrix(data, out_size=(256, 256), in_size=(1000, 500)) :
    src_pts = data.source_points.numpy()
    src_pts[:, 0] = src_pts[:, 0] * out_size[0] / in_size[0]
    src_pts[:, 1] = src_pts[:, 1] * out_size / in_size[1]
    dst_pts = data.destination_points.numpy()
    dst_pts[:, 0] = dst_pts[:, 0] * out_size[0] / in_size[0]
    dst_pts[:, 1] = dst_pts[:, 1] * out_size[1] / in_size[1] - out_size[1]

    matrix, _ = findHomography(src_pts, dst_pts)
    return matrix


if __name__ == "__main__" :
    # save_path = "./dense_grid/"
    # template_path = './dense_grid.npy'
    save_path = "./grid/"
    template_path = './grid.npy'

    if not os.path.exists(save_path) : os.mkdir(save_path)
    files_list = os.listdir(save_path)

    labels_path = '/path/to/WorldCup Soccer Homography/train_val/homographies'
    files = os.listdir(labels_path)
    files = [f for f in files if f.endswith(".homographyMatrix")]
    matrices = [np.loadtxt(os.path.join(labels_path, f)) for f in files]

    out_size = (1280, 720)
    final_size = (256, 256)

    display = False
    # display = True

    template = np.load(template_path)
    template = np.swapaxes(template, 2, 0)
    template = np.swapaxes(template, 0, 1)
    # template = resize(template, out_size)

    for img_name, h in zip(files, matrices) :

        heatmap_path = os.path.join(save_path, img_name)
        # h = h[0].numpy()

        if not np.isnan(h).any() :
            scale_factor = np.eye(3)
            scale_factor[0, 0] = out_size[0] / 115
            scale_factor[1, 1] = out_size[1] / 74
            h = scale_factor @ h
            h_back = np.linalg.inv(h)

            result = warpPerspective(template, h_back, out_size)
            result = resize(result, final_size)

            result = GaussianBlur(result, (5, 5), 0)
            result[result != 0] = 255
            # result = GaussianBlur(result, (3, 3), 0)
            result = result.astype(np.uint8)

            if not display :
                np.save(heatmap_path, result)

            else :
                if random() < 0.95 : continue

                lines_nb = 7
                try :
                    img = imread(os.path.join('/path/to/WorldCup Soccer Homography/train_val/images/train',
                                              img_name.replace('homographyMatrix', 'jpg')))
                    img = resize(img, final_size)
                except :
                    img = imread(
                        os.path.join('/path/to/WorldCup Soccer Homography/train_val/images/val',
                                     img_name.replace('homographyMatrix', 'jpg')))
                    img = resize(img, final_size)
                flat_max = np.max(result, axis=2)
                flat_max = np.expand_dims(flat_max, axis=2)
                flat_img_max = np.concatenate((flat_max, flat_max, flat_max), axis=2).astype(np.uint8)
                import cv2
                img = cv2.addWeighted(img, 0.7, flat_img_max, 0.5, 0)

                fig = plt.figure(figsize=(10, 10))
                fig.add_subplot(4, 2, 1)
                plt.imshow(np.max(template, axis=2))
                fig.add_subplot(4, 2, 2)
                plt.imshow(np.max(result, axis=2))

                fig.add_subplot(4, 2, 3)
                plt.imshow(np.max(template[:, :, :lines_nb], axis=2))
                fig.add_subplot(4, 2, 4)
                plt.imshow(np.max(result[:, :, :lines_nb], axis=2))

                fig.add_subplot(4, 2, 5)
                plt.imshow(np.max(template[:, :, lines_nb:], axis=2))
                fig.add_subplot(4, 2, 6)
                plt.imshow(np.max(result[:, :, lines_nb:], axis=2))

                fig.add_subplot(4, 2, 7)
                plt.imshow(np.max(template, axis=2))
                fig.add_subplot(4, 2, 8)
                plt.imshow(img)

                plt.show()
