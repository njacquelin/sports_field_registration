import numpy as np
from skimage.draw import circle, ellipse
import pickle as pk
import os

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def draw_point(data, x, y, channel, spot_size = (1, 1)) :
    rr, cc = ellipse(x, y, spot_size[1], spot_size[0], shape=(data.shape[0], data.shape[1]))
    data[channel][rr, cc] = 255


if __name__ == '__main__' :
    data_path_np = "./grid"

    field_length = 115
    markers = np.linspace(0, field_length, 15)
    field_width = 74
    lines = np.linspace(0, field_width, 7)

    # img_size = (256, 256)
    img_size = (720, 1280)

    spot_size = (10, 15) # pixel size of each element
    # spot_size = (3, 6) # pixel size of each element

    data_np = np.zeros((len(markers) + len(lines), *img_size))
    np_counter = 0

    ### LINES ###
    # for i in range(line_nb):
    for i, l in enumerate(lines) :
        lane_y = int(l * img_size[0] / field_width)
        if lane_y == 0 : lane_y = 1
        if lane_y == img_size[0]: lane_y = img_size[0] - 1
        for j, m in enumerate(markers):
            marker_x = int(m * img_size[1] / field_length)
            if marker_x == 0 : marker_x = 1
            if marker_x == img_size[1]: marker_x = img_size[1] - 1
            # rr, cc = circle(lane_y, marker_x, spot_size[0], shape=img_size)
            rr, cc = ellipse(lane_y, marker_x, spot_size[1], spot_size[0], shape=img_size)
            data_np[i][rr, cc] = 255
            data_np[len(lines) + j][rr, cc] = 255
    #############

    plt.imshow(np.max(data_np, axis=0))
    plt.show()

    np.save(data_path_np, data_np)


