import torch
from torch import load, unsqueeze, stack, no_grad
from torchvision import transforms
from torchvision.transforms.functional import rotate as rotate_tensor

import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
from scipy.ndimage import rotate
from cv2 import addWeighted
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from model import deeper_Unet_like, vanilla_Unet
from model_deconv import vanilla_Unet2

from utils.blobs_utils import get_boxes, a_link_to_the_past, get_local_maxima
from video_display_dataloader import get_video_dataloaders
from utils.grid_utils import get_landmarks_positions, get_faster_landmarks_positions,\
     get_homography_from_points, conflicts_managements, display_on_image


def compare(out, img, thresholod=None):
    heatmap = np.absolute(out - img)
    if thresholod is not None :
        heatmap = np.where(heatmap > thresholod, 1., 0.)
    heatmap = np.amax(heatmap, 2)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=2)
    return heatmap


def tensor_to_image(out, inv_trans=True, batched=False, to_uint8=True) :
    if batched : index_shift = 1
    else : index_shift = 0
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = torch.tensor([0.485, 0.456, 0.406])
    if inv_trans :
        for t, m, s in zip(out, mean, std):
            t.mul_(s).add_(m)
    out = out.cpu().numpy()
    if to_uint8 :
        out *= 256
        out = out.astype(np.uint8)
    out = np.swapaxes(out, index_shift + 0, index_shift + 2)
    out = np.swapaxes(out, index_shift + 0, index_shift + 1)
    return out


def get_transform(x) :
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    tensor = img_transform(x)
    tensor = unsqueeze(tensor, 0).float()
    return tensor.cuda()


def get_video_name(epochs, full_images_path, size, text_addition, save_projection, threshold) :
    video = full_images_path.split('/')[-1]
    # video = video[:-7] # remove "_frames" at the end of the name
    video_epochs = video + "_" + str(epochs) + "_" + str(threshold).split(".")[-1]
    video_epochs_academy = video_epochs + '_' + str(size[0]) + text_addition
    video_epochs_academy += "_projected" if save_projection else ""
    video_epochs_avi = video_epochs_academy + '.avi'
    return video_epochs_avi


if __name__=='__main__':
    torch.cuda.empty_cache()
    size = (256, 256)

    threshold = 0.75

    epochs = 100

    save_projection = True
    # save_projection = False

    field_length = 50
    markers_x = np.linspace(0, field_length, 11)
    field_width = 25
    lines_y = np.linspace(0, field_width, 11)

    path = 'pool model.pth'
    model = vanilla_Unet2(final_depth=len(markers_x) + len(lines_y))

    batch_size = 64

    models_path = './models/'

    full_images_path = '/home/nicolas/datasets/Neptune Dataset/frames/test'


    video_name = get_video_name(epochs, full_images_path, size, "", save_projection, threshold)
    video_path = './videos/' + video_name

    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    video_flow = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 25, (size[1], size[0]))
    prev_boxes = []
    timer_limit = 10
    min_blob_size = 5

    dataloader = get_video_dataloaders(full_images_path, size, batch_size=batch_size)

    i = 0
    j = 1
    with no_grad() :
        for batch in dataloader :
            batch_tensors = batch['tensor_img'].cuda()

            batch_out = model(batch_tensors)
            batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)

            batch_img = batch['img']
            imgs = batch_img.numpy()

            for img, out in zip(imgs, batch_out) :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if save_projection :
                    img, src_pts, dst_pts, entropies = get_faster_landmarks_positions(img, out, threshold,
                                                                                      write_on_image=False,
                                                                                      lines_nb=len(lines_y),
                                                                                      markers_x=markers_x, lines_y=lines_y)
                    src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)
                    H = get_homography_from_points(src_pts, dst_pts, size,
                                                   field_length=field_length, field_width=field_width)
                    if H is not None:
                        img = cv2.warpPerspective(img, H, size)
                else :
                    out = np.where(out < threshold, 0, 1)
                    out *= 255
                    out = np.expand_dims(out, 2)
                    lines = np.max(out[:, :, :, :len(lines_y)], axis=3)
                    markers = np.max(out[:, :, :, len(lines_y):], axis=3)
                    display_out = np.concatenate((lines, markers, markers), axis=2)
                    display_out = display_out.astype(np.uint8)
                    img = display_out#cv2.addWeighted(img, 0.7, display_out, 0.3, 0)
                    # x_list, y_list = get_local_maxima(out, threshold=threshold)
                    # lines = np.argmax(out[:, :, :len(lines_y)], axis=2)
                    # markers = np.argmax(out[:, :, len(lines_y):], axis=2)
                    # for x, y in zip(x_list, y_list):
                    #     line = lines[y, x]
                    #     marker = markers[y, x]
                    #     pool_x = markers_x[marker]
                    #     pool_y = lines_y[line]
                    #     img = display_on_image(img, pool_x, pool_y, x, y, show_text=False)

                video_flow.write(img)

                # from random import random
                # if random() > 0.9 :
                #     plt.imshow(img)
                #     plt.show()

                # fig = plt.figure(figsize=(10, 5))
                # fig.add_subplot(3, 2, 1)
                # plt.imshow(out[:, :, 0, 0])
                # plt.title("lines")
                # fig.add_subplot(3, 2, 2)
                # plt.imshow(out[:, :, 0, 7])
                # plt.title("markers")
                # fig.add_subplot(3, 2, 3)
                # plt.imshow(out[:, :, 0, 1])
                # fig.add_subplot(3, 2, 4)
                # plt.imshow(out[:, :, 0, 8])
                # fig.add_subplot(3, 2, 5)
                # plt.imshow(out[:, :, 0, 2])
                # fig.add_subplot(3, 2, 6)
                # plt.imshow(out[:, :, 0, 9])
                # plt.show()

                if i==31 :
                    print(i*j)
                    i = -1
                    j+=1
                i += 1
    video_flow.release()
