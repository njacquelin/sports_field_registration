import torch
from torch import load, unsqueeze, stack, no_grad
from torchvision import transforms

import os
from skimage.transform import resize
from cv2 import addWeighted
import cv2
import numpy as np
from numpy import median
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from model import vanilla_Unet
from model_deconv import vanilla_Unet2

from utils.geometry_utils import get_whole_fields_intersection
from benchmark_dataloader import get_benchmark_dataloaders
from utils.grid_utils import get_landmarks_positions, get_homography_from_points, conflicts_managements


def get_IOU(projected_template, projected_template_truth) :
    intersection = (projected_template != 0) & (projected_template_truth != 0)
    union = (projected_template != 0) | (projected_template_truth != 0)
    if np.count_nonzero(union) != 0 :
        IOU = np.count_nonzero(intersection) / np.count_nonzero(union)
    else :
        IOU = 0
    return IOU

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


if __name__=='__main__':
    torch.cuda.empty_cache()
    size = (256, 256)

    threshold = 0.75
    model = vanilla_Unet2(final_depth=len(lines_y) + len(markers_x))
    batch_size = 64
    models_path = './models/'

    ### SWIMMING POOL REGISTATION ###
    field_length = 50
    markers_x = np.linspace(0, field_length, 11)
    field_width = 25
    lines_y = np.linspace(0, field_width, 11)
    path = 'pool model.pth'
    full_images_path = 'path/to/Neptune Dataset/frames/test'
    homographies_path = '/path/to/Neptune Dataset/homographies'
    ###
    
    # ### SOCCER REGISTRATION ###
    # field_length = 115
    # markers_x = np.linspace(0, field_length, 11)
    # field_width = 74
    # lines_y = np.linspace(0, field_width, 11)
    # path = 'soccer model.pth'
    # full_images_path = 'path/to/WorldCup Soccer Dataset/frames/test'
    # homographies_path = '/path/to/WorldCup Soccer Dataset/homographies'
    # ###

    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    model = model.cuda()
    model.eval()

    dataloader = get_benchmark_dataloaders(full_images_path, homographies_path, size, batch_size=batch_size)

    i = 0
    j = 1

    template = np.ones(size)
    size_t = (500, 1000)
    template_t = np.ones(size_t)

    IOU_list = []
    IOU_whole_list = []
    fail_count = 0

    with no_grad() :
        for batch in dataloader :
            batch_tensors = batch['tensor_img'].cuda()

            batch_out = model(batch_tensors)
            batch_out = tensor_to_image(batch_out, inv_trans=False, batched=True, to_uint8=False)

            batch_img = batch['img']
            imgs = batch_img.numpy()

            h_truth = batch['matrix']
            h_truth = h_truth.numpy()

            for img, out, h_t in zip(imgs, batch_out, h_truth) :
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                projected_template_truth = cv2.warpPerspective(template_t, h_t, size_t[::-1])
                projected_template_truth = resize(projected_template_truth, size)
                projected_img = resize(img, size_t)
                projected_img = cv2.warpPerspective(projected_img, h_t, size_t[::-1])
                projected_img = resize(projected_img, size)

                img, src_pts, dst_pts, entropies = get_landmarks_positions(img, out, threshold,
                                                                           write_on_image=False, lines_nb=len(lines_y),
                                                                           markers_x=markers_x, lines_y=lines_y)
                src_pts, dst_pts = conflicts_managements(src_pts, dst_pts, entropies)

                H = get_homography_from_points(src_pts, dst_pts, size,
                                               field_length=field_length, field_width=field_width)

                if H is None:
                    H = np.eye(3)
                    fail_count += 1

                img = cv2.warpPerspective(img, H, size)
                projected_template = cv2.warpPerspective(template, H, size)
                IOU = get_IOU(projected_template, projected_template_truth)
                IOU_list.append(IOU)

                whole_IOU = get_whole_fields_intersection(h_t, size_t, H, size)
                IOU_whole_list.append(whole_IOU)

                if i==32 :
                    print(i*j)
                    i = -1
                    j+=1
                i += 1

    print(f'\nIOU mean = {sum(IOU_list) / len(IOU_list):.4f}')
    print(f'IOU median = {median(IOU_list):.4f}')
    print(f'\nwhole IOU mean = {sum(IOU_whole_list) / len(IOU_whole_list):.4f}')
    print(f'whole IOU median = {median(IOU_whole_list):.4f}')
