import numpy as np
import cv2
import skimage
import gc

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

from matplotlib import pyplot as plt

import sys
import inspect


def remove_blob(x, y, heatmap, min_blob_size):
    shift = 1
    to_zero_kernel = 0
    explore_list = [(y, x)]
    x_close_list = []
    y_close_list = []
    blob_size = 0

    while len(explore_list) != 0 :
        y, x = explore_list.pop(0)
        x_close_list.append(x)
        y_close_list.append(y)
        if 0 < y < heatmap.shape[0]-1 \
        and 0 < x < heatmap.shape[1]-1 :
            heatmap[y-to_zero_kernel:y+to_zero_kernel+1,
                    x-to_zero_kernel:x+to_zero_kernel+1] = 0
            blob_size += 1
            if heatmap[y, x - shift] != 0 and (y, x - shift) not in explore_list:
                explore_list.append((y, x - shift))
            if heatmap[y, x + shift] != 0 and (y, x + shift) not in explore_list:
                explore_list.append((y, x + shift))
            if heatmap[y - shift, x] != 0 and (y - shift, x) not in explore_list:
                explore_list.append((y - shift, x))
            if heatmap[y + shift, x] != 0 and (y + shift, x) not in explore_list:
                explore_list.append((y + shift, x))

    xmin, xmax = min(x_close_list), max(x_close_list)
    ymin, ymax = min(y_close_list), max(y_close_list)

    if min_blob_size != None :
        if blob_size < min_blob_size :
            return heatmap, -1
    return heatmap, (xmin, ymin, xmax, ymax)


def get_blob(x, y, heatmap, min_blob_size):
    shift = 1
    to_zero_kernel = 0
    explore_list = [(y, x)]
    x_close_list = []
    y_close_list = []
    blob_size = 0

    mask = np.zeros_like(heatmap)

    while len(explore_list) != 0 :
        y, x = explore_list.pop(0)
        x_close_list.append(x)
        y_close_list.append(y)
        if 0 < y < heatmap.shape[0]-1 \
        and 0 < x < heatmap.shape[1]-1 :
            heatmap[y-to_zero_kernel:y+to_zero_kernel+1,
                    x-to_zero_kernel:x+to_zero_kernel+1] = 0
            mask[y - to_zero_kernel:y + to_zero_kernel + 1,
                 x - to_zero_kernel:x + to_zero_kernel + 1] = 1
            blob_size += 1
            if heatmap[y, x - shift] != 0 and (y, x - shift) not in explore_list:
                explore_list.append((y, x - shift))
            if heatmap[y, x + shift] != 0 and (y, x + shift) not in explore_list:
                explore_list.append((y, x + shift))
            if heatmap[y - shift, x] != 0 and (y - shift, x) not in explore_list:
                explore_list.append((y - shift, x))
            if heatmap[y + shift, x] != 0 and (y + shift, x) not in explore_list:
                explore_list.append((y + shift, x))

    if min_blob_size != None :
        if blob_size < min_blob_size :
            return heatmap, -1, 0
    return heatmap, mask, blob_size


def get_max_coords(max_index, shape) :
    x = max_index % shape[1]
    y = (max_index-x) // shape[1]
    return x, y


def black_borders(heatmap):
    heatmap[0] = 0
    heatmap[-1] = 0
    heatmap[:, 0] = 0
    heatmap[:, -1] = 0
    return heatmap


def get_boxes(heatmap_source, threshold=0.8, min_blob_size=None, max_blob_size=None) :
    heatmap = np.copy(heatmap_source)
    if len(heatmap.shape) != 2 : # RGB-friendly heatmaps
        if np.argmin(heatmap.shape) == 2 :
            heatmap = heatmap[:, :, 0]
        elif np.argmin(heatmap.shape) == 0 :
            heatmap = heatmap[0, :, :]
    # heatmap = skimage.transform.resize(heatmap, resize)
    heatmap = np.where(heatmap > threshold, 1, 0)
    heatmap = heatmap.astype('uint8')
    heatmap = black_borders(heatmap)
    lin_heatmap = np.ravel(heatmap)
    max_index = np.argmax(lin_heatmap)

    if min_blob_size != None :
        kernel = np.ones((min_blob_size, min_blob_size))
        heatmap = cv2.erode(heatmap, kernel, iterations=1)
        heatmap = cv2.dilate(heatmap, kernel, iterations=1)
    if max_blob_size != None :
        kernel = np.ones((max_blob_size, max_blob_size))
        changed_heatmap = cv2.erode(heatmap, kernel, iterations=1)
        changed_heatmap = cv2.dilate(changed_heatmap, kernel, iterations=1)
        heatmap = np.bitwise_xor(heatmap, changed_heatmap)

    bboxes = []

    while lin_heatmap[max_index] != 0 :
        x, y = get_max_coords(max_index, heatmap.shape)
        heatmap, box = remove_blob(x, y, heatmap, min_blob_size)
        if box != -1 :
            bboxes.append(box)

        lin_heatmap = np.ravel(heatmap)
        max_index = np.argmax(lin_heatmap)

    return bboxes


def get_local_maxima(out, threshold) :
    out_copy = np.where(out < threshold, 0, out)
    flat_out = np.max(out_copy, axis=2)
    neighborhood_size = 10
    data_max = filters.maximum_filter(flat_out, neighborhood_size)
    maxima = (flat_out == data_max)
    data_min = filters.minimum_filter(flat_out, neighborhood_size)
    diff = ((data_max - data_min) > 0.05)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) // 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) // 2
        y.append(y_center)
    return x, y


def get_masks(heatmap_source, threshold=0.8, min_blob_size=None) :
    heatmap = np.copy(heatmap_source)
    if len(heatmap.shape) != 2 : # RGB-friendly heatmaps
        if np.argmin(heatmap.shape) == 2 :
            heatmap = heatmap[:, :, 0]
        elif np.argmin(heatmap.shape) == 0 :
            heatmap = heatmap[0, :, :]
    # heatmap = skimage.transform.resize(heatmap, resize)
    heatmap = np.where(heatmap > threshold, 1, 0)
    heatmap = heatmap.astype('uint8')
    heatmap = black_borders(heatmap)
    lin_heatmap = np.ravel(heatmap)
    max_index = np.argmax(lin_heatmap)

    if min_blob_size != None :
        kernel = np.ones((min_blob_size, min_blob_size))
        heatmap = cv2.erode(heatmap, kernel, iterations=1)
        heatmap = cv2.dilate(heatmap, kernel, iterations=1)

    masks = []
    areas_size = []

    while lin_heatmap[max_index] != 0 :
        x, y = get_max_coords(max_index, heatmap.shape)
        heatmap, mask, blob_size = get_blob(x, y, heatmap, min_blob_size)
        if blob_size != 0 :
            masks.append(mask)
            areas_size.append(blob_size)
        lin_heatmap = np.ravel(heatmap)
        max_index = np.argmax(lin_heatmap)

    return masks, areas_size


def get_masks_faster(heatmap_source, threshold, min_blob_size=None) :
    heatmap = np.copy(heatmap_source)
    if len(heatmap.shape) != 2:  # RGB-friendly heatmaps
        if np.argmin(heatmap.shape) == 2:
            heatmap = heatmap[:, :, 0]
        elif np.argmin(heatmap.shape) == 0:
            heatmap = heatmap[0, :, :]
    heatmap = np.where(heatmap > threshold, 1, 0)
    heatmap = heatmap.astype('uint8')
    # heatmap = black_borders(heatmap)
    # lin_heatmap = np.ravel(heatmap)
    # max_index = np.argmax(lin_heatmap)

    labeled, num_objects = ndimage.label(heatmap)
    slices = ndimage.find_objects(labeled)
    masks = np.zeros((len(slices), heatmap.shape[1], heatmap.shape[0]))
    areas_size = []
    for i, (dy, dx) in enumerate(slices):
        masks[i, dy.start:dy.stop, dx.start:dx.stop] = 1
        area = (dy.start-dy.stop) *  (dx.start-dx.stop)
        areas_size.append(area)
    masks = masks.astype(np.int8)
    return masks, areas_size


def get_IOU(box, prev_box) :
    xmin = max(box[0], prev_box[0])
    ymin = max(box[1], prev_box[1])
    xmax = min(box[2], prev_box[2])
    ymax = min(box[3], prev_box[3])

    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    prev_box_area = (prev_box[2] - prev_box[0] + 1) * (prev_box[3] - prev_box[1] + 1)

    intersection = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    IOU = intersection / float(box_area + prev_box_area - intersection)
    return IOU


def a_link_to_the_past(box, prev_boxes, IOU_threshold=0.3) :
    best_IOU = -1.
    best_box_index = -1.
    if prev_boxes == [] :
        prev_boxes.append((box, 0))
    else :
        for i, (prev_box, timer) in enumerate(prev_boxes) :
            IOU = get_IOU(box, prev_box)
            if IOU > best_IOU and IOU > IOU_threshold :
                best_IOU = IOU
                best_box_index = i
        if best_box_index != -1 : prev_boxes[best_box_index] = (box, 0)
        else : prev_boxes.append((box, 0))
    return best_box_index, prev_boxes


def mAP(truth, found, threshold=0.5) :
    boxes_found_number = len(found)
    if boxes_found_number == 0 : return 0
    AP_counter = 0
    for t in truth :
        best_IOU = -1
        best_index = -1
        for i, f in enumerate(found) :
            IOU = get_IOU(t, f)
            if IOU > best_IOU and IOU > threshold :
                best_IOU = IOU
                best_index = i
        if best_index != -1 :
            AP_counter += 1
            found.pop(best_index)
    mAP = AP_counter / boxes_found_number
    return mAP


def get_last_batch_mAP(batch_truth, batch_out, threshold=0.6, heatmap_threshold=0.8, min_blob_size=None) :
    mAPs = []
    for truth, out in zip(batch_truth, batch_out) :
        true_boxes = get_boxes(truth.cpu().detach().numpy(), threshold=0.5, min_blob_size=0)
        found_boxes = get_boxes(out.cpu().detach().numpy(), threshold=heatmap_threshold, min_blob_size=min_blob_size)
        mAP_50 = mAP(true_boxes, found_boxes, threshold)
        mAPs.append(mAP_50)
    mmAP50 = sum(mAPs) / len(mAPs)
    return mmAP50


def mAR(truth, found, threshold=0.5) :
    true_boxes_number = len(truth)
    if true_boxes_number == 0 : return 0
    AR_counter = 0
    for t in truth :
        best_IOU = -1
        best_index = -1
        for i, f in enumerate(found) :
            IOU = get_IOU(t, f)
            if IOU > best_IOU and IOU > threshold :
                best_IOU = IOU
                best_index = i
        if best_index != -1 :
            AR_counter += 1
            found.pop(best_index)
    mAR = AR_counter / true_boxes_number
    return mAR


def get_last_batch_mAR(batch_truth, batch_out, threshold=0.6, heatmap_threshold=0.8, min_blob_size=None) :
    mARs = []
    for truth, out in zip(batch_truth, batch_out) :
        true_boxes = get_boxes(truth.cpu().detach().numpy(), threshold=0.5, min_blob_size=0)
        found_boxes = get_boxes(out.cpu().detach().numpy(), threshold=heatmap_threshold, min_blob_size=min_blob_size)
        mAR_N = mAR(true_boxes, found_boxes, threshold)
        mARs.append(mAR_N)
    mmAPR = sum(mARs) / len(mARs)
    return mmAPR


def mAPR(truth, found, threshold=0.5) :
    true_boxes_number = len(truth)
    boxes_found_number = len(found)
    if true_boxes_number == 0 and boxes_found_number == 0 :
        return 0, 0

    true_positives = 0
    for t in truth :
        best_IOU = -1
        best_index = -1
        for i, f in enumerate(found) :
            IOU = get_IOU(t, f)
            if IOU > best_IOU and IOU > threshold :
                best_IOU = IOU
                best_index = i
        if best_index != -1 :
            true_positives += 1
            found.pop(best_index)
    mAP = 0 if boxes_found_number == 0 else true_positives / boxes_found_number
    mAR = 0 if true_boxes_number == 0 else true_positives / true_boxes_number
    return mAP, mAR


def get_last_batch_mAPR(batch_truth, batch_out, threshold=0.6, heatmap_threshold=0.8, min_blob_size=None) :
    mAPs = []
    mARs = []
    for truth, out in zip(batch_truth, batch_out) :
        true_boxes = get_boxes(truth.cpu().detach().numpy(), threshold=0.5, min_blob_size=0)
        found_boxes = get_boxes(out.cpu().detach().numpy(), threshold=heatmap_threshold, min_blob_size=min_blob_size)
        (mAP_N, mAR_N) = mAPR(true_boxes, found_boxes, threshold)
        mAPs.append(mAP_N)
        mARs.append(mAR_N)
    mmAP = sum(mAPs) / len(mARs)
    mmAR = sum(mARs) / len(mARs)
    return mmAP, mmAR


# Returns the biggest boxes indices
def keep_biggest_boxes(boxes, N) :
    if len(boxes) < N : return None
    # elif len(boxes) == N: return boxes

    areas = [(xmax-xmin)*(ymax-ymin) for (xmin, ymin, xmax, ymax) in boxes]
    indices = range(len(boxes))

    zipped_lists = list(zip(areas, indices))
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [indice for _, indice in sorted_zipped_lists]
    main_N_boxes = sorted_list1[:N]
    return main_N_boxes


def unite_N_biggest_boxes(areas_size, N) :
    if len(areas_size) < N : return None
    indices = range(len(areas_size))
    zipped_lists = zip(areas_size, indices)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [element for _, element in sorted_zipped_lists]
    main_N_boxes = sorted_list1[-N:]
    return main_N_boxes

