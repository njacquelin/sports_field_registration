import numpy as np
from utils.blobs_utils import get_boxes, keep_biggest_boxes, get_masks, get_local_maxima, get_masks_faster
import cv2
from scipy.stats import entropy


def hough_cleaning(out) :
    for i in range(len(out.shape[-1])) :
        this_map = out[:,:,i]

        cleaned_out = None
        out[:, :, i] = cleaned_out


def conflicts_managements(src_pts, dst_pts, entropies) :
    dst_pts = np.array(dst_pts)
    src_pts = np.array(src_pts)
    entropies = np.array(entropies)
    dst_unique, counts = np.unique(dst_pts, axis=0, return_counts=True)
    conflicts = dst_unique[counts > 1]
    final_src = []
    final_dst = []
    for d, s in zip(dst_pts, src_pts) :
        if not (d[0] in conflicts[:, 0] and d[1] in conflicts[:, 1]) :
            final_dst.append(d)
            final_src.append(s)
    for c in conflicts :
        conflict_index = (dst_pts[:, 0] == c[0]) & (dst_pts[:, 1] == c[1])
        best_entropy = np.argmin(entropies[conflict_index])
        best_src = src_pts[conflict_index][best_entropy]
        final_dst.append(c)
        final_src.append(best_src)
    return final_src, final_dst



def display_on_image(img, pool_x, pool_y, img_x, img_y) :
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 0

    text = "(" + str(round(pool_x, 1)) + "m,  " + str(round(pool_y, 1)) + "m)"

    img[img_y - 6:img_y + 6,
        img_x - 6:img_x + 6] = (23 * pool_x, 23 * pool_x, 23 * pool_x)
    img[img_y - 4:img_y + 4,
        img_x - 4:img_x + 4] = (0, 50 * pool_y, 0)

    org = (img_x + 5, img_y + 5)
    img = cv2.putText(img, text, org, font, fontScale,
                      (0, 0, 255), thickness, cv2.LINE_AA)
    return img


def get_homography_from_points(src_pts, dst_pts, size=(256, 256), homography_method=cv2.RANSAC,
                               field_length=115, field_width=74) :
    if len(src_pts) < 4 : return None

    x_coef = size[0] / field_length
    y_coef = size[1] / field_width
    dst_pts = [(x * x_coef, y * y_coef) for (x, y) in dst_pts]

    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)

    H, _ = cv2.findHomography(src_pts, dst_pts, homography_method)
    # H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RHO)
    return H


def get_landmarks_positions(img, out, threshold, lines_nb = 11, write_on_image=True,
                            markers_x=None, lines_y=None) :
    if markers_x is None: markers_x = [0, 5, 15, 25, 35, 45, 50]
    if lines_y is None: lines_y = np.linspace(0, 25, 11)
    src_pts = []
    dst_pts = []
    entropies = []
    out = np.where(out < threshold, 0, 1)
    flat_out = np.max(out, axis=2)
    blobs, blobs_area = get_masks_faster(flat_out, threshold=threshold)
    # blobs, blobs_area = get_masks(flat_out, threshold=threshold)
    lines = np.argmax(out[:, :, :lines_nb], axis=2)
    markers = np.argmax(out[:, :, lines_nb:], axis=2)
    for blob, area in zip(blobs, blobs_area) :
        line = get_most_frequent_in_blob(blob, lines, area)
        marker = get_most_frequent_in_blob(blob, markers, area)
        y, x = get_blob_barycenter(blob, area)
        if out[y, x, line] > threshold and out[y, x, marker + lines_nb] > threshold :
            lines_entropy, markers_entropy = get_positionnal_vector_entropy(out[y, x], lines_nb)
            entropies.append(markers_entropy + lines_entropy)
            # if lines_entropy < 1 and markers_entropy < 0.8 :
            pool_x = markers_x[marker]
            pool_y = lines_y[line]
            src_pts.append((x, y))
            dst_pts.append((pool_x, pool_y))
            if write_on_image :
                img = display_on_image(img, pool_x, pool_y, x, y)
    # print(sum(entropies)/len(entropies), max(entropies), min(entropies))
    return img, src_pts, dst_pts, entropies


def dataset_test_landmarks(img, out, threshold, lines_nb = 11, write_on_image=True,
                            markers_x=None, lines_y=None) :
    if markers_x is None: markers_x = np.linspace(0, 115, 15)
    if lines_y is None: lines_y = np.linspace(0, 74, 11)
    src_pts = []
    dst_pts = []
    x_list, y_list = get_local_maxima(out, threshold=threshold)
    lines = out[:, :, 0]
    markers = out[:, :, 1]
    for x, y in zip(x_list, y_list) :
        line = lines[y, x]
        marker = markers[y, x]
        pool_x = markers_x[marker]
        pool_y = lines_y[line]
        src_pts.append((x, y))
        dst_pts.append((pool_x, pool_y))
        if write_on_image :
            img = display_on_image(img, pool_x, pool_y, x, y)
    return img, src_pts, dst_pts


def get_faster_landmarks_positions(img, out, threshold, lines_nb = 11, write_on_image=True,
                            markers_x=None, lines_y=None) :
    if markers_x is None: markers_x = np.linspace(0, 115, 15)
    if lines_y is None: lines_y = np.linspace(0, 74, 11)
    src_pts = []
    dst_pts = []
    entropies = []
    x_list, y_list = get_local_maxima(out, threshold=threshold)
    lines = np.argmax(out[:, :, :lines_nb], axis=2)
    markers = np.argmax(out[:, :, lines_nb:], axis=2)
    for x, y in zip(x_list, y_list) :
        line = lines[y, x]
        marker = markers[y, x]
        lines_entropy, markers_entropy = get_positionnal_vector_entropy(out[y, x], lines_nb)
        entropies.append(markers_entropy + lines_entropy)
        pool_x = markers_x[marker]
        pool_y = lines_y[line]
        src_pts.append((x, y))
        dst_pts.append((pool_x, pool_y))
        if write_on_image :
            img = display_on_image(img, pool_x, pool_y, x, y)
    return img, src_pts, dst_pts, entropies


def get_most_frequent_in_blob(blob, array, area) :
    zeros_to_remove = array.shape[0] * array.shape[1] - area
    masked_array = blob * array
    ravelled_array = np.ravel(masked_array)
    channels_frequency = np.bincount(ravelled_array)
    channels_frequency[0] -= zeros_to_remove
    most_frequent = np.argmax(channels_frequency)
    return most_frequent


def get_blob_barycenter(blob, area) :
    x_center, y_center = np.argwhere(blob == 1).sum(0) // area
    return x_center, y_center


def get_positionnal_vector_entropy(vector, lines_nb) :
    lines_encoding = vector[:lines_nb]
    markers_encoding = vector[lines_nb:]
    lines_entropy = entropy(lines_encoding)
    markers_entropy = entropy(markers_encoding)
    return lines_entropy, markers_entropy