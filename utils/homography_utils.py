import numpy as np
from random import random
import cv2


def get_rotation_matrix(tetha, phi, psi) :
    rot_tetha = np.identity(3)
    rot_tetha[1, 1] = np.cos(tetha)
    rot_tetha[2, 2] = np.cos(tetha)
    rot_tetha[1, 2] = - np.sin(tetha)
    rot_tetha[2, 1] = np.sin(tetha)

    rot_phi = np.identity(3)
    rot_phi[0, 0] = np.cos(phi)
    rot_phi[2, 2] = np.cos(phi)
    rot_phi[2, 0] = - np.sin(phi)
    rot_phi[0, 2] = np.sin(phi)

    rot_psi = np.identity(3)
    rot_psi[0, 0] = np.cos(psi)
    rot_psi[1, 1] = np.cos(psi)
    rot_psi[0, 1] = - np.sin(psi)
    rot_psi[1, 0] = np.sin(psi)

    rotation_matrix = np.matmul(rot_tetha, rot_phi)
    rotation_matrix = np.matmul(rotation_matrix, rot_psi)
    return rotation_matrix


def get_translation_matrix(x, y, z) :
    translation_matrix = np.empty((3, 1))
    translation_matrix[0] = x
    translation_matrix[1] = y
    translation_matrix[2] = z
    return translation_matrix


def get_H(theta, phi, psi, x, y, z, intrinsic_m) :
    extrinsic_m = np.identity(3)
    rotation_matrix = get_rotation_matrix(tetha=theta,
                                          phi=phi,
                                          psi=psi)
    translation_matrix = get_translation_matrix(x=x,
                                                y=y,
                                                z=z)
    extrinsic_m[:, :2] = rotation_matrix[:, :2]
    extrinsic_m[:, 2] = translation_matrix[:, 0]
    H = np.matmul(intrinsic_m, extrinsic_m)
    H[2, 2] = 1
    return  H


def augment_matrix(img, out, step=0.05):
    r = random()
    if r < 0.33:
        psi = (random() * 2 - 1) * step
        phi, theta = 0, 0
    elif r < 0.66:
        phi = (random() * 0.05 - 0.025) * step
        psi, theta = 0, 0
    else:
        theta = (random() * 0.05 - 0.025) * step
        phi, psi = 0, 0

    # zoom = random() * 0.4 + 0.8
    x = 0#(random() * 10 - 5)
    y = 0#(random() * 10 - 5)

    intrinsic_m = np.identity(3)
    intrinsic_m[0, 2] = img.shape[0] / 2
    intrinsic_m[1, 2] = img.shape[1] / 2

    augmented_matrix = get_H(theta, phi, psi, x, y, 0, intrinsic_m)

    img = cv2.warpPerspective(img, augmented_matrix, (img.shape[1], img.shape[0]))
    out = cv2.warpPerspective(out, augmented_matrix, (out.shape[1], out.shape[0]))
    return img, out


def get_augmented_matrix(h, step=0.1):
    theta = (random() * 0.05 - 0.025) * step
    phi = (random() * 0.05 - 0.025) * step
    psi = (random() * 2 - 1) * step

    zoom = random() * 0.4 + 0.8
    x = (random() * 10 - 5)
    y = (random() * 10 - 5)

    intrinsic_m = np.identity(3)
    intrinsic_m[0, 2] = 128
    intrinsic_m[1, 2] = 128

    augmented_matrix = get_H(theta, phi, psi, x, y, 0, intrinsic_m)
    augmented_matrix[0, 0] *= zoom
    augmented_matrix[1, 1] *= zoom

    augmented_matrix = augmented_matrix @ h
    return augmented_matrix