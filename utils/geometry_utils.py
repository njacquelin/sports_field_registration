import numpy as np
# from sympy import Point, Polygon
from shapely.geometry import Polygon


def get_whole_fields_intersection(h_t, size_t, h, size) :
    size_t = size_t[::-1]

    corners = [(0, 0), (size[0], 0), (size[0], size[1]), (0, size[1])]
    corners_t = [(c[0] * size_t[0] / size[0], c[1] * size_t[1] / size[1], 1) for c in corners]

    inv_h_t = np.linalg.inv(h_t)
    inv_h_t /= inv_h_t[2, 2]
    proj_corners_t = [inv_h_t @ c for c in corners_t]
    proj_corners = [(c[0] * size[0] / size_t[0] / c[2], c[1] * size[1] / size_t[1] / c[2], 1) for c in proj_corners_t]

    detected_corners = [h @ c for c in proj_corners]
    detected_corners = [(c[0] / c[2], c[1] / c[2]) for c in detected_corners]

    poly1 = Polygon(corners).buffer(0)
    poly2 = Polygon(detected_corners).buffer(0)
    try :
        intersection = poly1.intersection(poly2).area
    except :
        a=0
    union = poly1.union(poly2).area
    IOU = intersection / union
    return IOU