import numpy as np
import scipy


def flood_fill_hull(image, cls):
    out_img = np.zeros(image.shape)
    points = np.transpose(np.where(image==cls))
    if len(points) == 0:
        return out_img, None
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)

    out_img[out_idx] = 1
    out_img = out_img.astype(np.uint8)
    return out_img, hull