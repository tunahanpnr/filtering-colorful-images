import numpy as np


def mean(src_img, window_size):
    pad = window_size // 2
    target_img = src_img.copy()
    for y in range(pad, src_img.shape[0] - pad):
        for x in range(pad, src_img.shape[1] - pad):
            window = src_img[y - pad: y + pad + 1,
                     x - pad: x + pad + 1]
            target_img[y][x] = np.mean(window, axis=(0, 1))

    return target_img
