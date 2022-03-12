import numpy as np


def generate_gaussian_filter(kernel_size, sigma):
    x = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    g = np.exp(-0.5 * np.square(x) / np.square(sigma))
    k = np.outer(g, g)
    return k / np.sum(k)


def gaussian(src_img, window_size, sigma):
    target_img = src_img.copy()
    filter = generate_gaussian_filter(window_size, sigma)
    padding_value = window_size // 2
    for y in range(padding_value, src_img.shape[0] - padding_value):
        for x in range(padding_value, src_img.shape[1] - padding_value):
            window = src_img[y - padding_value: y + padding_value + 1,
                     x - padding_value: x + padding_value + 1]
            res = [np.sum(np.multiply(window[:, :, d], filter)) for d in range(window.shape[-1])]
            target_img[y][x] = res

    return target_img
