import cv2
import numpy as np


def get_subwindow_index(sub, y, x, window_size):
    index = {
        0: [y - window_size // 2, y + 1, x, x + window_size // 2 + 1],
        1: [y - window_size // 2, y + 1, x - window_size // 2, x + 1],
        2: [y, y + window_size // 2 + 1, x - window_size // 2, x + 1],
        3: [y, y + window_size // 2 + 1, x, x + window_size // 2 + 1]
    }

    return index[sub]


def kuwahara(src_img, window_size):
    target_img = src_img.copy()
    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

    image_width, image_height, image_channel = src_img.shape
    distance = window_size // 2
    for y in range(distance, image_width - distance):
        for x in range(distance, image_height - distance):
            window = hsv_img[y - distance: y + distance + 1,
                     x - distance: x + distance + 1, 2]

            subwindows = np.array(
                [
                    window[0: distance + 1, distance: window_size],
                    window[0: distance + 1, 0: distance + 1],
                    window[distance: window_size, 0: distance + 1],
                    window[distance: window_size, distance: window_size]
                ]
            )

            stds = [np.std(sub) for sub in subwindows]
            min_std = np.argmin(stds)

            sub_index = get_subwindow_index(min_std, y, x, window_size)

            means = [np.mean(src_img[sub_index[0]:sub_index[1], sub_index[2]:sub_index[3], m]) for m in range(3)]
            target_img[y][x] = means

    return target_img
