import cv2

from gaussian import gaussian
from kuwahara import kuwahara
from mean import mean

if __name__ == '__main__':
    path = './images/img.png'
    src_img = cv2.imread(path)

    res_kuwahara = kuwahara(src_img, 3)
    res_mean = mean(src_img, 9)
    res_gaussian = gaussian(src_img, 9, 2)

    cv2.imwrite('output_kuwahara.png', res_kuwahara)
    cv2.imwrite('output_mean.png', res_mean)
    cv2.imwrite('output_gaussian.png', res_gaussian)

