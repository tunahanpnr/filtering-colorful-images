import cv2

from gaussian import gaussian
from kuwahara import kuwahara
from mean import mean

if __name__ == '__main__':
    path = './img.png'
    src_img = cv2.imread(path)

    res_kuwahara = kuwahara(src_img, 3)
    res_mean = mean(src_img, 9)
    res_gaussian = gaussian(src_img, 9, 2)

    cv2.imwrite('output_kuwahara.png', res_kuwahara)
    cv2.imwrite('output_mean.png', res_mean)
    cv2.imwrite('output_gaussian.png', res_gaussian)


    #for i in [3, 5, 9, 15]:
    ##    start = time.time()
     #   res_gaussian = kuwahara(src_img, i)
      #  print(f'completed mean with {i} in   {time.time() - start} sec')
     #   cv2.imwrite(f'output_kuwahara_{i}.png', res_gaussian)
