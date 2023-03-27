import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import math

import hdr
import align
from const import *

def luminance(img):
    # Args taken from Dynamic Range Reduction Inspired by Photoreceptor Physiology, IEEE TVCG 2005
    delta = 1e-5
    output_img = np.zeros(img.shape[:2], dtype=float)
    output_img[:, :] = img[:, :, BLUE] * 0.06 + img[:, :, GREEN] * 0.67 + img[:, :, RED] * 0.27 + delta
    return output_img

def Reinhard_2002_global(hdrImg, alpha=0.18):
    lw = luminance(hdrImg)
    lmax = np.max(lw)
    height, width = lw.shape
    N = height * width
    lw_avg = np.exp(np.sum(np.log(lw)) / N)
    scaled_l = (alpha * lw) / lw_avg
    ld = scaled_l * (1 + scaled_l / lmax ** 2) / (1 + scaled_l)
    mapped_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        mapped_img[:, :, i] = (ld * hdrImg[:, :, i] / lw) * 255
    return np.clip(mapped_img, 0.0, 255.0).astype(np.uint8)

def test():
    image = cv2.imread("output/pic2_align.hdr")
    # lum = luminance(image)
    # print(lum.shape)
    # print(np.where(lum != 0.0))    
    result = Reinhard_2002_global(image)
    cv2.imwrite("output/reinhard_global_align.png", result)
    result = Reinhard_2002_global(image, alpha=0.36)
    cv2.imwrite("output/reinhard_global_align_0.36.png", result)
    result = Reinhard_2002_global(image, alpha=0.8)
    cv2.imwrite("output/reinhard_global_align_0.8.png", result)
    result = Reinhard_2002_global(image, alpha=1)
    cv2.imwrite("output/reinhard_global_align_1.png", result)

    image = cv2.imread("output/pic2.hdr")
    result = Reinhard_2002_global(image)
    cv2.imwrite("output/reinhard_global_0.18.png", result)
    result = Reinhard_2002_global(image, alpha=0.36)
    cv2.imwrite("output/reinhard_global_0.36.png", result)
    result = Reinhard_2002_global(image, alpha=0.8)
    cv2.imwrite("output/reinhard_global_0.8.png", result)

if __name__ == "__main__":
    test()
