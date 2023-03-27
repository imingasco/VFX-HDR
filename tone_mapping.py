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
    output_img = np.zeros(img.shape[:2], dtype=float)
    output_img[:, :] = img[:, :, BLUE] * 0.0721 + img[:, :, GREEN] * 0.7154 + img[:, :, RED] * 0.2125
    return output_img

def globalOperator(hdrImg, alpha = 0.8):
    pixels = luminance(hdrImg)
    epsilon = 0.0000001
    height, width = pixels.shape
    print(height, width)
    pixels[pixels < epsilon] = epsilon
    pixels = np.log(pixels)

    lw_avg = np.sum(pixels) / (height * width)
    print(lw_avg)

def test():
    image = cv2.imread("output/result.hdr")
    # lum = luminance(image)
    # print(lum.shape)
    # print(np.where(lum != 0.0))

    globalOperator(image)

if __name__ == "__main__":
    test()
