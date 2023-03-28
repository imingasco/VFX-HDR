import os
import cv2
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt

import hdr
import align
from const import *

# Implementation of Photographic Tone Reproduction for Digital Images, SIGGRAPH 2002
class Reinhard_2022:
    def __init__(self, hdr, alpha=0.18, delta=1e-6, phi=8, epsilon=0.05, scale=5):
        self.hdr = hdr
        self.shape = hdr.shape
        self.N = hdr.shape[0] * hdr.shape[1]
        self.alpha = alpha
        self.delta = delta
        self.phi = phi
        self.epsilon = epsilon
        self.scale = scale
        self.calc_l()

    def calc_l(self):
        self.lw = self.luminance()
        self.lmax = np.max(self.lw)
        self.lw_avg = np.exp(np.sum(np.log(self.lw) / self.N))
        self.lm = (self.alpha * self.lw) / self.lw_avg

    def luminance(self):
        # Coefficients are directly taken from the paper
        result = np.zeros(self.hdr.shape[:2], dtype=float)
        result[:, :] = self.hdr[:, :, BLUE] * 0.06 + self.hdr[:, :, GREEN] * 0.67 + self.hdr[:, :, RED] * 0.27
        return result
    
    def global_op(self):
        ld = self.lm * (1 + self.lm / self.lmax ** 2) / (1 + self.lm)
        result = np.zeros(self.shape, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (ld * self.hdr[:, :, i] / self.lw) * 255
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
    
    def adjust_alpha(self, alpha):
        self.alpha = alpha
        # We should recalculate lm after adjusting alpha
        self.calc_l()

    def circular_symmetric_gaussian(self, alpha, s=1):
        length = 1 + s * 2
        filter = np.zeros((length, length),  dtype=np.float32)
        for i in range(-s, s + 1):
            for j in range(-s, s + 1):
                filter[i][j] = np.exp(-(i ** 2 + j ** 2) / (alpha * s) ** 2) / (np.pi * (alpha * s) ** 2)
        return filter
    
    def dodge_and_burn(self):
        # Parameters are directly taken from the paper
        alpha_1 = 1 / (2 * np.sqrt(2))
        alpha_2 = 1.6 * alpha_1
        scale = [int(1 + 1.6 ** i) for i in range(self.scale)]
        # Processing from larger scale to smaller scale and update at each iteration,
        # so that we can get the smallest scale satisfing the condition
        scale.reverse()
        V1 = np.zeros((self.shape[0], self.shape[1]), dtype=np.float32)
        for i, s in enumerate(scale):
            # print(i)
            filter_1 = self.circular_symmetric_gaussian(alpha_1, s)
            filter_2 = self.circular_symmetric_gaussian(alpha_2, s)
            V1_s = scipy.signal.convolve2d(self.lm, filter_1, mode="same")
            V2_s = scipy.signal.convolve2d(self.lm, filter_2, mode="same")
            V_s = np.abs((V1_s - V2_s) / (2 ** self.phi * self.alpha / s ** 2 + V1_s))
            V_map = V_s < self.epsilon
            V1[V_map] = V_s[V_map]
        ld = self.lm / (1 + V1)
        result = np.zeros(self.shape, dtype=np.float32)
        for i in range(3):
            result[:, :, i] = (ld * self.hdr[:, :, i] / self.lw) * 255
        return np.clip(result, 0.0, 255.0).astype(np.uint8)
    
def ReinhardTonemap(hdrImg):
    tonemapReinhard = cv2.createTonemapReinhard(1.5, -1.0, 0.3, 0.3)
    ldrReinhard = tonemapReinhard.process(hdrImg)
    return np.clip(ldrReinhard * 255, 0.0, 255.0)

def test():
    image = cv2.imread("output/pic2_align.hdr")
    reinhard = Reinhard_2022(image)
    result = reinhard.global_op()
    cv2.imwrite("output/reinhard_global_align_0.18.png", result)
    result = reinhard.dodge_and_burn()
    cv2.imwrite("output/reinhard_local_align_0.18.png", result)
    reinhard.adjust_alpha(0.36)
    result = reinhard.global_op()
    cv2.imwrite("output/reinhard_global_align_0.36.png", result)
    result = reinhard.dodge_and_burn()
    cv2.imwrite("output/reinhard_local_align_0.36.png", result)
    reinhard.adjust_alpha(0.09)
    result = reinhard.global_op()
    cv2.imwrite("output/reinhard_global_align_0.09.png", result)
    result = reinhard.dodge_and_burn()
    cv2.imwrite("output/reinhard_local_align_0.09.png", result)
    reinhard.adjust_alpha(0.135)
    result = reinhard.global_op()
    cv2.imwrite("output/reinhard_global_align_0.135.png", result)
    result = reinhard.dodge_and_burn()
    cv2.imwrite("output/reinhard_local_align_0.135.png", result)

if __name__ == "__main__":
    test()
