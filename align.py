import os
import sys
import cv2
import numpy as np

from const import *

def grayscale(image):
    image = image.astype(int)
    output_image = (54 * image[:, :, RED] + 183 * image[:, :, GREEN] + 19 * image[:, :, BLUE]) / 256
    return output_image.astype(np.uint8)

def threshold_noise(image, lower, upper):
    output_image = np.logical_or(image < lower, image > upper).astype(np.uint8) * 255
    return output_image

def downsample(image: np.ndarray, scale):
    # return image if scale == 1 else image[::scale, ::scale]
    return image if scale == 1 else cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

def MTB_and_exclusive_bm(image):
    gray_image = grayscale(image)
    median = int(np.median(gray_image))
    bitmap_image = (gray_image > median).astype(np.uint8) * 255
    exclusion_bitmap = threshold_noise(gray_image, median - 10, median + 10)
    return bitmap_image, exclusion_bitmap

def image_shift(image, rshift, cshift):
    mat = np.array([[1, 0, cshift], [0, 1, rshift]], dtype=float)
    return cv2.warpAffine(image, mat, (image.shape[1], image.shape[0]))

def get_shift(ref_image, shift_image, level):
    if level == 0:
        shift = np.array([0, 0], dtype=int)
    else:
        shift = get_shift(downsample(ref_image, 2), downsample(shift_image, 2), level - 1) * 2
    
    ref_mtb, ref_exclusive_bm = MTB_and_exclusive_bm(ref_image)
    shift_mtb, shift_exclusive_bm = MTB_and_exclusive_bm(shift_image)
    min_diff, best_shift = np.inf, None
    prob_shift = [
        [ 0,  0], # default not to shift the image
        [-1, -1], [-1, 0], [-1, 1],
        [ 0, -1],          [ 0, 1],
        [ 1, -1], [ 1, 0], [ 1, 1]
    ]
    for r, c in prob_shift:
        rshift, cshift = shift[0] + r, shift[1] + c
        _shift_mtb = image_shift(shift_mtb, rshift, cshift)
        _shift_exclusive_bm = image_shift(shift_exclusive_bm, rshift, cshift)
        diff = cv2.bitwise_xor(ref_mtb, _shift_mtb)
        diff = cv2.bitwise_and(diff, ref_exclusive_bm)
        diff = cv2.bitwise_and(diff, _shift_exclusive_bm)
        diff = np.count_nonzero(diff)
        # print(diff, min_diff)
        if diff < min_diff:
            min_diff, best_shift = diff, np.array([rshift, cshift])

    return best_shift

def align(images, max_shift=64):
    # Let's pick the first image as reference image
    result = [images[0]]
    pyramid_level = int(np.log2(max_shift))
    for image in images[1:]:
        rshift, cshift = get_shift(images[0], image, pyramid_level)
        print(rshift, cshift)
        # Layer-wise shift
        image_t = np.transpose(image, (2, 0, 1))
        output_image = np.zeros(image_t.shape)
        for channel in range(image_t.shape[0]):
            output_image[channel] = image_shift(image_t[channel], rshift, cshift)
        result.append(np.transpose(output_image, (1, 2, 0)))
    return result

def test():
    img = cv2.imread("img/exposures/img01.jpg")
    cv2.imwrite("output/gray_img01.jpg", grayscale(img))

if __name__ == "__main__":
    test()

