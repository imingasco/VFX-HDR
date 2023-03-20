import os
import cv2
import numpy as np

RED = 2
GREEN = 1
BLUE = 0

def grayscale(image):
    if image.dtype != float:
        image = image.astype(float) / 255
    output_image = (54 * image[:, :, RED] + 183 * image[:, :, GREEN] + 19 * image[:, :, BLUE]) / 256
    return output_image

def threshold_noise(image, lower, upper):
    if image.dtype != float:
        image = image.astype(float) / 255
    output_image = np.logical_or(image < lower, image > upper)
    return output_image.astype(float)

def downsample(image: np.ndarray, scale):
    return image if scale == 1 else image[::scale, ::scale]

def MTB_and_exclusive_bm(image):
    gray_image = grayscale(image)
    median = np.median(gray_image)
    bitmap_image = gray_image > median
    bitmap_image = bitmap_image.astype(float)
    # the value 4.0 was directly taken from the paper
    exclusion_bitmap = threshold_noise(gray_image, median - 4.0 / 255, median + 4.0 / 255)
    return bitmap_image, exclusion_bitmap

def image_shift(image, rshift, cshift):
    mat = np.array([[1, 0, cshift], [0, 1, rshift]], dtype=float)
    return cv2.warpAffine(image, mat, (image.shape[1], image.shape[0])).astype(float) / 255

def get_shift(ref_image, shift_image, level):
    if level == 0:
        shift = np.array([0, 0], dtype=int)
    else:
        shift = get_shift(downsample(ref_image, 2), downsample(shift_image, 2), level - 1) * 2
    
    ref_mtb, ref_exclusive_bm = MTB_and_exclusive_bm(ref_image)
    shift_mtb, shift_exclusive_bm = MTB_and_exclusive_bm(shift_image)
    min_diff, best_shift = np.inf, None
    prob_shift = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1), ( 0, 0), ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    for r, c in prob_shift:
        rshift, cshift = shift[0] + r, shift[1] + c
        _shift_mtb = image_shift(shift_mtb, rshift, cshift)
        _shift_exclusive_bm = image_shift(shift_exclusive_bm, rshift, cshift)
        diff = cv2.bitwise_xor(ref_mtb, _shift_mtb)
        diff = cv2.bitwise_and(diff, ref_exclusive_bm)
        diff = cv2.bitwise_and(diff, _shift_exclusive_bm)
        diff = np.sum(diff)
        if diff < min_diff:
            min_diff, best_shift = diff, np.array([rshift, cshift])

    return best_shift

def align(images, max_offset=64):
    # Let's pick the first image as reference image
    result = [images[0]]
    pyramid_level = int(np.log2(max_offset))
    for image in images[1:]:
        rshift, cshift = get_shift(images[0], image, pyramid_level)
        # Layer-wise shift
        image_t = np.transpose(image, (2, 0, 1))
        output_image = np.zeros(image_t.shape)
        for channel in range(image_t.shape[0]):
            output_image[channel] = image_shift(image_t[channel], rshift, cshift)
        result.append(np.transpose(output_image, (1, 2, 0)))
    return result

def test():
    image = cv2.imread("exposures/img01.jpg")
    gray_image = grayscale(image)
    cv2.imwrite("exposures/gray_img01.jpg", gray_image * 255)
    downsampled_gray_image = downsample(gray_image * 255, 2)
    cv2.imwrite("exposures/gray_down_img01.jpg", downsampled_gray_image)
    shift_image = image_shift(gray_image * 255, 100, 500)
    cv2.imwrite("exposures/gray_shift_img01.jpg", shift_image)
    buf = np.array([[1.1, 2, 3.5, -1], [1.2, 3.3, -2, -3]], dtype=float)
    threshold_noise(buf, 0, 2)
    images = []
    for i in range(1, 14):
        images.append(cv2.imread(f"exposures/img{i:02}.jpg"))
    res = align(images)
    for i, img in enumerate(res):
        img *= 255
        img = img.astype(int)
        cv2.imwrite(f"exposures/align_img{i+1:02}.jpg", img)


if __name__ == "__main__":
    test()

