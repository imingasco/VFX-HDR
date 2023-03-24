import os
import cv2
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

import align
from const import *

def sample(imgs, N=100):
    samples = []
    for img in imgs:
        samples.append(cv2.resize(img,(10, 10)).flatten())
    return np.array(samples, dtype=np.uint8)

def plot_gcurve(g):
    plt.plot(g[BLUE], np.arange(256), "b")
    plt.plot(g[GREEN], np.arange(256), "g")
    plt.plot(g[RED], np.arange(256), "r")
    plt.xlabel("$ln{E_i} + ln{\delta t_j}$")
    plt.ylabel("$g(Z_ij)$")
    plt.show()
    # plt.savefig("curve.png")
    plt.clf()

def gsolve(Z, log_t, l, w):
    n = 256
    N = np.size(Z, 0)
    P = np.size(Z, 1)

    A = np.zeros((N * P + n + 1, n + N))
    b = np.zeros((A.shape[0], 1))
    k = 0

    for i in range(N):
        for j in range(P):
            wij =  w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k,0] = wij * log_t[j]
            k += 1
    
    A[k, 127] = 1
    k = k + 1

    for i in range(n - 1):
        A[k, i] = w[i + 1] * l
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = w[i + 1] * l
        k = k + 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n].flatten()

    return g

def gcurve(Z, log_t, l, w, plot=False):
    gs = []
    
    # b, g, r
    for i in range(3):
        gs.append(gsolve(Z[:, :, i], log_t, l, w))
    
    if plot:
        plot_gcurve(gs)

    return gs

def genlE(Z, log_t, l, g, w):
    P, height, width = Z.shape
    d = np.sum(w[Z], axis=0)
    n = np.sum(w[Z] * (g[Z] - log_t.reshape(P, 1, 1)), axis=0)
    d[d < 0.01] = 1
    return n / d
    

def radiance_map(imgs, log_t, l, g, w):
    P = len(imgs)
    height, width, _ = imgs[0].shape
    maps = []

    for i in range(3):
        maps.append(np.exp(genlE(imgs[:, :, :, i], log_t, l, g[i], w)))
    maps = np.transpose(np.array(maps), (1, 2, 0))

    return maps     

def HDR(imgs, exposure, plot=False, l=50):
    imgs = np.array(imgs, dtype=np.uint8)
    samplePoints = 100

    Z_b = sample(imgs[:, :, :, BLUE], samplePoints)
    Z_g = sample(imgs[:, :, :, GREEN], samplePoints)
    Z_r = sample(imgs[:, :, :, RED], samplePoints)
    Z = np.transpose(np.array([Z_b, Z_g, Z_r]), (2, 1, 0))
    log_t = np.log(exposure)
    w = np.array([i if i <= 128 else 256 - i for i in range(256)], dtype=np.uint8)

    g = gcurve(Z, log_t, l, w, plot)
    rad_map = radiance_map(imgs, log_t, l, g, w)

    return rad_map

def main(args):
    # Read images from input directory
    try:
        files = [os.path.join(args.input, filename) for filename in sorted(os.listdir(args.input))]
    except OSError as e:
        print(e)
        print("Please check your input directory.")
        exit(1)
    
    image_exts = ["png", "JPG", "jpg", "jpeg", "JPEG"]
    images = []
    exposure = []
    for f in files:
        ext = f.split(".")[-1]
        if ext in image_exts:
            images.append(cv2.imread(f))
            # Ref: https://stackoverflow.com/questions/21697645/how-to-extract-metadata-from-an-image-using-python
            pil_image = Image.open(f)
            exif = { ExifTags.TAGS[k]: v for k, v in pil_image._getexif().items() if k in ExifTags.TAGS }
            denominator, numerator = exif["ExposureTime"]
            exposure.append(denominator / numerator)
    if args.align:
        print("Performing alignment...")
        images = align.align(images, max_shift=args.shift)
        print("Alignment done.")
    print("Performing HDR algorithm...")
    rad_map = HDR(images, exposure, plot=args.plot, l=args.l)
    print("HDR done.")
    if args.plot:
        print("Showing the radiance map...")
        plt.imshow(np.log(rad_map), cmap="jet")
        plt.colorbar()
        plt.show()
        plt.clf()
    print(f"Saving the result to {os.path.join(args.output, args.hdr)}")
    cv2.imwrite(os.path.join(args.output, args.hdr), np.log(rad_map))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to the directory containing source images", required=True)
    parser.add_argument("-o", "--output", help="Path to the directory for the output images", required=True)
    parser.add_argument("--hdr", help="Desired file name for the output hdr image", default="result.hdr")
    parser.add_argument("-a", "--align", help="Images will be aligned before performing HDR if specified", action="store_false")
    parser.add_argument("-p", "--plot", help="gcurves and radiance maps will be shown if specified", action="store_true")
    parser.add_argument("-t", "--test", help="test mode", action="store_true")
    parser.add_argument("-l", help="smoothing factor of the hdr function", default=50)
    parser.add_argument("-s", "--shift", help="maximum shift for MTB algorithm", default=64)
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
