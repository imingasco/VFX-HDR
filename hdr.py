import os
import cv2
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt

import align

BLUE = 0
GREEN = 1
RED = 2

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
    plt.savefig("curve.png")
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

def gcurve(Z, log_t, l, w):
    gs = []
    
    # b, g, r
    for i in range(3):
        gs.append(gsolve(Z[:, :, i], log_t, l, w))
    
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
        

def HDR(imgs, expTimes):
    imgs = np.array(imgs, dtype=np.uint8)
    samplePoints = 100
    l = 35
    Z_b = sample(imgs[:, :, :, BLUE], samplePoints)
    Z_g = sample(imgs[:, :, :, GREEN], samplePoints)
    Z_r = sample(imgs[:, :, :, RED], samplePoints)
    Z = np.transpose(np.array([Z_b, Z_g, Z_r]), (2, 1, 0))
    log_t = np.log(expTimes)
    w = np.array([i if i <= 128 else 256 - i for i in range(256)], dtype=np.uint8)

    g = gcurve(Z, log_t, l, w)
    rad_map = radiance_map(imgs, log_t, l, g, w)

    return rad_map

def test():
    img_dir = "img/memorial"
    images = []
    exp_time = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    # exp_time = [0.06666667, 0.00625, 0.5, 0.003125, 0.25, 0.025, 0.0015625, 0.0125, 0.0125, 1, 15, 2, 20, 30, 4, 8]
    files = [os.path.join(img_dir, filename) for filename in sorted(os.listdir(img_dir))]
    for f in files:
        ext = f.split(".")[-1]
        if ext == "png" or ext == "jpg" or ext == "JPG":
            images.append(cv2.imread(f))
    print(f"fuck {images[0].shape}")
    exp_time = [1 / t for t in exp_time]
    aligned_images = align.align(images)
    print(aligned_images[0].shape)
    rad_map = HDR(aligned_images, exp_time)
    print(rad_map.shape)
    plt.imshow(np.log(rad_map[:, :, 1]))
    plt.show()
    cv2.imwrite("rad_map.png", np.log(rad_map)[:, :, 1])

if __name__ == "__main__":
    test()