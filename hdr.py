import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import math
import align

def w_gen():
    n = 256

    w = np.zeros(n)
    for i in range(n):
        if i > n // 2:
            w[i] = n - i
        else:
            w[i] = i
    
    return w

def sample(imgs, N=100):
    samples = []
    for img in imgs:
        # print(img.shape)
        # print(np.array(img).shape)
        samples.append(cv2.resize(img,(10, 10)).flatten())

    # print(np.array(samples, dtype = np.int))

    return np.array(samples, dtype = np.int)

def exposureLog(expTimes):
    return np.log(expTimes)

def gsolve(Z, B, l, w):
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
            b[k,0] = wij * B[j]
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
    lE = x[n + 1:].flatten()

    return g, lE

def gCurve(Z, B, l, w):
    gs = []
    
    # b, g, r
    for i in range(3):
        g, _ = gsolve(Z[:, :, i], B, l, w)
        gs.append(g)
    
    plt.plot(gs[0], np.arange(256), "b")
    plt.plot(gs[1], np.arange(256), "g")
    plt.plot(gs[2], np.arange(256), "r")
    plt.title("g")
    # plt.show()
    plt.savefig("basdas.png")
    plt.clf()
    
    
    return gs

def genlE(Z, B, l, g, w):
    P, height, width = Z.shape
    Z = Z.astype(int)
    d = np.sum(w[Z], axis = 0)
    n = np.sum(w[Z] * (g[Z].reshape(P, height, width) - B.reshape(P, 1, 1)), axis = 0)
    # d[d <= 0] = 1

    lE = (n / d)
    lE = lE.reshape([height, width])

    return lE
    

def radianceMap(imgs, B, l, g, w):
    P = len(imgs)
    height, width, _ = imgs[0].shape
    lE = []

    for i in range(3):
        lE.append(genlE(imgs[:, :, :, i], B, l, g[i], w))
    
    # print(lE[0].shape)
    maps = np.zeros([height, width, 3], dtype = np.float32)

    for i in range(3):
        maps[:, :, i] = lE[i]

    maps = np.exp(maps)
    # print(maps.shape, maps)

    return maps
        

def HDR (imgs, expTimes):
    imgs = np.array(imgs, dtype=np.float32)
    # print(imgs.shape)
    samplePoints = 100
    l = 50
    Z_b = sample(imgs[:, :, :, 0], samplePoints)
    Z_g = sample(imgs[:, :, :, 1], samplePoints)
    Z_r = sample(imgs[:, :, :, 2], samplePoints)
    Z = np.transpose(np.array([Z_b, Z_g, Z_r]), (2, 1, 0)).astype(int)
    # print(Z.shape)
    B = exposureLog(expTimes)
    w = w_gen()

    g = gCurve(Z, B, l, w)
    radiaMaps = radianceMap(imgs, B, l, g, w)

    return radiaMaps

def test():
    images = []
    exp_time = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    for file in sorted(os.listdir("memorial")):
        if file[-3:] == "png":
            images.append(cv2.imread(os.path.join("memorial", file), cv2.IMREAD_UNCHANGED))
    
    # cv2.imshow("windows", images[0])
    # print(images[0].shape)
    exp_time = [1 / t for t in exp_time]
    aligned_images = align.align(images)
    for i in range(len(aligned_images)):
        aligned_images[i] *= 255
        aligned_images[i] = aligned_images[i].astype(int)
    print(aligned_images[0].shape)
    radiance_maps = HDR(aligned_images, exp_time)
    cv2.imwrite("rad_map.png", np.log(radiance_maps[:,:,0]))

if __name__ == "__main__":
    test()