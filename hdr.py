import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import math
import align

def sample(imgs, N=100):
    samples = []
    for img in imgs:
        samples.append(cv2.resize(img,(10, 10)).flatten())
    return np.array(samples, dtype = np.uint8)

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
    lE = x[n + 1:].flatten()

    return g, lE

def gCurve(Z, log_t, l, w):
    gs = []
    
    # b, g, r
    for i in range(3):
        g, _ = gsolve(Z[:, :, i], log_t, l, w)
        gs.append(g)

    plt.plot(gs[0], np.arange(256), "b")
    plt.plot(gs[1], np.arange(256), "g")
    plt.plot(gs[2], np.arange(256), "r")
    plt.title("g")
    plt.show()
    # plt.savefig("basdass.png")
    # plt.clf()    
    return gs

def genlE(Z, log_t, l, g, w):
    P, height, width = Z.shape
    d = np.sum(w[Z], axis=0)
    n = np.sum(w[Z] * (g[Z] - log_t.reshape(P, 1, 1)), axis=0)
    d[d < 0.01] = 1
    return n / d
    

def radianceMap(imgs, log_t, l, g, w):
    P = len(imgs)
    height, width, _ = imgs[0].shape
    maps = []

    for i in range(3):
        # lE.append(genlE(imgs[:, :, :, i], log_t, l, g[i], w))
        maps.append(np.exp(genlE(imgs[:, :, :, i], log_t, l, g[i], w)))

    maps = np.transpose(np.array(maps), (1, 2, 0))
    
    # print(lE[0].shape)
    # maps = np.zeros([height, width, 3], dtype = np.uint8)

    # for i in range(3):
        # maps[:, :, i] = lE[i]

    # maps = np.exp(maps)
    # print(maps.shape, maps)

    return maps
        

def HDR(imgs, expTimes):
    imgs = np.array(imgs, dtype=np.uint8)
    samplePoints = 100
    l = 40
    Z_b = sample(imgs[:, :, :, 0], samplePoints)
    Z_g = sample(imgs[:, :, :, 1], samplePoints)
    Z_r = sample(imgs[:, :, :, 2], samplePoints)
    Z = np.transpose(np.array([Z_b, Z_g, Z_r]), (2, 1, 0))
    log_t = np.log(expTimes)
    w = np.array([i if i <= 128 else 256 - i for i in range(256)], dtype=np.uint8)

    g = gCurve(Z, log_t, l, w)
    rad_map = radianceMap(imgs, log_t, l, g, w)

    return rad_map

def test():
    # img_dir = "./Memorial"
    img_dir = "./pic2"
    images = []
    # exp_time = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]
    exp_time = [0.06666667, 0.00625, 0.5, 0.003125, 0.25, 0.025, 0.0015625, 0.0125, 0.0125, 1.0, 15.0, 2.0, 20.0, 30.0, 4.0, 8.0]
    files = [os.path.join(img_dir, filename) for filename in sorted(os.listdir(img_dir))]
    print(files)
    # input('>')
    for f in files:
        ext = f.split(".")[-1]
        if ext == "png" or ext == "jpg" or ext == "JPG":
            images.append(cv2.imread(f))
    print(f"fuck {images[0].shape}")
    # exp_time = [1 / t for t in exp_time]
    aligned_images = align.align(images)
    rad_map = HDR(aligned_images, exp_time)
    cv2.imwrite("rad_map.hdr", rad_map)
    res = np.log(cv2.cvtColor(rad_map.astype(np.float32), cv2.COLOR_RGB2GRAY))
    # res = cv2.cvtColor(((res + 10 ) / 20  * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    # res = ((res + 10 ) / 20  * 255).astype(np.uint8)
    plt.imshow(res, cmap='jet')
    plt.colorbar()
    plt.show()

    # tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    # res = tonemapDrago.process(res)
    # res = 3 * res
    # cv2.imwrite("ldr-Drago.jpg", res * 255)
    # print(res.dtype)
    # print(type(res))
    # print(np.min(res[:, :, 0]), np.max(res[:, :, 0]))
    # print(np.min(res[:, :, 1]), np.max(res[:, :, 1]))
    # print(np.min(res[:, :, 2]), np.max(res[:, :, 2]))
    # cv2.imwrite("rad_map.png", res) # -10 ~ 10 -> 0 ~ 20 / 20 * 255


if __name__ == "__main__":
    test()