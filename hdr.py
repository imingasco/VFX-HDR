import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ExifTags

import align
from const import *
import tone_mapping as tm

def sample(imgs, N=100):
    samples = []
    for img in imgs:
        samples.append(cv2.resize(img,(10, 10)).flatten())
    return np.array(samples)

def plot_gcurve(g, output_dir=None):
    plt.plot(g[BLUE], np.arange(256), "b")
    plt.plot(g[GREEN], np.arange(256), "g")
    plt.plot(g[RED], np.arange(256), "r")
    plt.xlabel("$ln{E_i} + ln{\Delta t_j}$")
    plt.ylabel("$g(Z_ij)$")
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, "gcurves.png"))
    plt.show()
    plt.close()

def gsolve(Z, log_t, l, w):
    n = 256
    N = np.size(Z, 0)
    P = np.size(Z, 1)

    A = np.zeros((N * P + n - 1, n + N), dtype=np.float32)
    b = np.zeros((A.shape[0], 1), dtype=np.float32)
    k = 0

    for i in range(N):
        for j in range(P):
            wij =  w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k, 0] = wij * log_t[j]
            k += 1
    
    A[k, 127] = 1
    k = k + 1

    for i in range(1, n - 1):
        A[k, i - 1] = w[i] * l
        A[k, i] = -2 * w[i] * l
        A[k, i + 1] = w[i] * l
        k = k + 1

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    g = x[:n].flatten()

    return g

def gcurve(Z, log_t, l, w, plot=False, output_dir=None):
    gs = []
    
    # b, g, r
    for i in range(3):
        gs.append(gsolve(Z[:, :, i], log_t, l, w))
    
    if plot:
        plot_gcurve(gs, output_dir=output_dir)

    return gs

def genlE(Z, log_t, l, g, w):
    P, _, _ = Z.shape
    d = np.sum(w[Z], axis=0)
    n = np.sum(w[Z] * (g[Z] - log_t.reshape(P, 1, 1)), axis=0)
    d[d < 0.01] = 1
    return n / d
    

def radiance_map(imgs, log_t, l, g, w):
    maps = []

    for i in range(3):
        maps.append(np.exp(genlE(imgs[:, :, :, i], log_t, l, g[i], w)))
    maps = np.transpose(np.array(maps), (1, 2, 0))

    return maps     

def HDR(imgs, exposure, plot=False, l=50, output_dir=None):
    imgs = np.array(imgs, dtype=np.uint8)
    samplePoints = 100

    Z_b = sample(imgs[:, :, :, BLUE], samplePoints)
    Z_g = sample(imgs[:, :, :, GREEN], samplePoints)
    Z_r = sample(imgs[:, :, :, RED], samplePoints)
    Z = np.transpose(np.array([Z_b, Z_g, Z_r]), (2, 1, 0))
    log_t = np.log(exposure)
    w = np.array([i if i <= 128 else 256 - i for i in range(256)], dtype=np.float32)

    g = gcurve(Z, log_t, l, w, plot, output_dir=output_dir)
    rad_map = radiance_map(imgs, log_t, l, g, w)

    return rad_map.astype(np.float32)

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
    if type(args.l) == str:
        args.l = int(args.l)
    rad_map = HDR(images, exposure, plot=args.plot, l=args.l, output_dir=args.output)
    print("HDR done.")

    if args.plot:
        print("Showing the radiance map...")
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(np.log(rad_map)[:, :, 0], cmap="jet")
        ax[0].set_title("B channel")
        ax[1].imshow(np.log(rad_map)[:, :, 1], cmap="jet")
        ax[1].set_title("G channel")
        ax[2].imshow(np.log(rad_map)[:, :, 2], cmap="jet")
        ax[2].set_title("R channel")
        plt.show()
        plt.close()
    
    hdr_path = os.path.join(args.output, args.hdr)
    print(f"Saving hdr result to {hdr_path}...")
    cv2.imwrite(hdr_path, rad_map)

    print("Performing tone mapping...")
    print("Performing Reinhard's algorithm (SIGGRAPH 2002)...")
    reinhard = tm.Reinhard_2022(rad_map, alpha=0.18)
    global_result = reinhard.global_op()
    reinhard.adjust_alpha(0.09)
    local_result = reinhard.dodge_and_burn()
    global_ldr_path = os.path.join(args.output, f"reinhard_2002_global_{args.ldr}.png")
    local_ldr_path = os.path.join(args.output, f"reinhard_2002_local_{args.ldr}.png")
    print(f"Saving ldr result to {global_ldr_path} and {local_ldr_path}...")
    cv2.imwrite(global_ldr_path, global_result)
    cv2.imwrite(local_ldr_path, local_result)

    print("Performing Reinhard's algorithm (SIGGRAPH 2005)...")
    result = tm.ReinhardTonemap(rad_map)
    ldr_path = os.path.join(args.output, f"reinhard_2005_{args.ldr}.png")
    print(f"Saving ldr result to {ldr_path}...")
    cv2.imwrite(ldr_path, result)

def test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to the directory containing source images", required=True)
    parser.add_argument("-o", "--output", help="Path to the directory for the output images", required=True)
    parser.add_argument("--hdr", help="Desired file name for the output hdr image", default="result.hdr")
    parser.add_argument("--ldr", help="Desired file name suffix (excluding extension) for the output ldr image", default="result")
    parser.add_argument("-a", "--align", help="Images will be aligned before performing HDR if specified", action="store_true")
    parser.add_argument("-p", "--plot", help="gcurves and radiance maps will be shown if specified", action="store_true")
    parser.add_argument("-t", "--test", help="test mode", action="store_true")
    parser.add_argument("-l", help="smoothing factor of the hdr function", default=30)
    parser.add_argument("-s", "--shift", help="maximum shift for MTB algorithm", default=32)
    parser.add_argument("--scale", help="maximum scale of Reinhard's doged and burn algorithm", default=5)
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main(args)
