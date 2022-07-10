import numpy as np
import os
import imageio
import cv2
import glob
from main import *
import matplotlib.pyplot as plt

def find_files(dir, exts):
    if os.path.isdir(dir):
        # types should be ['*.png', '*.jpg']
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

if __name__ == "__main__":
    # hazy_img_path = "/home/dennis/nerfplusplus/data/carla_data/hazy/9actors"
    # img_files = find_files(hazy_img_path, exts=['*.png', '*.jpg'])
    # # img = imageio.imread(img_files[0])[..., ::-1].astype(np.float32) / 255.
    # img = cv2.imread(img_files[0]).astype(np.float64) / 255.
    # cv2.imshow("input", img)

    img = np.load("./data/image.npy")[..., ::-1].astype(np.float64)
    result, dark, coarse_t, fine_t, A = dehaze(img, 0.95, 15, 0.0001)
    result = np.clip(result, 0., 1.)
    cv2.imshow("dark", dark)
    cv2.imshow("coarse_t", coarse_t)
    cv2.imshow("fine_t", fine_t)
    cv2.imshow('J', result)
    cv2.waitKey()

