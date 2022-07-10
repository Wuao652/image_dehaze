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
    hazy_img_path = "/home/dennis/nerfplusplus/data/carla_data/hazy/9actors"
    img_files = find_files(hazy_img_path, exts=['*.png', '*.jpg'])
    os.makedirs(f"./carla_results", exist_ok=True)
    for i in range(10):
        print(f"start {str(i)} image ======>>>>>>")
        img = cv2.imread(img_files[i]).astype(np.float64) / 255.
        # img = np.load("./data/image.npy")[..., ::-1].astype(np.float64)
        result, dark, coarse_t, fine_t, A = dehaze(img, 0.95, 15, 0.0001)
        result = np.clip(result, 0., 1.)
        cv2.imwrite(f"./carla_results/dark_r_{str(i).zfill(5)}.png", dark * 255.)
        cv2.imwrite(f"./carla_results/coarse_t_r_{str(i).zfill(5)}.png", coarse_t * 255.)
        cv2.imwrite(f"./carla_results/fine_t_r_{str(i).zfill(5)}.png", fine_t * 255.)
        cv2.imwrite(f"./carla_results/clear_r_{str(i).zfill(5)}.png", result * 255.)
        break
