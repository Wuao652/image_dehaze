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
    hazy_img_path = "/home/dennis/carla_to_rosbag/data/img_files/hazy"
    img_files = find_files(hazy_img_path, exts=['*.png', '*.jpg'])
    os.makedirs(f"/home/dennis/carla_to_rosbag/data/img_files/hazy_trans", exist_ok=True)
    for i in range(len(img_files)):
        print(f"start {str(i)} image ======>>>>>>")
        img = cv2.imread(img_files[i]).astype(np.float64) / 255.
        # img = np.load("./data/image.npy")[..., ::-1].astype(np.float64)

        # # 1. compute the dark channel image of the input image.
        # dark_channel = get_dark_channel(img, 15)
        # # 2. estimate the atmospheric light
        # A = get_atmosphere(img, dark_channel)
        # # 3. estimate the coarse transmission map
        # coarse_t = get_transmission_estimate(img, A, 0.95, 15)

        result, dark, coarse_t, fine_t, A = dehaze(img, 0.95, 15, 0.0001)
        result = np.clip(result, 0., 1.)
        cv2.imwrite(f"./carla_results/dark_r_{str(i).zfill(5)}.png", dark * 255.)
        cv2.imwrite(f"./carla_results/coarse_t_r_{str(i).zfill(5)}.png", coarse_t * 255.)
        cv2.imwrite(f"./carla_results/fine_t_r_{str(i).zfill(5)}.png", fine_t * 255.)
        cv2.imwrite(f"./carla_results/clear_r_{str(i).zfill(5)}.png", result * 255.)

        # print(A)
        # print(A[:, ::-1])
        # np.savetxt(f"/home/dennis/carla_to_rosbag/data/img_files/hazy_trans/a_{str(i).zfill(5)}.txt", A[:, ::-1])
        # cv2.imwrite(f"/home/dennis/carla_to_rosbag/data/img_files/hazy_trans/t_{str(i).zfill(5)}.png", coarse_t * 255.)
