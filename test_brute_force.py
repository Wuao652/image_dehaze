import os
import numpy as np
import matplotlib.pyplot as plt
from main import *

# image
img_file = '/home/dennis/carla_to_rosbag/data/img_files/hazy/r_00173.png'
img = cv2.imread(img_file).astype(np.float64) / 255.

# dcp result
target_file = '/home/dennis/image_dehaze/carla_results/clear_r_00173.png'
dcp_img = cv2.imread(target_file).astype(np.float64) / 255.

dark_channel = get_dark_channel(img, 15)
A = get_atmosphere(img, dark_channel)

# depth
depth_file = '/home/dennis/carla_to_rosbag/data/img_files/hazy_depth_npy/d_00173.npy'
depth_map = np.load(depth_file).astype(np.float64)
trans_map = np.exp(-0.025 * depth_map)

radiance = get_radiance(img, trans_map, A)
plt.figure()
plt.imshow(dcp_img[..., ::-1])
plt.show()
plt.figure()
plt.imshow(radiance[..., ::-1])
plt.show()