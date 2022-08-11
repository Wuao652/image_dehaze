import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from main import get_atmosphere, get_dark_channel

# rgb_image
rgb_img_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors/r_00004.png'
rgb_img = cv2.imread(rgb_img_file).astype(np.float32) / 255.

# rgb_clear_image
rgb_clear_img_file = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors/r_00004.png'
rgb_clear_img = cv2.imread(rgb_clear_img_file).astype(np.float32) / 255.

# depth_map
depth_map_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors_depth_npy/d_00004.npy'
depth_map = np.load(depth_map_file).astype(np.float32)

dark_channel = get_dark_channel(rgb_img, 15)
A = get_atmosphere(rgb_img, dark_channel)
A = np.tile(A.reshape((-1, 1, 3)), (480, 640, 1))

trans_map = (rgb_img - A) / (rgb_clear_img - A)
beta = - np.log(trans_map) / depth_map.reshape((480, 640, 1))
print(np.nanmean(beta))
# plt.figure()
# plt.imshow(rgb_img[..., ::-1])
# plt.figure()
# plt.imshow(rgb_clear_img[..., ::-1])
# plt.show()