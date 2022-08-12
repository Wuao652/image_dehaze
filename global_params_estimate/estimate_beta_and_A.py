import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import os
import glob
# # rgb_image
# rgb_img_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors/r_00004.png'
# rgb_img = cv2.imread(rgb_img_file).astype(np.float32) / 255.
#
# # rgb_clear_image
# rgb_clear_img_file = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors/r_00004.png'
# rgb_clear_img = cv2.imread(rgb_clear_img_file).astype(np.float32) / 255.
#
# # depth_map
# depth_map_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors_depth_npy/d_00004.npy'
# depth_map = np.load(depth_map_file).astype(np.float32) / 1000

# read rgb, rgb_clear and depth image
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

rgb_img_dir = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors'
rgb_clear_img_dir = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors'
depth_img_dir = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors_depth_npy'

rgb_img_files = find_files(rgb_img_dir, ['*.png', '*.jpg', '*.PNG', '*.JPG'])
rgb_clear_img_files = find_files(rgb_clear_img_dir, ['*.png', '*.jpg', '*.PNG', '*.JPG'])
depth_img_files = find_files(depth_img_dir, ['*.npy'])

cnt = len(depth_img_files)
print(len(rgb_img_files))
print(len(rgb_clear_img_files))
print(len(depth_img_files))

I, J, d = [], [], []
for idx in range(50):
    rgb_img = cv2.imread(rgb_img_files[idx]).astype(np.float32) / 255.
    rgb_clear_img = cv2.imread(rgb_clear_img_files[idx]).astype(np.float32) / 255.
    depth_img = np.load(depth_img_files[idx]).astype(np.float32) / 1000.

    I.append(rgb_img)
    J.append(rgb_clear_img)
    d.append(depth_img)

H, W, C = I[0].shape
# N_img * [480, 640, 3]
I = np.stack(I, axis=0).reshape(-1, C)
J = np.stack(J, axis=0).reshape(-1, C)
d = np.stack(d, axis=0).reshape(-1)

def fun(x, i, j, d):
    beta, A_r, A_g, A_b = x[0], x[1], x[2], x[3]
    i_r, i_g, i_b = i[..., 0], i[..., 1], i[..., 2]
    j_r, j_g, j_b = j[..., 0], j[..., 1], j[..., 2]
    t = np.exp(-d * beta)
    res_r = j_r * t + A_r * (1.0 - t) - i_r
    res_g = j_g * t + A_g * (1.0 - t) - i_g
    res_b = j_b * t + A_b * (1.0 - t) - i_b
    return np.concatenate([res_r, res_g, res_b])

# init all the params to be estimated
# x = [beta, A_r, A_g, A_b]
x0 = np.array([20., 1., 1., 1.], dtype=np.float32)
b = ([0., 0., 0., 0.], [np.inf, 1., 1., 1.])

res_lsq = least_squares(fun, x0, bounds=b, args=(I, J, d))

beta, air_light = res_lsq.x[0], res_lsq.x[1:]
print(beta)
print(air_light)

# t = np.exp(-beta * depth_map)
# result = (rgb_img - air_light) / t.reshape(480, 640, -1) + air_light
#
# residual = rgb_clear_img - result
# plt.figure()
# plt.imshow(residual[..., ::-1])
# plt.show()




