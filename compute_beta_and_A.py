import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# rgb_image
rgb_img_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors/r_00004.png'
rgb_img = cv2.imread(rgb_img_file).astype(np.float32) / 255.

# rgb_clear_image
rgb_clear_img_file = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors/r_00004.png'
rgb_clear_img = cv2.imread(rgb_clear_img_file).astype(np.float32) / 255.

# depth_map
depth_map_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors_depth_npy/d_00004.npy'
depth_map = np.load(depth_map_file).astype(np.float32) / 1000

# # only in the red color
# I_r = rgb_img[..., 0].reshape(-1)
# J_r = rgb_clear_img[..., 0].reshape(-1)
# d = depth_map.reshape(-1)
# # x = [beta, A_r]
# def fun(x, i, j, d):
#     beta, A_r = x[0], x[1]
#     return j * np.exp(-d * beta) + A_r * (1.0 - np.exp(-d * beta)) - i
#
# # default
# x0 = np.array([0.5, 0.5], dtype=np.float32)
# res_lsq = least_squares(fun, x0, args=(I_r, J_r, d))

H, W, C = rgb_img.shape
I = rgb_img.reshape(-1, C)
J = rgb_clear_img.reshape(-1, C)
d = depth_map.reshape(-1)

# x = [beta, A_r, A_g, A_b]
def fun(x, i, j, d):
    beta, A_r, A_g, A_b = x[0], x[1], x[2], x[3]
    i_r, i_g, i_b = i[..., 0], i[..., 1], i[..., 2]
    j_r, j_g, j_b = j[..., 0], j[..., 1], j[..., 2]
    t = np.exp(-d * beta)
    res_r = j_r * t + A_r * (1.0 - t) - i_r
    res_g = j_g * t + A_g * (1.0 - t) - i_g
    res_b = j_b * t + A_b * (1.0 - t) - i_b
    return np.concatenate([res_r, res_g, res_b])


x0 = np.array([20., 1., 1., 1.], dtype=np.float32)
b=([0., 0., 0., 0.], [np.inf, 1., 1., 1.])
res_lsq = least_squares(fun, x0, bounds=b, args=(I, J, d))

beta, air_light = res_lsq.x[0], res_lsq.x[1:]
print(beta)
print(air_light)
t = np.exp(-beta * depth_map)
result = (rgb_img - air_light) / t.reshape(480, 640, -1) + air_light
plt.figure()
plt.imshow(rgb_clear_img[..., ::-1] - result[..., ::-1])
plt.show()

