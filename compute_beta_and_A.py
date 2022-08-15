import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# rgb_image
rgb_img_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors/r_00015.png'
rgb_img = cv2.imread(rgb_img_file).astype(np.float32) / 255.

# rgb_clear_image
rgb_clear_img_file = '/home/dennis/nerfplusplus/data/carla_data/gt/9actors/r_00015.png'
rgb_clear_img = cv2.imread(rgb_clear_img_file).astype(np.float32) / 255.

# depth_map
depth_map_file = '/home/dennis/nerfplusplus/data/carla_data/hazy/9actors_depth_npy/d_00015.npy'
depth_map = np.load(depth_map_file).astype(np.float32) / 1000

H, W, C = rgb_img.shape
I = rgb_img.reshape(-1, C)
J = rgb_clear_img.reshape(-1, C)
d = depth_map.reshape(-1)


def fun(x, i, j, d):
    beta, A_r, A_g, A_b = x[0], x[1], x[2], x[3]
    i_r, i_g, i_b = i[..., 0], i[..., 1], i[..., 2]
    j_r, j_g, j_b = j[..., 0], j[..., 1], j[..., 2]
    t = np.exp(-d * beta)
    res_r = j_r * t + A_r * (1.0 - t) - i_r
    res_g = j_g * t + A_g * (1.0 - t) - i_g
    res_b = j_b * t + A_b * (1.0 - t) - i_b
    return np.concatenate([res_r, res_g, res_b])

# set the initial guess
# x = [beta, A_r, A_g, A_b]
x0 = np.array([20., 1., 1., 1.], dtype=np.float32)

# set the bounds of the variables
b=([0., 0., 0., 0.], [np.inf, 1., 1., 1.])

# solve NLS
res_lsq = least_squares(fun, x0, bounds=b, args=(I, J, d))

beta, air_light = res_lsq.x[0], res_lsq.x[1:]
print(beta)
print(air_light)

# beta, air_light = 38.77924838921209, np.array([0.75376985, 0.7725226, 0.79939912], dtype=np.float32)
t = np.exp(-beta * depth_map)
result = (rgb_img - air_light) / t.reshape(H, W, -1) + air_light

residual = rgb_clear_img - result

plt.figure()
plt.subplot(221)
plt.imshow(rgb_clear_img[..., ::-1])
plt.title('clean image')
plt.axis('off')
plt.subplot(222)
plt.imshow(rgb_img[..., ::-1])
plt.title('hazy image')
plt.axis('off')
plt.subplot(223)
plt.imshow(result[..., ::-1])
plt.title('dehazed image')
plt.axis('off')
plt.subplot(224)
plt.imshow(residual[..., ::-1])
plt.title('residual image')
plt.axis('off')
plt.figtext(0.5, 0.06,
            f"The estimated beta : {beta:.2f}",
            ha="center",
            fontsize=10,
            )
plt.figtext(0.5, 0.02,
            f"The estimated air_light : [{air_light[0]:.2f}, {air_light[1]:.2f}, {air_light[2]:.2f}]",
            ha="center",
            fontsize=10,
            )
plt.show()

