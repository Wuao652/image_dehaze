import os
import numpy as np
import matplotlib.pyplot as plt
from main import *

img_file = '/home/dennis/carla_to_rosbag/data/img_files/hazy/r_00173.png'
img = cv2.imread(img_file).astype(np.float64) / 255.

result, dark, coarse_t, fine_t, A = dehaze(img, 0.95, 15, 0.0001)
print(A)
# plt.subplot(231)
# plt.imshow(img[..., ::-1])
# plt.subplot(232)
# plt.imshow(depth_img)
plt.figure()
plt.imshow(result[..., ::-1])
plt.show()