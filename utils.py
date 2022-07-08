import numpy as np
def ind2sub(H, W, idx):
    return idx % H, idx // H
    pass
if __name__ == "__main__":
    H, W = 3, 4
    a = np.arange(H * W).reshape((W, H)).T
    print(a)
    print(ind2sub(H, W, 10))
