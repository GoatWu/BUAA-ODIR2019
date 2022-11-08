import cv2
import numpy as np

threshold = 3
img = cv2.imread('./420_left.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

h, w = img_gray.shape[0], img_gray.shape[1]
index = np.argwhere(img_gray > threshold)
print(index)
hmin = index[:, 0].min()
hmax = index[:, 0].max()
print(hmin, hmax)
wmin = index[:, 1].min()
wmax = index[:, 1].max()
print(wmin, wmax)

cv2.imwrite('420_left_new.jpg', img[hmin:hmax, wmin:wmax])
