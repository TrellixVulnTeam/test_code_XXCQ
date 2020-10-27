import cv2
import numpy as np

img1 = cv2.imread('ia_500000113.jpg')
img2 = cv2.imread('ia_500000114.jpg')
img3 = cv2.imread('ia_500000115.jpg')
img4 = cv2.imread('ia_500000116.jpg')
img5 = cv2.imread('ia_500000117.jpg')
vis = np.concatenate((img1, img2, img3, img4, img5), axis=1)
cv2.imwrite('out.jpg', vis)