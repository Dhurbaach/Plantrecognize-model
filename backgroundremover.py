import cv2
import numpy as np

img = cv2.imread("28.jpg")

mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Rectangle around subject (adjust manually)
rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_fg = img * mask2[:, :, np.newaxis]

# White background
white_bg = np.ones_like(img) * 255
result = img_fg + white_bg * (1 - mask2[:, :, np.newaxis])

cv2.imwrite("output_white_bg.jpg", result)