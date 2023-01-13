import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


IMG_PATH = "/home/ejpark/paddleocr_det/water_coins.jpg"
IMG_PATH = "/home/ejpark/paddleocr_det/binarized.jpg"


fn =IMG_PATH.split(os.sep)[-1]
fn, fn_ext = os.path.splitext(fn)


img = cv2.imread(IMG_PATH)

# color to gray
thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imwrite('{}_thresh.jpg'.format(fn), thresh)


# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)


# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.imwrite('./{}_dist_transform.jpg'.format(fn), dist_transform)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# dark blue = unkown
cv2.imwrite('coins_markers.jpg', markers)

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

print(len(markers.shape)) # 2
print(len(img.shape)) # 3 

cv2.imwrite('{}_result.jpg'.format(fn), img)
cv2.imwrite('{}_markers_last.jpg'.format(fn), markers)