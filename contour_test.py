import cv2
import numpy as np 

xmin = 502
xmax = 570
ymin = 856
ymax = 869


mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

box = np.array([[503.05026,856.9351], [569.1472,858.01874], [568.9785,868.3111],[502.88156,867.2275]])


# (1, 4, 2)
cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 255)

cv2.imwrite('./mask_sample.jpg', mask)




