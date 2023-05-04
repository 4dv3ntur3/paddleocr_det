import cv2
import matplotlib.pyplot as plt
import numpy as np
# import pyplot as plt

image = (cv2.imread('./test.png') * 255.).astype(np.uint8)

# image = cv2.resize(image, (256, 256))

# plt.imshow(image)


blur = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB), ksize=(3,3), sigmaX=0)
plt.imshow(blur)


new_image_width = 500
new_image_height = 500
color = (255, 255, 255)
result = np.full((new_image_height, new_image_width, 3), color, dtype=np.uint8)

# compute center offset
x_center = (new_image_width - 256) // 2
y_center = (new_image_height - 256) // 2

# copy img image into center of result image
result[y_center:y_center+256, 
       x_center:x_center+256] = blur

plt.imshow(result)


mask = (result[..., 0] < 100).astype(np.uint8)

retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

x, y, w, h, area = stats[1]

rectangled_image = cv2.rectangle(result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=3)

plt.imshow(rectangled_image)

cv2.imwrite('./test__.png', rectangled_image)