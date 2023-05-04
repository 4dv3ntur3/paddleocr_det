import cv2
import numpy as np
import random
import os


dir = "C/pdf_to_img_2022-11-03__15-49-43_MASK/a.jpg"

print(os.sep.join(dir.split(os.sep)[:-1]))
quit()


src_path = "hsv_test/infer_src_001.png"
dest_path = "hsv_test"

src = cv2.imread(src_path)

fn, ext = os.path.splitext(src_path)


for step in range(0, 540, 30):
    
    img = src.copy()
    img = img.astype('float32')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] += step
    bgr_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    grayscale = cv2.cvtColor(bgr_new, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite('./hsv_test/hsv_value_{}_added_{}'.format(step, ext), bgr_new)
    cv2.imwrite('./hsv_test/hsv_value_{}_added_grayscale_{}'.format(step, ext), grayscale)