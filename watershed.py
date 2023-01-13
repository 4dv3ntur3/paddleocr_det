### affinity score -> 이거는 이미 character-level annotation이 있어야 가능한거네 
### 이러한 문제를 해결하기 위해 CRAFT 연구팀에서는 기존의 Word-level Annotation이 포함된 데이터 집합을 활용하여 Character-box를 생성하는 방법을 제안했다.
### watershed algorithm 


import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "/home/ejpark/test.png"

img = cv2.imread(IMG_PATH)
score_text_vis = img.copy()
h, w = img.shape[:2]

# resize while preserving aspect ratio
new_w = 1000
ratio = new_w / w
new_h = int(h*ratio)

dims = (new_w, new_h)
img = cv2.resize(img, dims, cv2.INTER_LINEAR)


# binaray image로 변환
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#Morphology의 opening, closing을 통해서 노이즈나 Hole제거
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

# dilate를 통해서 확실한 Backgroud
sure_bg = cv2.dilate(opening,kernel,iterations=3) # 전체 객체들의 경계 (border)

#distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
# 즉, 중심으로 부터 점점 옅어져 가는 영상.
# 그 결과에 thresh를 이용하여 확실한 FG를 파악
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5) # 각 픽셀과 가장 가까운 0인 픽셀과의 거리 계산 
ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
sure_fg = np.uint8(sure_fg)

# Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
unknown = cv2.subtract(sure_bg, sure_fg)

# FG에 Labelling작업
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0


color_markers = np.uint8(markers)
color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET) # region map
# cv2.imwrite('./craft_colormap.jpg', color_markers)


# watershed를 적용하고 경계 영역에 색지정
markers = cv2.watershed(img,markers)
img[markers == -1] = [0, 0, 255] # -1인 곳에 색칠

boxes = []
for i in range(0, np.max(markers) + 1):
    np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
    # print(np_contours.shape)
    rectangle = cv2.minAreaRect(np_contours)
    box = cv2.boxPoints(rectangle)
    w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
    box_ratio = max(w, h) / (min(w, h) + 1e-5)
    if abs(1 - box_ratio) <= 0.1:
        l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
        t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
        box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

    # make clock-wise order
    startidx = box.sum(axis=1).argmin()
    box = np.roll(box, 4 - startidx, 0)
    box = np.array(box)
    boxes.append(box)
    score_text_vis_ = score_text_vis.copy()
    cv2.polylines(score_text_vis_, [np.array(box,dtype=np.int)], True, (0, 255, 255), 1)
    # cv2.namedWindow("water", cv2.WINDOW_NORMAL)
    # cv2.imshow("water", score_text_vis_)
    # cv2.waitKey(0)
    print(np.array(boxes))


print(len(boxes)) # 99개???? 개많은디 


for box in boxes:
    
    cv2.polylines(img, [box.astype(np.int)], True, (0, 255, 255), 1)
    


cv2.imwrite('./craft_watershed_save.png', img)



# images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, img]
# titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']

# for i in range(len(images)):
#     plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

# plt.savefig('./watershed.png', dpi=300)