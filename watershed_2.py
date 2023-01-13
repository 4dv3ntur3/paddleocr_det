import cv2
import numpy as np
# import Polygon3 as plg
import matplotlib.pyplot as plg


def watershed(image, viz=False):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("binary", binary)
        cv2.waitKey()
    # 形态学操作，进一步消除图像中噪点
    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    sure_bg = mb

    sure_bg = binary
    if viz:
        cv2.imshow("sure_bg", mb)
        cv2.waitKey()
    # 距离变换
    # dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    # if viz:
    #     cv2.imshow("dist", dist)
    #     cv2.waitKey()
    ret, sure_fg = cv2.threshold(gray, 0.7 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)
    # 获取maskers,在markers中含有种子区域
    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)
    print("nlabels: ", nLabels)
    # 分水岭变换
    markers = labels.copy() + 1
    # markers = markers+1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()

    if viz:
        color_markers = np.uint8(markers + 1)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers1", color_markers)
        cv2.waitKey()

    for i in range(0, np.max(markers)):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        # segmap = np.zeros(gray.shape, dtype=np.uint8)
        # segmap[markers == i] = 255
        # size = np_contours.shape[0]
        # x, y, w, h = cv2.boundingRect(np_contours)
        # if w == 0 or h == 0:
        #     continue
        #
        # niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        # sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # # boundary check
        # if sx < 0: sx = 0
        # if sy < 0: sy = 0
        # if ex >= gray.shape[1]: ex = gray.shape[1]
        # if ey >= gray.shape[0]: ey = gray.shape[0]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        # segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        # np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        # box = np.array(box)
        boxes.append(box)
    return np.array(boxes)



def watershed1(image, viz=False):
    
    score_text_vis = image.copy()

    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    
    cv2.imwrite('./binarized__.jpg', binary)

    # 形态学操作，进一步消除图像中噪点
    # kernel = np.ones((3, 3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    sure_fg = cv2.erode(mb, kernel, iterations=2)

    unknown = cv2.subtract(sure_bg, sure_fg)


    # 距离变换
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)

    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    ret, sure_fg = cv2.threshold(dist, 0.7 * np.max(dist), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换

    unknown = cv2.subtract(sure_bg, surface_fg)
    # 获取maskers,在markers中含有种子区域
    ret, markers = cv2.connectedComponents(surface_fg)

    # 分水岭变换
    markers = markers + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imwrite("watershed_1_color_markers.jpg", color_markers)

    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    
    print(len(markers)) # 312 
    print(np.max(markers)) # 3


    # print(np.max(markers))
    
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
        # score_text_vis_ = score_text_vis.copy()
        # cv2.polylines(score_text_vis_, [np.array(box,dtype=np.uint8)], True, (0, 255, 255), 1)
        # cv2.namedWindow("water", cv2.WINDOW_NORMAL)
        # cv2.imshow("water", score_text_vis_)
        # cv2.waitKey(0)
    return np.array(boxes)

IMG_PATH = "/home/ejpark/paddleocr_det/PaddleOCR/watershed_src/scan_test_1_bitmap.jpg"
# IMG_PATH = "/home/ejpark/paddleocr_det/crop_test.png"
IMG_PATH = "/home/ejpark/paddleocr_det/water_coins.jpg"

if __name__ == "__main__":
    
    img = cv2.imread(IMG_PATH)
    # h, w = img.shape[:2]
    
    # # resize while preserving aspect ratio
    # new_w = 1000
    # ratio = new_w / w
    # new_h = int(h*ratio)

    # dims = (new_w, new_h)
    # img = cv2.resize(img, dims, cv2.INTER_LINEAR)
    
    img_bg = img.copy()
    # boxes = watershed(img)
    boxes = watershed1(img, viz=True)

    print(len(boxes)) # 18
    
    for box in boxes:
        
        color = [int(i) for i in np.random.randint(0, 255, 3)]
        cv2.polylines(img_bg, [box.astype(np.uint)], True, color, 2)
        
    cv2.imwrite('./scan_test_watershed_1.png', img_bg)