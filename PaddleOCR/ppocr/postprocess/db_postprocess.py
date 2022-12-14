# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refered from:
https://github.com/WenmuZhou/DBNet.pytorch/blob/master/post_processing/seg_detector_representer.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import paddle
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    # box_thresh=0.3
    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 score_mode="fast",
                 use_polygon=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.                                score_mode = score_mode
        self.use_polygon = use_polygon
        assert score_mode in [
            "slow", "fast"
        ], "Score mode must be in [slow, fast] but got: {}".format(score_mode)

        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])
        
        ###
        self.counter = 1

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores


    ### boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
    
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, image_pure_name, erode_kernel):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape
        bitmap__ = (bitmap * 255).astype(np.uint8)
        
        ### save pred
        _pred = (pred*255).astype(np.uint8)
        cv2.imwrite('./{}_pred.jpg'.format(image_pure_name), _pred)
    
        ### save bitmap
        cv2.imwrite('./{}_bitmap.jpg'.format(image_pure_name), bitmap__)
        
        ### erosion 
        if erode_kernel:
            kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
            erosion = cv2.erode(bitmap__, kernel, iterations=1)

        # ????????? ????????? ????????? ????????? 
        # outs = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ### preprocess contour
        # # inverted_bitmap = 255 - bitmap__
        # blurred_bitmap = cv2.blur(bitmap__, (2, 2))
        
        contour_input = erosion if erode_kernel else bitmap__
        
        # cv2.imwrite('./{}_origin_mask.jpg'.format(image_pure_name), bitmap__)
        # cv2.imwrite('./{}_blurred_mask.jpg'.format(image_pure_name), blurred_bitmap)

        outs = cv2.findContours(contour_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ###
        # if self.counter == 2:
            
        #     print(outs[1])
        #     quit()
            
        
        # from PIL import Image
        # im = Image.fromarray((bitmap*255).astype(np.uint8))
        # im.save('./img_bitmap_mask_{}.jpg'.format(self.counter))
        
        # save bitmap
        cv2.imwrite('./det_res_{}_mask.jpg'.format(image_pure_name), contour_input)

        
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]
        
        num_contours = min(len(contours), self.max_candidates) # 1000?????? ???????????? ??? ????????????? 
        
        # src_path = './infer_src/scan_test_{}.jpg'.format(self.counter)
        # src_img = cv2.imread()
        
        src_img = cv2.cvtColor((bitmap * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) # 3?????? 

        # for c_idx, cont in enumerate(contours):
            
        #     color = [int(i) for i in np.random.randint(0, 255, 3)]
        #     cv2.drawContours(src_img, contours, c_idx, color, 3)
        #     cv2.putText(src_img, str(c_idx), tuple(cont[1][0]), cv2.FONT_HERSHEY_PLAIN, \
        #                                                         0.5, (0,0,255), 2, 0)

        # cv2.imwrite('./draw_contour_index_{}.jpg'.format(self.counter), src_img)


        boxes = []
        scores = []
        for index in range(num_contours): # countour ????????? box ????????? point ??????, score 
            contour = contours[index]
            
            points, sside = self.get_mini_boxes(contour, src_img) # src_img??? ?????? X 
            
            # if index == 73:
            #     # print(contour)
            #     color = [int(i) for i in np.random.randint(0, 255, 3)]
            #     # cv2.drawContours(src_img, contours, c_idx, color, 3)
            #     for point in points:

            if sside < self.min_size: # 3 ???, width??? height ??? ??? ?????? ????????? 3????????? ????????? ???????????? 
                continue
            
            points = np.array(points)
            # print(points.shape) # 4, 2
            # quit() 
            if self.score_mode == "fast": # 

                score = self.box_score_fast(pred, points.reshape(-1, 2)) # ??? box ??? ?????? score 
            else:
                score = self.box_score_slow(pred, contour)
                
            if self.box_thresh > score: # box thresh??? ??? box ?????? ??? ?????? score??? ?????? 
                continue
            
            # box ?????? ??? unclip (??????)
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            
            # unlcip?????? ?????? ?????? ?????? box ?????? 
            box, sside = self.get_mini_boxes(box) # contour??? ????????? box ????????? ???????????? 
            if sside < self.min_size + 2: # 5?????? ?????? ??? 
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width) # ?????? ?????? 
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
            
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour, src_img=None):
        
        # print(contour) # many points 15, 1, 2
        # print(type(contour))
        # print(contour.shape)
        # quit()
        
        bounding_box = cv2.minAreaRect(contour) 
        # bounding_box[0] = ????????? ????????? ?????? x, y
        # bounding_box[1] = width, height 
        # print(len(bounding_box)) # 3 
        # print(bounding_box[0]) # 1??? ?????? 
        # print(bounding_box[1]) (66.10587310791016, 10.29369831085205)
        
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        
        # print(points) # 2, 4 
        # quit()
        
        # for c_idx, cont in enumerate(points):
        #     color = [int(i) for i in np.random.randint(0, 255, 3)]
        #     cv2.drawContours(src_img, contour, c_idx, color, 3)
        #     cv2.putText(src_img, str(c_idx), tuple(cont[1][0]), cv2.FONT_HERSHEY_PLAIN, \
        #                                                         0.5, (0,0,255), 2, 0)
        # cv2.imwrite('./{}_draw_bbox_{}.jpg'.format(image_pure_name, self.counter), src_img)


        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1]) # width, height ?????? ?????? ???? 

    def box_score_fast(self, bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        
        print(h, w) # 896, 672
        print(box.shape) # 4, 2 ( x, y ?????? 4???) 
        # print(box)
        # quit()
        
        # np.clip(array, min, max)
        
        # box??? ??????
        # box ????????? ??? ?????? x?????? ????????? ????????? -> ?????? -> int??? ?????? -> 0?????? ????????? w -1 ?????? ?????? 0 or w -1 ??? ?????? (?????? ????????? ????????? ??? ???????????????)
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1) 
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)
        
        # contour ??? ???????????? ??? min, max??? ?????? ()
        
        print("xmin")
        print(xmin, xmax, ymin, ymax) # 502, 570, 865, 869
        
        print("box")
        print(box)
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8) # box (w, h)?????? 0?????? ?????? mask (0??? black, 1??? white)
        box[:, 0] = box[:, 0] - xmin # x???????????? xmin ?????? 
        box[:, 1] = box[:, 1] - ymin # y???????????? ymin ?????? 
        
        print('after box')
        print(box)
        
        print("=======================")
        print(mask.shape) # (14, 69)
        print(box.shape) # (4, 2)
        # a = box.reshape(1, -1, 2).astype(np.int32) # (1, 4, 2) # ????????? ????????? ???????????? -1 ????????? ????????? ?????? ?????? ????????? ????????? 

        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1) # mask?????? box ????????? ?????????? (0: black, 255: white)
        # (1, 4, 2) -> 
        print(box)
        
        # cv2.imwrite('./box_masked.jpg', mask)
        # quit()
        # (img, points, color)
        # mask ?????? ????????? ?????????? 1??? ????????? ?????? ?????? ???????? color=1? 
        # print(cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask))
        # print(np.max(bitmap), np.min(bitmap)) # 1.0, 0.0 ??????? 
        # quit()
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0] # bitmap?????? mask ????????? ???????????? ????????? ???????????? ??????? 
        # 4???????????? ????????? ??? ??????, ??? ????????? ????????? 0?????? ?????????. bitmap??? grayscale????????? ?????? 1?????? ??????. ????????? [0]???. 

    def box_score_slow(self, bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list, image_pure_name, erode_kernel): # preds, shape_list
        
        pred = outs_dict['maps'] 
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :] # N, C, H, W
        segmentation = pred > self.thresh # bitmap
        
        # print(pred.shape) # 1, 896, 672
        # segmentation_ = (pred*255).astype(np.uint8)[0]
        # cv2.imwrite("./output_thresh_{}_{}.jpg".format(self.thresh, self.counter), (segmentation*255).astype(np.uint8)[0])
        # # print(np.max(segmentatiion_), np.min(segmentatiion_))
        # print(segmentation) 
        # print(self.dilation_kernel) # None

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else: ###
                mask = segmentation[batch_index]

            # ????????? ?????? ?????? ??????? -> ?????? X 
            '''
            https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html 
            '''
            import os
            image_pure_name = os.path.splitext(image_pure_name)[0]
                
            if self.use_polygon is True:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index],
                                                          mask, src_w, src_h)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, ###
                                                       src_w, src_h, image_pure_name, erode_kernel)

            boxes_batch.append({'points': boxes, 'scores':scores}) ###
            # print(boxes[0])
            
        self.counter += 1
        return boxes_batch


class DistillationDBPostProcess(object):
    def __init__(self,
                 model_name=["student"],
                 key=None,
                 thresh=0.3,
                 box_thresh=0.6,
                 max_candidates=1000,
                 unclip_ratio=1.5,
                 use_dilation=False,
                 score_mode="fast",
                 use_polygon=False,
                 **kwargs):
        self.model_name = model_name
        self.key = key
        self.post_process = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=max_candidates,
            unclip_ratio=unclip_ratio,
            use_dilation=use_dilation,
            score_mode=score_mode,
            use_polygon=use_polygon)

    def __call__(self, predicts, shape_list):
        results = {}
        for k in self.model_name:
            results[k] = self.post_process(predicts[k], shape_list=shape_list)
        return results
