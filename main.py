# coding:utf8

import sys

import cv2
import numpy as np
import math
import pytesseract
import os
from PIL import Image
from pytesseract import Output
import pytesseract
import argparse
import cv2
import imutils
import numpy as np
import easyocr





def findTextRegion(img):
    region = []

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(area)
        if area < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        region.append(box)

    return region





def avaDir(im):
    dir_name = 'output/' + str(im)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

# test
def cropImage(gray):
    threshold = 30
    nrowW = gray.shape[0]
    ncolH = gray.shape[1]

    row = gray[:, int(1 / 2 * nrowW)]
    col = gray[int(1 / 2 * ncolH), :]

    r_flag = np.argwhere(row > threshold)
    c_flag = np.argwhere(col > threshold)

    left, bottom, right, top = r_flag[0, 0], c_flag[-1, 0], r_flag[-1, 0], c_flag[0, 0]
    img = gray[left:right, top:bottom]
    cv2.imwrite('pro2.png', img)
    return img


def pyocr(image):
    # image = cv2.imread("test/181.png")
    h, w = image.shape[:2]
    newImg = np.ones((h, w, 3), dtype=np.uint8) * 255
    print(newImg.shape)
    print(image.shape)

    results = pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')
    print(results)

    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from the current result
        tmp_tl_x = results["left"][i]
        tmp_tl_y = results["top"][i]
        tmp_br_x = tmp_tl_x + results["width"][i]
        tmp_br_y = tmp_tl_y + results["height"][i]
        tmp_level = results["level"][i]
        conf = float(results["conf"][i])
        # print(type(conf))
        text = results["text"][i]
        if float(results["height"][i]) > float(results["width"][i]) * 2:
            continue
        if tmp_level == 4 or tmp_level == 5:
            cv2.rectangle(image, (tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y - 5), (0, 0, 0), 2)
            cv2.rectangle(newImg, (tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y - 5), (0, 0, 0), -1)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("image.png", newImg)
    cv2.imwrite("imageo.png", image)
    return newImg


def resizeShape(img):
    rn = 1000
    # image height and width
    H, W = img.shape[:2]
    # Aspect ratio of the image
    ratio = float(H / W)
    # If any value is less than 1000, then image enlargement is performed
    print(ratio)
    print(W < rn)
    if min(H, W) < rn:
        if ratio >= 1:
            H1 = int(ratio * rn)
            img = cv2.resize(img, (rn, H1))
        else:
            W1 = int(rn / ratio)
            img = cv2.resize(img, (W1, rn))

    print('new img:', img.shape)
    return img


# NMS 方法（Non Maximum Suppression，非极大值抑制）
def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")


def grouper(iterable, interval=2):
    prev = None
    group = []
    for item in iterable:
        if not prev or (abs(item[3] - prev[3]) <= interval):
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def divideRecByAngle(rect_all):
    list = []
    list_0 = []
    list_45 = []
    list_90 = []
    for rect in rect_all:
        if (0 <= rect[2] < 10) or(90 >= rect[2] >= 80):
            list_0.append(rect)
            continue
        if 10 < rect[2] < 50:
            list_45.append(rect)
            continue
        if 50 < rect[2] < 80:
            list_90.append(rect)
            continue
    list.append(list_0)
    list.append(list_45)
    list.append(list_90)
    return list


def divideThreeImg():

    img_0 = cv2.imread('img_test_0.png')
    img_1 = cv2.imread('img_test_1.png')
    img_2 = cv2.imread('img_test_2.png')
    img_3 = cv2.imread('img_test_3.png')
    h, w = img.shape[:2]
    img_angle_all = np.ones((h, w, 3), dtype=np.uint8) * 255


    #img_0
    img_final_gray = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
    ret, binary_final = cv2.threshold(img_final_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes_list = []
    heights = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 0, 0), -1)
        bboxes_list.append([x, y, x + w,
                            y + h])  # Create list of bounding boxes, with each bbox containing the left-top and right-bottom coordinates
        heights.append(h)

    cv2.imwrite('img_rec_final.png', img_final)
    # print(rect_all_final)

    heights = sorted(heights)  # Sort heights
    median_height = heights[len(heights) // 2] / 2  # Find half of the median height
    bboxes_list = sorted(bboxes_list, key=lambda k: k[3])
    combined_bboxes = grouper(bboxes_list, median_height)
    for group in combined_bboxes:
        x_min = min(group, key=lambda k: k[0])[0]  # Find min of x1
        x_max = max(group, key=lambda k: k[2])[2]  # Find max of x2
        y_min = min(group, key=lambda k: k[1])[1]  # Find min of y1
        y_max = max(group, key=lambda k: k[3])[3]  # Find max of y2
        cv2.rectangle(img_0, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
        cv2.rectangle(img_angle_all, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)


    cv2.imwrite('img_angle_all.png', img_angle_all)


    #img_1
    img_gray2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img_gray_1.png', img_gray2)
    ret, binary_rec = cv2.threshold(img_gray2, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    img_1_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    img_1_dilate = cv2.morphologyEx(binary_rec, cv2.MORPH_DILATE, img_1_element, iterations=1)
    img_1_dilate = cv2.morphologyEx(img_1_dilate, cv2.MORPH_CLOSE, img_1_element, iterations=2)

    img_1_region = findTextRegion(img_1_dilate)
    print(img_1_region)
    for box in img_1_region:
        print(box)
        cv2.drawContours(img_1, [box], 0, (127, 0, 0), thickness=-1)
        cv2.drawContours(img_angle_all, [box], 0, (0, 0, 0), thickness=-1)

    cv2.imwrite('img_angle_all.png', img_angle_all)
    #img_2
    img_gray3 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img_gray_1.png', img_gray3)
    ret, binary_rec = cv2.threshold(img_gray3, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    img_2_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    img_2_dilate = cv2.morphologyEx(binary_rec, cv2.MORPH_DILATE, img_2_element, iterations=1)
    img_2_1 = cv2.morphologyEx(img_2_dilate, cv2.MORPH_CLOSE, img_2_element, iterations=2)
    img_2_region = findTextRegion(img_2_1)
    print(img_2_region)
    for box in img_2_region:
        print(box)
        cv2.drawContours(img_2, [box], 0, (0, 127, 70), thickness=-1)
        cv2.drawContours(img_angle_all, [box], 0, (0, 0, 0), thickness=-1)
    cv2.imwrite('img_angle_all.png', img_angle_all)
    return img_angle_all



if __name__ == '__main__':
    # img = cropImage(img)

    # im_list = ['167', 168, 169, 170, 174, 176, 180, 181, 182, 183, 184, 185, 186,187,188]
    im = '167.png'
    imagePath = 'test/' + im
    img = cv2.imread(imagePath)
    img = resizeShape(img)
    # Batch reading and generation of text line images
    dir_name = avaDir(im)
    print('img name：', im)
    print("image size：", img.shape)
    img09 = img.copy()
    img_rec = img.copy()
    # 图像预处理
    # Converting images to grayscale and removing noise using the medianBlur() method
    gray_imag = cv2.cvtColor(img09, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('gray.png', gray_imag)
    cv2.imwrite('crop_image.png', img)

    # mser
    img7 = img.copy()
    mser = cv2.MSER_create()
    regions, bboxs = mser.detectRegions(gray_imag)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img09, hulls, 1, (127, 0, 0))
    keep = []
    for box in bboxs:
        x, y, w, h = box
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(img7,(x,y),(x+w,y+h), (127, 0, 0), 1)
    cv2.imwrite('final_mser.png', img09)
    keep2 = np.array(keep)
    cv2.imwrite('img7.png',img7)
    h, w = img.shape[:2]
    img_rec_1 = np.ones((h, w, 3), dtype=np.uint8) * 255
    for h in hulls:
        rect = cv2.minAreaRect(h)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_rec, [box], 0, (127, 0, 255), 1)
        cv2.drawContours(img_rec_1, [box], 0, (0, 0, 0), -1)
    cv2.imwrite('img_rec.png', img_rec)
    cv2.imwrite('img_rec_1.png', img_rec_1)
    rect_all = []
    img_rec_gray = cv2.cvtColor(img_rec_1, cv2.COLOR_BGR2GRAY)
    ret, binary_rec = cv2.threshold(img_rec_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

    img_rec_dilate = cv2.morphologyEx(binary_rec, cv2.MORPH_DILATE, element2, iterations=1)
    # img_rec_dilate = cv2.morphologyEx(binary_rec, cv2.MORPH_CLOSE, element3, iterations=1)
    img_rec_dilate = cv2.morphologyEx(img_rec_dilate, cv2.MORPH_ERODE, element2, iterations=1)
    img_rec_dilate = cv2.morphologyEx(img_rec_dilate, cv2.MORPH_OPEN, element3, iterations=2)

    cv2.imwrite('img_rec_dilate.png', binary_rec)
    cv2.imwrite('img_rec_dilate.png', img_rec_dilate)
    img_rec_region = findTextRegion(img_rec_dilate)
    #
    for rec in img_rec_region:
        rect = cv2.minAreaRect(rec)
        rect_all.append(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_rec_dilate, [box], 0, (127, 0, 255), 1)
    cv2.imwrite('img_rec_dilate_up.png', img_rec_dilate)

    img_NMS_original = img.copy()
    # final image
    h, w = img.shape[:2]
    img_NMS = np.ones((h, w, 3), dtype=np.uint8) * 255
    box_final = nms(keep2, 0.3)
    for (x, y, x1, y1) in box_final:
        # draw edges on image_NMS_original and draw and Full rectangle of the final output image_NMS
        cv2.rectangle(img_NMS_original, (x, y), (x1, y1), (255, 0, 0), 2)
        cv2.rectangle(img_NMS, (x, y), (x1, y1), (0, 0, 0), 2)
    cv2.imwrite('img_NMS_original.png', img_NMS_original)
    cv2.imwrite('img_NMS.png', img_NMS)

    q = divideRecByAngle(rect_all)
    for i in range(len(q)):
        print(q[i])

    h, w = img.shape[:2]
    img_text = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(len(q)):
        list = q[i]

        img_text1 = np.ones((h, w, 3), dtype=np.uint8) * 255

        print('rectangle number:', len(list))
        for l in list:
            box = cv2.boxPoints(l)
            box = np.int0(box)
            cv2.drawContours(img_text1, [box], 0, (0, 0, 0), -1)
            cv2.drawContours(img_text, [box], 0, (0, 0, 0), -1)
        # Save images in each direction
        cv2.imwrite('img_test_' + str(i) + '.png', img_text1)

        cv2.imwrite('img_test_all' + '.png', img_text)
    h, w = img.shape[:2]
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255

    bboxes_list = []
    heights = []
    # 由于 输入文本主要呈现水平矩形的方式，在经过开闭操作后，如若外接矩阵的角度变化不大，可看作矩阵的书写方式呈水平，于是，可以利用同行上纵坐标的相等去分类矩形。
    # 找到现在的外接矩形
    img_final_gray = cv2.cvtColor(img_text, cv2.COLOR_BGR2GRAY)
    ret, binary_final = cv2.threshold(img_final_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_final = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img_final, (x, y), (x + w, y + h), (0, 0, 0), -1)
        bboxes_list.append([x, y, x + w,
                            y + h])  # Create list of bounding boxes, with each bbox containing the left-top and right-bottom coordinates
        heights.append(h)

    cv2.imwrite('img_rec_final.png', img_final)
    if abs(len(rect_all)) - abs(len(q[0])) < abs(len(rect_all) // 5):
        heights = sorted(heights)  # Sort heights
        median_height = heights[len(heights) // 2] / 2  # Find half of the median height
        bboxes_list = sorted(bboxes_list, key=lambda k: k[3])
        combined_bboxes = grouper(bboxes_list, median_height)
        for group in combined_bboxes:
            x_min = min(group, key=lambda k: k[0])[0]  # Find min of x1
            x_max = max(group, key=lambda k: k[2])[2]  # Find max of x2
            y_min = min(group, key=lambda k: k[1])[1]  # Find min of y1
            y_max = max(group, key=lambda k: k[3])[3]  # Find max of y2
            cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (127,0 , 0), 3)
        cv2.imwrite('vis1.png', vis)
        cv2.imwrite('img_rec1.png', img)

    else :
        img_all_angle = divideThreeImg()
        cv2.imwrite('vis1.png', img_all_angle)
        cv2.imshow('visi',img_all_angle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



