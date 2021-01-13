#coding=utf-8
import numpy as np
import cv2
def max_overlap(image_path, target_bbox, results_history):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
#center x ,y
    search_factor = 1.5
    center_x = target_bbox[0] + target_bbox[2] / 2
    center_y = target_bbox[1] + target_bbox[3] / 2
# define search boundary
    if center_x + target_bbox[2] * search_factor > image.shape[1]:
        search_area_right = image.shape[1]
    else:
        search_area_right = center_x + target_bbox[2] * search_factor
    if center_x - target_bbox[2] * search_factor < 0:
        search_area_left = 0
    else:
        search_area_left = center_x - target_bbox[2] * search_factor

    if center_y + target_bbox[3] * search_factor > image.shape[0]:
        search_area_bottom = image.shape[0]
    else:
        search_area_bottom = center_y + target_bbox[3] * search_factor
    if center_y - target_bbox[3] * search_factor < 0:
        search_area_top = 0
    else:
        search_area_top = center_y - target_bbox[3] * search_factor
    # draw search boundary
#    start_point = (int(search_area_left), int(search_area_top))
#    end_point = (int(search_area_right), int(search_area_bottom))
    ret, thresh = cv2.threshold(edged, 127, 255, 0)
#   cv2.imshow('gray', thresh)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    # get contours
    imagecontours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
    #    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)
        if x < search_area_left or x + w > search_area_right:
            continue
        elif y < search_area_top or y + h > search_area_bottom:
            continue
        else:
            roi.append([x, y, x + w, y + h])
#            cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)
    if len(roi) >= 2:
        '''
        roi_cluster = np.array(roi)
        roi_x_left_sorted = np.sort(roi_cluster[:, 0])
        roi_y_top_sorted = np.sort(roi_cluster[:, 1])
        roi_x_right_sorted = np.sort(roi_cluster[:, 2])
        roi_y_bottom_sorted = np.sort(roi_cluster[:, 3])

        pre_x_left = int(np.mean(roi_x_left_sorted[0:2]))
        pre_y_top = int(np.mean(roi_y_top_sorted[0:2]))
        pre_x_right = int(np.mean(roi_x_right_sorted[-1:-3:-1]))
        pre_y_bottom = int(np.mean(roi_y_bottom_sorted[-1:-3:-1]))
        pre_weight = pre_x_right - pre_x_left
        pre_height = pre_y_bottom - pre_y_top
 #       print('l:{},r:{}, t:{}, b:{}'.format(pre_x_left,pre_y_top,pre_x_right,pre_y_bottom))
    #smooth update
        target_bbox_area = target_bbox[2] * target_bbox[3]
        target_bbox_ratio = target_bbox[2] / (target_bbox[3] + 0.01)
        previous_area = results_history[2] * results_history[3]
        previous_ratio = results_history[2] / (results_history[3] + 0.01)
        pre_area = pre_weight * pre_height
        pre_ratio = pre_weight / (pre_height + 0.01)

 #       if (previous_ratio / pre_ratio) > 1.5 and previous_ratio > 1:
 #           pre_weight = pre_weight *

        area_ratio = previous_area / (pre_area + 0.01)
        if area_ratio > 1.2:
            predict_box = np.array([pre_x_left, pre_y_top, pre_weight * 1.1, pre_height * 1.1])
            target_bbox = (predict_box + target_bbox) / 2
#            predict_box = results_history
        elif area_ratio < 0.7:
            predict_box = np.array([pre_x_left, pre_y_top, pre_weight * area_ratio, pre_height * area_ratio])
            target_bbox = (predict_box + target_bbox) / 2
#            predict_box = results_history
        else:
            target_bbox = np.array([pre_x_left, pre_y_top, pre_weight, pre_height])
        '''
        target_bbox = np.array([target_bbox[0]*0.97, target_bbox[1]*0.97, target_bbox[2] * 0.968, target_bbox[3] * 0.968])
        return target_bbox
    else:
        return target_bbox

    '''
    cv2.rectangle(thresh, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (255, 255, 255), 2)
    cv2.imshow("img", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.rectangle(edged, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (255, 255, 255), 1)
    cv2.rectangle(image, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (0, 255, 0), 1)
    cv2.imshow("Edge with BB", edged)
    cv2.imshow("Image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
'''
image = cv2.imread('E:\OTB2015\CarScale\img\\0020.jpg')
orig = image.copy()

gt_path = 'E:\OTB2015\CarScale\groundtruth_rect.txt'
with open(gt_path) as f:
    gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
init_bbox = gt[19]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
#center x ,y
target_bbox = init_bbox
center_x = target_bbox[0] + target_bbox[2] / 2
center_y = target_bbox[1] + target_bbox[3] / 2
# define search boundary
if center_x + target_bbox[2] > image.shape[1]:
    search_area_right = image.shape[1]
else:
    search_area_right = center_x + target_bbox[2]
if center_x - target_bbox[2] < 0:
    search_area_left = 0
else:
    search_area_left = center_x - target_bbox[2]

if center_y + target_bbox[3] > image.shape[0]:
    search_area_bottom = image.shape[0]
else:
    search_area_bottom = center_y + target_bbox[3]
if center_y - target_bbox[3] < 0:
    search_area_top = 0
else:
    search_area_top = center_y - target_bbox[3]
# draw search boundary
start_point = (int(search_area_left), int(search_area_top))
end_point = (int(search_area_right), int(search_area_bottom))
ret, thresh = cv2.threshold(edged, 127, 255, 0)
cv2.imshow('gray', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
# get contours
imagecontours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
roi = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
#    if w * h < 100:
#        continue
    cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)
    if x < search_area_left or x + w > search_area_right:
        continue
    elif y < search_area_top or y + h > search_area_bottom:
        continue
    else:
        roi.append([x, y, x + w, y + h])
        cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)

roi_x_left_sorted = sorted(roi, key=(lambda x: x[0]))
roi_y_top_sorted = sorted(roi, key=(lambda x: x[1]))
roi_x_right_sorted = sorted(roi, key=(lambda x: x[2]))
roi_y_bottom_sorted = sorted(roi, key=(lambda x: x[3]))

pre_x_left = int(np.mean(roi_x_left_sorted[0:1], axis=0)[0])
pre_y_top = int(np.mean(roi_y_top_sorted[0:1], axis=0)[1])
pre_x_right = int(np.mean(roi_x_right_sorted[-1:-2:-1], axis=0)[2])
pre_y_bottom = int(np.mean(roi_y_bottom_sorted[-1:-2:-1], axis=0)[3])

predict_box = np.array([pre_x_left, pre_y_top, pre_x_right - pre_x_left, pre_y_bottom - pre_y_top])


cv2.rectangle(thresh, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (255, 255, 255), 2)
cv2.imshow("img", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.rectangle(edged, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (255, 255, 255), 1)
cv2.rectangle(image, (pre_x_left, pre_y_top), (pre_x_right, pre_y_bottom), (0, 255, 0), 1)
cv2.imshow("Edge with BB", edged)
cv2.imshow("Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''