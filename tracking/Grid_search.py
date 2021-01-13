import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def grid_search(image_path_current, previous_bbox):
    target_img = cv2.imread(image_path_current)
    target_bbox = previous_bbox
#    gt_path = 'E:\OTB2015\CarScale\groundtruth_rect.txt'

#    with open(gt_path) as f:
#        gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
#    target_bbox = gt[0]

    img_h = target_img.shape[0]
    img_w = target_img.shape[1]

    scale_factor = 1.0
    #create grid
    grid_num = 3

    grid_w = target_bbox[2] * scale_factor
    grid_h = target_bbox[3] * scale_factor

    left_border = target_bbox[0] - grid_w * (grid_num//2)
    if left_border < 0:
        left_border = 0
    right_border = target_bbox[0] + grid_w * (grid_num//2)
    if right_border > img_w:
        right_border = img_w - grid_w * (grid_num//2) - 1
    top_border = target_bbox[1] - grid_h * (grid_num//2)
    if top_border < 0:
        top_border = 0
    bottom_border = target_bbox[1] + grid_h * (grid_num//2)
    if bottom_border > img_h:
        bottom_border = img_h - grid_h * (grid_num//2) - 1

    outer_ring = []
    outer_ring.append([left_border, top_border, grid_w, grid_h])
    outer_ring.append([target_bbox[0], top_border, grid_w, grid_h])
    outer_ring.append([right_border, top_border, grid_w, grid_h])
    outer_ring.append([left_border, target_bbox[1], grid_w, grid_h])
    outer_ring.append([target_bbox[0], target_bbox[1], grid_w, grid_h])
    outer_ring.append([right_border, target_bbox[1], grid_w, grid_h])
    outer_ring.append([left_border, bottom_border, grid_w, grid_h])
    outer_ring.append([target_bbox[0], bottom_border, grid_w, grid_h])
    outer_ring.append([right_border, bottom_border, grid_w, grid_h])
    outer_ring = np.array(outer_ring)
    '''
    dpi = 80.0
    figsize = (img_w / dpi, img_h / dpi)
    fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(target_img, aspect='auto')
    for s_i in range(len(outer_ring)):
        samples_rect = plt.Rectangle(tuple(outer_ring[s_i, :2]), outer_ring[s_i, 2], outer_ring[s_i, 3], linewidth=1,
                                     edgecolor="#00ffff", zorder=1, fill=False)
        ax.add_patch(samples_rect)
    plt.pause(.01)
    plt.draw()
    plt.close()
    '''
    return outer_ring

# draw grid
target_img = cv2.imread('E:\OTB2015\Matrix\img\\0038.jpg')
previous_img = cv2.imread('E:\OTB2015\Matrix\img\\0037.jpg')
gt_path = 'E:\OTB2015\Matrix\groundtruth_rect.txt'
with open(gt_path) as f:
    gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
target_bbox = gt[37]
#    gt_path = 'E:\OTB2015\CarScale\groundtruth_rect.txt'

#    with open(gt_path) as f:
#        gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
#    target_bbox = gt[0]

img_h = target_img.shape[0]
img_w = target_img.shape[1]

scale_factor = 1.0
# create grid
grid_num = 3

grid_w = target_bbox[2] * scale_factor
grid_h = target_bbox[3] * scale_factor

left_border = target_bbox[0] - grid_w * (grid_num // 2)
if left_border < 0:
    left_border = 0
right_border = target_bbox[0] + grid_w * (grid_num // 2)
if right_border > img_w:
    right_border = img_w - grid_w * (grid_num // 2) - 1
top_border = target_bbox[1] - grid_h * (grid_num // 2)
if top_border < 0:
    top_border = 0
bottom_border = target_bbox[1] + grid_h * (grid_num // 2)
if bottom_border > img_h:
    bottom_border = img_h - grid_h * (grid_num // 2) - 1

outer_ring = []
outer_ring.append([left_border, top_border, grid_w, grid_h])
outer_ring.append([target_bbox[0], top_border, grid_w, grid_h])
outer_ring.append([right_border, top_border, grid_w, grid_h])
outer_ring.append([left_border, target_bbox[1], grid_w, grid_h])
outer_ring.append([target_bbox[0], target_bbox[1], grid_w, grid_h])
outer_ring.append([right_border, target_bbox[1], grid_w, grid_h])
outer_ring.append([left_border, bottom_border, grid_w, grid_h])
outer_ring.append([target_bbox[0], bottom_border, grid_w, grid_h])
outer_ring.append([right_border, bottom_border, grid_w, grid_h])
outer_ring = np.array(outer_ring)
dpi = 80.0
figsize = (img_w / dpi, img_h / dpi)
fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
im = ax.imshow(target_img, aspect='auto')
for s_i in range(len(outer_ring)):
    samples_rect = plt.Rectangle(tuple(outer_ring[s_i, :2]), outer_ring[s_i, 2], outer_ring[s_i, 3], linewidth=1, edgecolor="#00ffff", zorder=1, fill=False)
    ax.add_patch(samples_rect)
plt.pause(.01)
plt.draw()


