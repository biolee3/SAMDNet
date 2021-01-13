import torch
import numpy as np
from PIL import Image
import random
import math
def dataAug(img, box,n_samples):
#   width = im.size[0]
#    height = im.size[1]
    area = img.size[0] * img.size[1]
#random-erasing
    er_minarea = 0.02
    er_maxarea = 0.4
    aspect_minratio = 0.3
    re_boxes = np.tile(box[None, :],(int(n_samples/5),1))

    target_area = random.uniform(er_minarea,er_maxarea) * area
    aspect_ratio = random.uniform(aspect_minratio,1/aspect_minratio)
    er_areah = int(round(math.sqrt(target_area * aspect_ratio)))
    er_areaw = int(round(math.sqrt(target_area / aspect_ratio)))

    if er_areaw < box[2] and er_areah < box[3]:
        x1 = random.randint(box[0], box[0] + box[2] - er_areaw)
        y1 = random.randint(box[1], box[1] + box[3] - er_areah)
