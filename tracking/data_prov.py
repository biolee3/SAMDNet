import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import random
import math
from torchvision import transforms
import torch.utils.data as data

from modules.utils import crop_image2


class RegionExtractor():
    def __init__(self, image, samples, opts):
        self.image = np.asarray(image)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
#show_image
#        to_pil_image = transforms.ToPILImage()
#        imgp = to_pil_image(regions[0])
#        imgp.show()
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
class AugRegionExtractor():
    def __init__(self, image, samples, opts):
        self.image = np.asarray(image)
        self.samples = samples

        self.crop_size = opts['img_size']
        self.padding = opts['padding']
        self.batch_size = opts['batch_test']

        self.index = np.arange(len(samples))
        self.pointer = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    def extract_regions(self, index):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)

        regions = regions.transpose(0, 3, 1, 2)
        means = [0.4919, 0.4822, 0.4465]
        for i in range(regions.shape[0]):
            #   width = im.size[0]
            #    height = im.size[1]
            area = self.crop_size * self.crop_size
            # random-erasing
            er_minarea = 0.02
            er_maxarea = 0.4
            aspect_minratio = 0.3
            target_area = random.uniform(er_minarea, er_maxarea) * area
            aspect_ratio = random.uniform(aspect_minratio, 1 / aspect_minratio)
            er_areah = int(round(math.sqrt(target_area * aspect_ratio)))
            er_areaw = int(round(math.sqrt(target_area / aspect_ratio)))

            if er_areaw < self.crop_size and er_areah < self.crop_size:
                x1 = random.randint(0, self.crop_size - er_areaw)
                y1 = random.randint(0, self.crop_size - er_areah)

                regions[i, 0, x1:x1 + er_areaw, y1:y1 + er_areah] = means[0]
                regions[i, 1, x1:x1 + er_areaw, y1:y1 + er_areah] = means[1]
                regions[i, 2, x1:x1 + er_areaw, y1:y1 + er_areah] = means[2]
# show_image
#        regions = regions.transpose(0, 2, 3, 1)
#        to_pil_image = transforms.ToPILImage()
#        imgp = to_pil_image(regions[56])
#        imgp.show()
        regions = regions.astype('float32') - 128.
        return regions