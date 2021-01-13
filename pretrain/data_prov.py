import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import random
import math
from modules.sample_generator import SampleGenerator
from modules.utils import crop_image2
from torchvision import transforms

class RegionDataset(data.Dataset):
    def __init__(self, img_list, gt, opts):
        self.img_list = np.asarray(img_list)
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.batch_pos = opts['batch_pos']
        self.batch_neg = opts['batch_neg']

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('uniform', image.size,
                opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size,
                opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

#            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i)
#            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i)
            n_pos = 6
            n_neg = 10
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos,train_state=True)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples, Aug=True)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions)
        neg_regions = torch.from_numpy(neg_regions)
        return pos_regions, neg_regions

    next = __next__

    def extract_regions(self, image, samples, Aug = False):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)
        if Aug == True:
            means = [0.4919, 0.4822, 0.4465]
#Horizontal Flip
            imgp = Image.fromarray(regions[0])
            imgp = transforms.RandomHorizontalFlip(p=1)(imgp)
            regions[0] = np.array(imgp)
  #          imgp.show()
#Vertical Flip
            imgp = Image.fromarray(regions[1])
            imgp = transforms.RandomVerticalFlip(p=1)(imgp)
            regions[1] = np.array(imgp)
  #          imgp.show()
#Random Rotation
            imgp = Image.fromarray(regions[2])
            imgp = transforms.RandomRotation(180)(imgp)
            regions[2] = np.array(imgp)
  #          imgp.show()
#ColorJitter
            '''
            imgp = Image.fromarray(regions[3])
            imgp = transforms.ColorJitter(brightness=0.5)(imgp)
            regions[3] = np.array(imgp)
 #           imgp.show()
            imgp = Image.fromarray(regions[4])
            imgp = transforms.ColorJitter(brightness=0.5)(imgp)
            regions[4] = np.array(imgp)
#            imgp.show()
#Random Erasing
            '''
            area = self.crop_size * self.crop_size
            er_minarea = 0.02
            er_maxarea = 0.2
            aspect_minratio = 0.3
            target_area = random.uniform(er_minarea, er_maxarea) * area
            aspect_ratio = random.uniform(aspect_minratio, 1 / aspect_minratio)
            er_areah = int(round(math.sqrt(target_area * aspect_ratio)))
            er_areaw = int(round(math.sqrt(target_area / aspect_ratio)))

            if er_areaw < self.crop_size and er_areah < self.crop_size:
                x1 = random.randint(0, self.crop_size - er_areaw)
                y1 = random.randint(0, self.crop_size - er_areah)

                regions[3, x1:x1 + er_areaw, y1:y1 + er_areah, 0] = means[0]
                regions[3, x1:x1 + er_areaw, y1:y1 + er_areah, 1] = means[1]
                regions[3, x1:x1 + er_areaw, y1:y1 + er_areah, 2] = means[2]
            '''
            imgp = Image.fromarray(regions[5])
            regions[5] = np.array(imgp)
  #          imgp.show()
            imgp = Image.fromarray(regions[6])
            regions[6] = np.array(imgp)
 #           imgp.show()
            '''
        per = np.random.permutation(regions.shape[0])
        regions = regions[per, :, :, :]
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.

        return regions
