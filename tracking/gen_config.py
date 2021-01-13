import os
import json
import numpy as np
import math


def gen_config(args):
    img_dir = ''
    gt_path = ''
    if args.seq != '':
        # generate config from a sequence name

        seq_home = '/home/lee/Documents/RT-MDNet-master-attention/dataset/OTB2015/'
        result_home = 'results'

        seq_name = args.seq

        if seq_name.startswith('E:\\Temple'):
            img_dir = os.path.join(seq_home, seq_name, 'img')
            gt_path = os.path.join(seq_home, seq_name, seq_name[20:] + '_gt.txt')

        elif seq_name.startswith('E:\\OTB'):
            img_dir = os.path.join(seq_home, seq_name, 'img')
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
        elif seq_name.startswith('E:\\UAV'):
            img_dir = os.path.join(seq_home, seq_name)
            gt_path = 'E:\\UAV123_10fps_O\\anno\\UAV123_10fps\\' + seq_name[40:] + '.txt'
        #elif seq_name.startswith('C:\\Users'):
        else:
            img_dir = os.path.join(seq_home, seq_name)
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')
        #    img_dir = os.path.join(seq_home, seq_name, 'img')
        #    gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
        img_list_o = os.listdir(img_dir)
        img_list =[]
        # check img
        for i in img_list_o:
            if i.endswith('.jpg'):
                img_list.append(i)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]

        with open(gt_path) as f:
            if seq_name.startswith('E:\\UAV'):
                gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
                gt_t = [gt[0]]
                img_list_t = []
                flag = 0
                '''
                for index in range(len(gt)):
                    if math.isnan(gt[index][0]):
                        continue
                    else:
                        gt_t = np.append(gt_t, [gt[index]], axis = 0)
                        img_list_t.append(img_list[index])
                        flag = flag + 1
                gt_t = np.delete(gt_t, 0, axis=0)
                img_list = img_list_t
                gt = gt_t
                '''
                for index in range(len(gt)):
                    if math.isnan(gt[index][0]):
                        gt[index] = [0, 0, 1, 1]
                    else:
                        gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')


            elif seq_name.startswith('C:\\Users') or seq_name.startswith('E:\\vot2016'):
                gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
                x = np.zeros(gt.shape[0] * 4).reshape(gt.shape[0], 4)
                y = np.zeros(gt.shape[0] * 4).reshape(gt.shape[0], 4)

                x[:, 0] = gt[:, 0]
                x[:, 1] = gt[:, 2]
                x[:, 2] = gt[:, 4]
                x[:, 3] = gt[:, 6]
                x_sort = np.sort(x, axis=1)
                width = x_sort[:, 3] - x_sort[:, 0]

                y[:, 0] = gt[:, 1]
                y[:, 1] = gt[:, 3]
                y[:, 2] = gt[:, 5]
                y[:, 3] = gt[:, 7]
                y_sort = np.sort(y, axis=1)
                height = y_sort[:, 3] - y_sort[:, 0]
                # modify coordinate to x,y,w,h
                x[:, 0] = x_sort[:, 0]
                x[:, 1] = y_sort[:, 0]
                x[:, 2] = width
                x[:, 3] = height
                gt = x

            else:
                gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')

        init_bbox = gt[0]

        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(args.savefig):
            os.makedirs(args.savefig)
        savefig_dir = args.savefig
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
