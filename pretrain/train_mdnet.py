import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np
import collections
import torch
import torch.nn as nn
sys.path.insert(0,'.')
from pretrain.data_prov import RegionDataset
from modules.model import SAMDNet, set_optimizer, BCELoss, Precision

from os.path import *
from torch.autograd import Variable

#dir_path = dirname(dirname(abspath(__file__)))
dir_path = 'E:/SAMDNet/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_mdnet(opts):

    # Init dataset

    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)

    seq_home = 'E:/'
    for seqname, v in data.items():
        for i in range(len(v['images'])):
            v['images'][i] = v['images'][i].replace('datasets/VOT/',seq_home)

    '''
    data = collections.OrderedDict()
    seq_home = '/media/lee/PlatForm/vot2013'
    seq_name = os.listdir(seq_home)
    seq_path = []
    for i in seq_name:
        if i == 'list.txt':
            continue
        seq_path.append(seq_home + '/' + i)
    for j in seq_path:
        img_list = []
        data_1 = {}
        file_list = os.listdir(j)
        file_list.sort()
        gt_path = j + '/' + 'groundtruth.txt'
        with open(gt_path) as f:
            gt = np.loadtxt((x.replace('\t', ',') for x in f), delimiter=',')
        if not seq_home.endswith('2013'):
            gt_trans = []
            for i1 in range(len(gt)):
                X = []
                Y = []
                if len(gt[i1]) > 4:
                    for j1 in range(8):
                        if j1 % 2 == 0:
                            X.append(gt[i1][j1])
                        elif j1 % 2 != 0:
                            Y.append(gt[i1][j1])
                    gt_trans.append([round(min(X)), round(min(Y)), round(max(X) - min(X)), round(max(Y) - min(Y))])
                else:
                    gt_trans.append(gt[i1])
        for i in file_list:
            if os.path.splitext(i)[1] == '.jpg':
                img_list.append(j + '/' + i)
        data_1['images'] = img_list
        data_1['gt'] = gt
        data[j] = data_1
    '''
    K = len(data)
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)

    # Init model
    model = MDNet(opts['init_model_path'], K)

    if opts['use_gpu']:
        model = model.cuda()

    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])
    interDomainCriterion = nn.CrossEntropyLoss()
    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()

            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()

            '''
            pos_logts = model(pos_regions, k, out_layer='fc5')
            neg_logts = model(neg_regions, k, out_layer='fc5')
 #           logts = torch.cat((pos_logts, neg_logts), 0)
            
            labels_p = torch.ones(len(pos_logts), 1).long().cuda()
            labels_n = (torch.ones(len(neg_logts), 1) + 1).long().cuda()
            labels = torch.cat((labels_p, labels_n), 0).cuda()
            labels = labels.long()
            
            p_label = np.array([1] * len(pos_logts))
            p_label = torch.from_numpy(p_label)
            p_label = p_label.type(torch.long).to(device)
  #          p = np.tile(p, (len(pos_logts), 1))

            n_label = np.array([0] * len(neg_logts))
 #           n = np.tile(n, (len(neg_logts), 1))
            n_label = torch.from_numpy(n_label)
            n_label = n_label.type(torch.long).to(device)
          #  labelss = torch.cat((labels_pp, labels_nn), 0).cuda()
           

            margin = ArcMarginProduct(in_feature=512, out_feature=2, s=32.0)
            margin = margin.to(device)
            pos_score = margin(pos_logts, p_label)
            neg_score = margin(neg_logts, n_label)
            '''
            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)
            optimizer.zero_grad()
            cls_loss = criterion(pos_score, neg_score)

            # instance embedding loss

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:,1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0),1))
          
            K_perm = np.random.permutation(K)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if k == cidx:
                    continue
                else:
                    with torch.no_grad():
                        interclass_score = model(pos_regions, cidx)
                        total_interclass_score = torch.cat((total_interclass_score,interclass_score[:, 1].contiguous().view((interclass_score.size(0),1))),dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            totalInterClassLoss[k] = interclass_loss.item()

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            (0.85*cls_loss + 0.15*interclass_loss).backward()
#            cls_loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                 .format(i, opts['n_cycles'], j, len(k_list), k, cls_loss.item(), prec[k], toc))
            #print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), BinLoss {:.3f}, Precision {:.3f}, interLoss {:.3f}, Time {:.3f}'
            #        .format(i, opts['n_cycles'], j, len(k_list), k, cls_loss.item(), prec[k],totalInterClassLoss[k], toc))
        #print('Mean Precision: {:.3f}, Inter Loss: {:.3f}'.format(prec.mean(),totalInterClassLoss.mean()))
        print('Mean Precision: {:.3f}'.format(prec.mean()))

        if seq_home.endswith('2013'):
            train_modeled = dir_path + '/models/mdnet_vot' + '2013' + '-otb.pth'
            print('Save model to {:s}'.format(train_modeled))
        elif seq_home.endswith('2016'):
            train_modeled = dir_path + '/models/mdnet_vot' + '2016' + '-otb.pth'
            print('Save model to {:s}'.format(train_modeled))
        elif seq_home.endswith('2019'):
            train_modeled = dir_path + '/models/mdnet_vot' + '2019' + '-otb.pth'
            print('Save model to {:s}'.format(train_modeled))
        else:
            train_modeled = dir_path + '/models/mdnet_vot_otb_conv1PLM_Prelu_Aug_ArcLoss_58s.pth'
            print('Save model to {:s}'.format(train_modeled))
     
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states,train_modeled)
        #train_modeled = dir_path + 'models/mdnet_imagenet_vid_2015.pth'
        #print('Save model to {:s}'.format(train_modeled))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='imagenet', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open(dir_path + 'pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    train_mdnet(opts)
