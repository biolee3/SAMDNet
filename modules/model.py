import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                continue
                #raise RuntimeError('Duplicated param name: {:s}'.format(name))
def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
   # optimizer = optim.adam(param_list,lr = lr)
    return optimizer

class ChannelAttention(nn.Module):
    def __init__(self, in_planes= 96, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        #ot = self.avg_pool(x)
        #ot = self.fc1(ot)
        #ot = self.relu1(ot)
        #avg_out = self.fc2(ot)

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class PAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
      #  self.key_conv_test = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        with torch.no_grad():
            m_batchsize, C, height, width = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
    #        test = self.query_conv(x).view(m_batchsize, -1, width*height)
    #        test2 = self.key_conv_test(x).view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        print("Mish activation loaded...")
    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class SAMDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(SAMDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.PReLU(),
                                        nn.LocalResponseNorm(2),
                                        PAM_Module(96),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.PReLU(),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.PReLU(),
                                        )),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.PReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.PReLU()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                    nn.Linear(512, 2)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
 #               w = np.sqrt(2 / m.in_features)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)

        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
#                w = np.sqrt(2 / m.in_features)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
#        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
 #       neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]
        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()


class SDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(SDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.PReLU(),
                                    nn.LocalResponseNorm(2),
                                    PAM_Module(96),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.PReLU(),
                                    nn.LocalResponseNorm(2),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.PReLU(),
                                    )),
            ('fc4', nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                  nn.PReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.PReLU()))]))

#        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
#                                                     nn.Linear(512, 2)) for _ in range(K)])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                #               w = np.sqrt(2 / m.in_features)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)

        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                #                w = np.sqrt(2 / m.in_features)
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
#        for k, module in enumerate(self.branches):
#            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc5'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    x = module(x)
                    return x

        x = self.branches[k](x)
        if out_layer == 'fc6':
            return x
        elif out_layer == 'fc6_softmax':
            return F.softmax(x, dim=1)
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers)

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
