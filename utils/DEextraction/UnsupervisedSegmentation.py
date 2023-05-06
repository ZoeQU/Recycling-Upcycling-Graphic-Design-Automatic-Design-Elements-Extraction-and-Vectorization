# from __future__ import print_function
# from __future__ import division
import argparse
import setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
import math
import tensorwatch as tw
import hiddenlayer as h
import cv2
# print(cv2.__version__)
import imutils
from scipy import stats
from matplotlib import pyplot as plt
import sys
import os
import numpy as np
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
# from GraphBasedImageSegmentation import EfficientGraphSegmentation as Eff


def plot_loss(n, processPath, color):
    if color:
        path = processPath + 'loss2/epoch_{}.npy'
        plt_title = 'w/ Color Consistency Loss'
        figurename = processPath + '/loss2/loss_color_consistency.png'
    else:
        path = processPath + 'loss1/epoch_{}.npy'
        plt_title = 'Cross Entropy Loss'
        figurename = processPath + '/loss1/loss_ori.png'

    y = []
    for i in range(0, n):
        enc = np.load(path.format(i))
        tempy = enc.tolist()
        y.append(tempy)
    x = range(0, len(y))
    plt.plot(x, y, '.-')
    plt.title(plt_title)
    # plt.xlabel('per {} times'.format(str(n)))
    plt.xlabel('per each time')
    plt.ylabel('LOSS')
    plt.savefig(figurename)
    # plt.show()
    plt.close('all')


use_cuda = torch.cuda.is_available()

class MyNet(nn.Module):
    """CNN model"""
    def __init__(self, input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 100, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()

        for i in range(2-1):
            self.conv2.append( nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(100) )

        self.conv3 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(2-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ColorConsistencyLoss(nn.Module):
    def __init__(self, alpha_sal=0.3):
        super(ColorConsistencyLoss, self).__init__()
        self.alpha_sal = alpha_sal
        self.loss_fn1 = torch.nn.CrossEntropyLoss()

    def loss_fn2(self, input, img):
        input = input.data.cpu().numpy()
        l_inds = np.unique(input)
        std_all = 0
        for i in range(len(l_inds)):
            std = 0
            mask_01 = np.where(input == l_inds[i], 1, 0)
            for k in range(img.shape[2]):
                img_ = img[:, :, k]
                labels_per_sp = img_ * mask_01.reshape(img_.shape[0], img_.shape[1])
                std_ = np.std(labels_per_sp)
                std += std_
            std_all += std
        return 1 / (1 + math.exp(-std_all))


    def forward(self, output, target, img):
        loss_1 = self.loss_fn1(output, target)
        loss_2 = self.loss_fn2(target, img)
        total_loss = (1 - self.alpha_sal) * loss_1 + self.alpha_sal * loss_2
        return total_loss


def unsupervisedSeg(input, maxIter, dim, name, type, color_consistency, processPath, visualization):
    """load image"""
    im = input
    if not os.path.exists(os.path.join(processPath, 'loss2')):
        os.mkdir(os.path.join(processPath, 'loss2'))

    if not os.path.exists(os.path.join(processPath, 'loss1')):
        os.mkdir(os.path.join(processPath, 'loss1'))

    data = torch.from_numpy(np.array([im.transpose((2, 0, 1)).astype('float32')/255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    if type == 'SLIC':
        """slic, original"""
        seg_map = segmentation.slic(im, compactness=100, n_segments=1000)
        if visualization:
            cv2.imwrite(processPath + "_slic.png", seg_map)

    else:
        """EfficientGraphSegmentation"""
        seg_map = segmentation.felzenszwalb(im, scale=2, sigma=0.5, min_size=2048)  # 此处参数自定义,固定,(im, scale=32, sigma=0.5, min_size=32_ourinput/64_elba)
        if visualization:
            cv2.imwrite(processPath + name + '_out_' + 'felzenszwalb' + '.png', seg_map)

    seg_map = seg_map.flatten()
    l_inds = [np.where(seg_map == u_label)[0] for u_label in np.unique(seg_map)]

    model = MyNet(data.size(1))

    """net structure .pdf"""
    if visualization:
        vis_graph = h.build_graph(model, torch.zeros(dim))
        vis_graph.theme = h.graph.THEMES['blue'].copy()
        vis_graph.save('for_figures/net')

    if use_cuda:
        model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    label_colours = np.random.randint(200, size=(100, 3))  # random 200 color_rgb

    Loss_record = []
    min_delta = 0.01
    record_counter = 0
    for batch_idx in range(maxIter):
        """forwarding"""
        optimizer.zero_grad()
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, 100)  #将tensor的维度换位。
        ignore, target = torch.max(output, 1)   # softmax, torch.max(a, 1): 返回每一行的最大值，且返回索引（返回最大元素在各行的列索引）。
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        """refinement"""
        for i in range(len(l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique(labels_per_sp)
            hist = np.zeros(len(u_labels_per_sp))
            for j in range(len(hist)):
                hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]

        """backward"""
        target = torch.from_numpy(im_target)
        if use_cuda:
            target = target.cuda()
        target = Variable(target)

        if color_consistency:   # new loss
            loss_fn = ColorConsistencyLoss()
            loss = loss_fn(output, target, im)
            Loss0 = loss.data.cpu().numpy()
            np.save(processPath + '/loss2/epoch_{}'.format(batch_idx), Loss0)
        else:   # ori loss
            loss = loss_fn(output, target)
            Loss0 = loss.data.cpu().numpy()
            np.save(processPath + '/loss1/epoch_{}'.format(batch_idx), Loss0)

        Loss_record.append(loss)
        loss.backward()
        optimizer.step()
        # print(batch_idx, '/', maxIter, ':', nLabels, loss.item())

        if nLabels <= 3:
            # print("nLabels", nLabels, "reached minLabels", 3, ".")
            break
        elif record_counter >= 3:
            break
        else:
            pass

        if batch_idx > 0:
            delt = Loss_record[batch_idx - 1] - Loss_record[batch_idx]
            if delt < min_delta:
                record_counter += 1

        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, 100)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()

        im_target_rgb = np.zeros((im_target.shape[0], 3))
        for c in range(im_target.shape[0]):
            im_target_rgb[c] = np.array(label_colours[im_target[c] % 100])
        # im_target_rgb = np.array([label_colours[c % 100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape(im.shape).astype(np.uint8)
        # cv2.imwrite(processPath + name + '_unspervised_segment.png', im_target_rgb)

    """save output image"""
    if visualization:
        plt.imshow(im_target_rgb)
        plt.savefig('./' + processPath + name + "_unsuperseg.png")
        # plt.show()
        plt.close('all')
        # cv2.imwrite('./' + processPath + name + "_unsuperseg.png", im_target_rgb.astype('uint8'))
        plot_loss(len(Loss_record), processPath, color_consistency)
    return im_target





