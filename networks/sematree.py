#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
File: CE2P.py
Author: Me
Email: yourname@email.com
Github: https://github.com/yourname
Description: jijeiji
"""

import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
affine_par = True
import functools

import sys, os

from libs import InPlaceABN, InPlaceABNSync
from .se_module import SELayer
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .aspp import build_aspp

from DCNv2 import DCN

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )

    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv1 = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn1 = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv1, bn1)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [ F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class ResNetTree(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNetTree, self).__init__()
        self.inplanes = 128

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1,1,1))

        # Top layer
        self.toplayer = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Transformer module
        self.tansformer1_l = PSPModule(1024,512)
        self.tansformer1_r = PSPModule(1024,512)

        self.tansformer2_l = PSPModule(512+2,256)
        self.tansformer2_m = PSPModule(512+2,256)
        self.tansformer2_r = PSPModule(512+2,256)

        self.tansformer3_l = PSPModule(256+4,128)
        self.tansformer3_r = PSPModule(256+4,128)

        self.tansformer4_l = PSPModule(256+4,128)
        self.tansformer4_r = PSPModule(256+4,128)

        # self.tansformer1_l = build_aspp(1024, 512, 16, SynchronizedBatchNorm2d)
        # self.tansformer1_r = build_aspp(1024, 512, 16, SynchronizedBatchNorm2d)

        # self.tansformer2_l = build_aspp(512+2, 256, 16, SynchronizedBatchNorm2d)
        # self.tansformer2_m = build_aspp(512+2, 256, 16, SynchronizedBatchNorm2d)
        # self.tansformer2_r = build_aspp(512+2, 256, 16, SynchronizedBatchNorm2d)

        # self.tansformer3_l = build_aspp(256+4, 128, 16, SynchronizedBatchNorm2d)
        # self.tansformer3_r = build_aspp(256+4, 128, 16, SynchronizedBatchNorm2d)

        # self.tansformer4_l = build_aspp(256+4, 128, 16, SynchronizedBatchNorm2d)
        # self.tansformer4_r = build_aspp(256+4, 128, 16, SynchronizedBatchNorm2d)


        # Leaf module
        self.leaf0 = nn.Sequential(
            #nn.Conv2d(512+2, 256, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(512+2, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(256),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 2, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.leaf1 = nn.Sequential(
            #nn.Conv2d(256+4, 128, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(256+4, 128, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(128),
            InPlaceABNSync(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 5, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.leaf2 = nn.Sequential(
            #nn.Conv2d(128+3, 64, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(128+3, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(64),
            InPlaceABNSync(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 5, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.leaf3 = nn.Sequential(
            #nn.Conv2d(128+3, 64, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(128+3, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(64),
            InPlaceABNSync(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 4, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.leaf4 = nn.Sequential(
            #nn.Conv2d(128+3, 64, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(128+3, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(64),
            InPlaceABNSync(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 6, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.leaf5 = nn.Sequential(
            #nn.Conv2d(128+3, 64, kernel_size=3, padding=1, dilation=1, bias=True),
            DCN(128+3, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            SELayer(64),
            InPlaceABNSync(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 4, kernel_size=1, padding=0, dilation=1, bias=False)
            )

        # Router module
        self.router1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, dilation=1, bias=True),
            SELayer(512),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, 2, kernel_size=1, padding=0, dilation=1, bias=False),
            )
        self.router2 = nn.Sequential(
            nn.Conv2d(512+2, 256, kernel_size=1, padding=0, dilation=1, bias=True),
            SELayer(256),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 4, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.router3 = nn.Sequential(
            nn.Conv2d(256+4, 128, kernel_size=1, padding=0, dilation=1, bias=True),
            SELayer(128),
            InPlaceABNSync(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 3, kernel_size=1, padding=0, dilation=1, bias=False)
            )
        self.router4 = nn.Sequential(
            nn.Conv2d(256+4, 128, kernel_size=1, padding=0, dilation=1, bias=True),
            SELayer(128),
            InPlaceABNSync(128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 3, kernel_size=1, padding=0, dilation=1, bias=False)
            )

        # Fusion module
        self.fuse = nn.Sequential(
            DCN(20, 512, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            #nn.Conv2d(20, 512, kernel_size=1, padding=0, dilation=1, bias=True),
            SELayer(512),
            InPlaceABNSync(512),
            nn.Dropout2d(0.5),
            DCN(512, 256, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            #nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=True),
            SELayer(256),
            InPlaceABNSync(256),
            nn.Dropout2d(0.5),
            DCN(256, num_classes, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2),
            #nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False)
            )

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 512+2, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256+4, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 128+3, kernel_size=1, stride=1, padding=0)
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        c5 = self.toplayer(c5)
        # Tree Structure

        # Router Nodes
        # router1
        r1_pred =self.router1(c5)
        r1 = F.softmax(r1_pred, dim=1)
        r1_l, r1_r = torch.chunk(r1, 2, dim=1)

        c6_l = self.tansformer1_l(c5 * r1_l)
        c6_l = torch.cat([c6_l, r1_pred], dim=1)
        c6_l = self._upsample_add(c6_l, self.latlayer1(c4))

        c6_r = self.tansformer1_r(c5 * r1_r)
        c6_r = torch.cat([c6_r, r1_pred], dim=1)
        c6_r = self._upsample_add(c6_r, self.latlayer1(c4))

        # router2
        r2_pred = self.router2(c6_r)
        r2 = F.softmax(r2_pred, dim=1)
        r2_b, r2_l, r2_m, r2_r = torch.chunk(r2, 4, dim=1)

        c7_l = self.tansformer2_l(c6_r * r2_l)
        c7_l = torch.cat([c7_l, r2_pred], dim=1)
        c7_l = self._upsample_add(c7_l, self.latlayer2(c3))

        c7_m = self.tansformer2_m(c6_r * r2_m)
        c7_m = torch.cat([c7_m, r2_pred], dim=1)
        c7_m = self._upsample_add(c7_m, self.latlayer2(c3))

        c7_r = self.tansformer2_r(c6_r * r2_r)
        c7_r = torch.cat([c7_r, r2_pred], dim=1)
        c7_r = self._upsample_add(c7_r, self.latlayer2(c3))

        # router3
        r3_pred = self.router3(c7_m)
        r3 = F.softmax(r3_pred, dim=1)
        r3_b, r3_l, r3_r = torch.chunk(r3, 3, dim=1)

        c8_l = self.tansformer3_l(c7_m * r3_l)
        c8_l = torch.cat([c8_l, r3_pred], dim=1)
        c8_l = self._upsample_add(c8_l, self.latlayer3(c2))

        c8_r = self.tansformer3_r(c7_m * r3_r)
        c8_r = torch.cat([c8_r, r3_pred], dim=1)
        c8_r = self._upsample_add(c8_r, self.latlayer3(c2))

        # router4
        r4_pred = self.router4(c7_r)
        r4 = F.softmax(r4_pred, dim=1)
        r4_b, r4_l, r4_r = torch.chunk(r4, 3, dim=1)

        c9_l = self.tansformer4_l(c7_r * r4_l)
        c9_l = torch.cat([c9_l, r4_pred], dim=1)
        c9_l = self._upsample_add(c9_l, self.latlayer3(c2))

        c9_r = self.tansformer4_r(c7_r * r4_r)
        c9_r = torch.cat([c9_r, r4_pred], dim=1)
        c9_r = self._upsample_add(c9_r, self.latlayer3(c2))

        # Leaf node
        l0_conv = self.leaf0(c6_l)
        l0_conv = F.interpolate(l0_conv, size=(96, 96), mode='bilinear', align_corners=True)
        l0_b, ch0 = torch.chunk(l0_conv, 2, dim=1)
        l0 = F.softmax(l0_conv, dim=1, _stacklevel=3)

        l1_conv = self.leaf1(c7_l)
        l1_conv = F.interpolate(l1_conv, size=(96, 96), mode='bilinear', align_corners=True)
        l1_b, ch1, ch2, ch4, ch13 = torch.chunk(l1_conv, 5, dim=1)
        l1 = F.softmax(l1_conv, dim=1, _stacklevel=3)

        l2_conv = self.leaf2(c8_l)
        l2_b, ch5, ch6, ch7, ch11 = torch.chunk(l2_conv, 5, dim=1)
        l2 = F.softmax(l2_conv, dim=1, _stacklevel=3)

        l3_conv = self.leaf3(c8_r)
        l3_b, ch3, ch14, ch15 = torch.chunk(l3_conv, 4, dim=1)
        l3 = F.softmax(l3_conv, dim=1, _stacklevel=3)

        l4_conv = self.leaf4(c9_l)
        l4_b, ch9, ch10, ch12, ch16, ch17   = torch.chunk(l4_conv, 6, dim=1)
        l4 = F.softmax(l4_conv, dim=1, _stacklevel=3)

        l5_conv = self.leaf5(c9_r)
        l5_b, ch8, ch18, ch19 = torch.chunk(l5_conv, 4, dim=1)
        l5 = F.softmax(l5_conv, dim=1, _stacklevel=3)

        seg = torch.cat([ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10, ch11, ch12, ch13, ch14, ch15, ch16, ch17, ch18, ch19], dim=1)
        seg = self.fuse(seg)
        seg = F.softmax(seg, dim=1, _stacklevel=3)

        return [[seg], [r1, r2, r3, r4], [l0, l1, l2, l3, l4, l5]]

def Res_Deeplab(num_classes=21):
    model = ResNetTree(Bottleneck,[3, 4, 23, 3], num_classes)
    return model

