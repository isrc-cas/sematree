import os
import numpy as np
import random
import torch
import cv2
import json
from torch.utils import data
from utils.transforms import get_affine_transform

LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']


'''
(0, 'Background')
(1, 'Hat')
(2, 'Hair')
(3, 'Glove')
(4, 'Sunglasses')
(5, 'Upper-clothes')
(6, 'Dress')
(7, 'Coat')
(8, 'Socks')
(9, 'Pants')
(10, 'Jumpsuits')
(11, 'Scarf')
(12, 'Skirt')
(13, 'Face')
(14, 'Left-arm')
(15, 'Right-arm')
(16, 'Left-leg')
(17, 'Right-leg')
(18, 'Left-shoe')
(19, 'Right-shoe')
'''

class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        :rtype:
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset

        list_path = os.path.join(self.root, self.dataset + '_id.txt')

        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]

        im_path = os.path.join(self.root, self.dataset + '_images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train' or self.dataset == 'trainval':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]

                    center[0] = im.shape[1] - center[0] - 1
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset != 'train':
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))

            label_r1 = label_parsing.copy()
            label_r1[label_r1 == 0] = 0
            label_r1[label_r1 != 0] = 1

            label_l0 = label_parsing.copy()
            chnl_l0_0 = np.where(label_l0 != 0)
            chnl_l0_1 = np.where(label_l0 == 0)
            label_l0[chnl_l0_0] = 0
            label_l0[chnl_l0_1] = 1

            label_r2 = label_parsing.copy()
            chnl_r2_t0 = np.array([0])
            chnl_r2_t1 = np.array([1,2,4,13])
            chnl_r2_t2 = np.array([3,5,6,7,11,14,15])
            chnl_r2_t3 = np.array([8,9,10,12,16,17,18,19])
            chnl_r2_p0 = np.isin(label_r2, chnl_r2_t0)
            chnl_r2_p1 = np.isin(label_r2, chnl_r2_t1)
            chnl_r2_p2 = np.isin(label_r2, chnl_r2_t2)
            chnl_r2_p3 = np.isin(label_r2, chnl_r2_t3)
            label_r2[chnl_r2_p0] = 0
            label_r2[chnl_r2_p1] = 1
            label_r2[chnl_r2_p2] = 2
            label_r2[chnl_r2_p3] = 3


            label_l1 = label_parsing.copy()
            label_l1[~chnl_r2_p1]=0
            label_l1[label_l1==1]=1
            label_l1[label_l1==2]=2
            label_l1[label_l1==4]=3
            label_l1[label_l1==13]=4

            label_r3 = label_parsing.copy()
            chnl_r3_t0 = np.array([0,1,2,4,13,8,9,10,12,16,17,18,19])
            chnl_r3_t1 = np.array([5,6,7,11])
            chnl_r3_t2 = np.array([3,14,15])
            chnl_r3_p0 = np.isin(label_r3, chnl_r3_t0)
            chnl_r3_p1 = np.isin(label_r3, chnl_r3_t1)
            chnl_r3_p2 = np.isin(label_r3, chnl_r3_t2)
            label_r3[chnl_r3_p0] = 0
            label_r3[chnl_r3_p1] = 1
            label_r3[chnl_r3_p2] = 2

            label_l2 = label_parsing.copy()
            label_l2[~chnl_r3_p1]=0
            label_l2[label_l2 == 5]=1
            label_l2[label_l2 == 6]=2
            label_l2[label_l2 == 7]=3
            label_l2[label_l2 == 11]=4

            label_l3 = label_parsing.copy()
            label_l3[~chnl_r3_p2]=0
            label_l3[label_l3 == 3]=1
            label_l3[label_l3 == 14]=2
            label_l3[label_l3 == 15]=3

            label_r4 = label_parsing.copy()
            chnl_r4_t0 = np.array([0,1,2,4,13,3,5,6,7,11,14,15])
            chnl_r4_t1 = np.array([9,10,12,16,17])
            chnl_r4_t2 = np.array([8,18,19])
            chnl_r4_p0 = np.isin(label_r4, chnl_r4_t0)
            chnl_r4_p1 = np.isin(label_r4, chnl_r4_t1)
            chnl_r4_p2 = np.isin(label_r4, chnl_r4_t2)
            label_r4[chnl_r4_p0] = 0
            label_r4[chnl_r4_p1] = 1
            label_r4[chnl_r4_p2] = 2

            label_l4 = label_parsing.copy()
            label_l4[~chnl_r4_p1]=0
            label_l4[label_l4 == 9]=1
            label_l4[label_l4 == 10]=2
            label_l4[label_l4 == 12]=3
            label_l4[label_l4 == 16]=4
            label_l4[label_l4 == 17]=5

            label_l5 = label_parsing.copy()
            label_l5[~chnl_r4_p2]=0
            label_l5[label_l5 == 8]=1
            label_l5[label_l5 == 18]=2
            label_l5[label_l5 == 19]=3

            return input, label_parsing, label_r1, label_r2, label_r3, label_r4, label_l0, label_l1, label_l2, label_l3, label_l4, label_l5, meta
        else:

            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))

            label_r1 = label_parsing.copy()
            chnl_r1_t0 = np.array([0,255])
            chnl_r1_t1 = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
            chnl_r1_p0 = np.isin(label_r1, chnl_r1_t0)
            chnl_r1_p1 = np.isin(label_r1, chnl_r1_t1)
            label_r1[chnl_r1_p0] = 0
            label_r1[chnl_r1_p1] = 1

            label_l0 = label_parsing.copy()
            label_l0[~chnl_r1_p0] = 0
            label_l0[chnl_r1_p0] = 1

            label_r2 = label_parsing.copy()
            chnl_r2_t0 = np.array([0,255])
            chnl_r2_t1 = np.array([1,2,4,13])
            chnl_r2_t2 = np.array([3,5,6,7,11,14,15])
            chnl_r2_t3 = np.array([8,9,10,12,16,17,18,19])
            chnl_r2_p0 = np.isin(label_r2, chnl_r2_t0)
            chnl_r2_p1 = np.isin(label_r2, chnl_r2_t1)
            chnl_r2_p2 = np.isin(label_r2, chnl_r2_t2)
            chnl_r2_p3 = np.isin(label_r2, chnl_r2_t3)
            label_r2[chnl_r2_p0] = 0
            label_r2[chnl_r2_p1] = 1
            label_r2[chnl_r2_p2] = 2
            label_r2[chnl_r2_p3] = 3


            label_l1 = label_parsing.copy()
            label_l1[~chnl_r2_p1]=0
            label_l1[label_l1==1]=1
            label_l1[label_l1==2]=2
            label_l1[label_l1==4]=3
            label_l1[label_l1==13]=4

            label_r3 = label_parsing.copy()
            chnl_r3_t0 = np.array([0,255,1,2,4,13,8,9,10,12,16,17,18,19])
            chnl_r3_t1 = np.array([5,6,7,11])
            chnl_r3_t2 = np.array([3,14,15])
            chnl_r3_p0 = np.isin(label_r3, chnl_r3_t0)
            chnl_r3_p1 = np.isin(label_r3, chnl_r3_t1)
            chnl_r3_p2 = np.isin(label_r3, chnl_r3_t2)
            label_r3[chnl_r3_p0] = 0
            label_r3[chnl_r3_p1] = 1
            label_r3[chnl_r3_p2] = 2

            label_l2 = label_parsing.copy()
            label_l2[~chnl_r3_p1]=0
            label_l2[label_l2 == 5]=1
            label_l2[label_l2 == 6]=2
            label_l2[label_l2 == 7]=3
            label_l2[label_l2 == 11]=4

            label_l3 = label_parsing.copy()
            label_l3[~chnl_r3_p2]=0
            label_l3[label_l3 == 3]=1
            label_l3[label_l3 == 14]=2
            label_l3[label_l3 == 15]=3

            label_r4 = label_parsing.copy()
            chnl_r4_t0 = np.array([0,255,1,2,4,13,3,5,6,7,11,14,15])
            chnl_r4_t1 = np.array([9,10,12,16,17])
            chnl_r4_t2 = np.array([8,18,19])
            chnl_r4_p0 = np.isin(label_r4, chnl_r4_t0)
            chnl_r4_p1 = np.isin(label_r4, chnl_r4_t1)
            chnl_r4_p2 = np.isin(label_r4, chnl_r4_t2)
            label_r4[chnl_r4_p0] = 0
            label_r4[chnl_r4_p1] = 1
            label_r4[chnl_r4_p2] = 2

            label_l4 = label_parsing.copy()
            label_l4[~chnl_r4_p1]=0
            label_l4[label_l4 == 9]=1
            label_l4[label_l4 == 10]=2
            label_l4[label_l4 == 12]=3
            label_l4[label_l4 == 16]=4
            label_l4[label_l4 == 17]=5

            label_l5 = label_parsing.copy()
            label_l5[~chnl_r4_p2]=0
            label_l5[label_l5 == 8]=1
            label_l5[label_l5 == 18]=2
            label_l5[label_l5 == 19]=3

            return input, label_parsing, label_r1, label_r2, label_r3, label_r4, label_l0, label_l1, label_l2, label_l3, label_l4, label_l5, meta
