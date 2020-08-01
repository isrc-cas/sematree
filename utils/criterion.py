import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


from .dice_loss import DiceLoss, BinaryDiceLoss, make_one_hot
from .focalloss import FocalLoss

LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']
ROUTER = [
            [0,1],         # Router 1
            [0,1,2,3],     # Router 2
            [0,1,2],       # Router 3
            [0,1,2]        # Router 4
        ]

LEAF=[
        [0,1],         # Leaf 1
        [0,1,2,3,4],   # Leaf 2
        [0,1,2,3,4],   # Leaf 3
        [0,1,2,3],     # Leaf 4
        [0,1,2,3,4,5], # Leaf 5
        [0,1,2,3]      # Leaf 6
    ]

WEIGHT_PARSE = 3.0
WEIGHT_ROUTER = 1.0
WEIGHT_LEAF = 1.5



class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        #self.criterion = FocalLoss(gamma=0)
        self.dice_loss = DiceLoss()

    def parsing_loss(self, preds, targets):
        h, w = targets[0][0].size(1), targets[0][0].size(2)

        parse_weights = [torch.sum(targets[0][0] == i, dtype=torch.float) for i in range(len(LABELS))]
        parse_weights = [1- parse_weights[i] / sum(parse_weights) if parse_weights[i] != 0 else 0
                for i in range(len(parse_weights))]
        parse_weights = torch.tensor(parse_weights)
        parse_weights = parse_weights.cuda()


        # total loss
        loss = 0

        # loss for parsing
        preds_parsing = preds[0]
        targets_parsing = targets[0]
        if isinstance(preds_parsing, list):
            for index in range(0, len(preds_parsing)):
                pred_parsing = preds_parsing[index]
                target_parsing = targets_parsing[index]

                pred_parsing = F.interpolate(input=pred_parsing, size=(h, w),
                                           mode='bilinear', align_corners=True)
                pred_parsing = torch.log(pred_parsing)
                loss_parsing = F.nll_loss(pred_parsing, target_parsing, weight=parse_weights,
                        ignore_index=self.ignore_index)

                loss += loss_parsing * WEIGHT_PARSE
                #print('parsing loss is {}'.format(loss_parsing))
        else:
            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, targets[0])

        # loss for router
        preds_router = preds[1]
        targets_router = targets[1]
        if isinstance(preds_router, list):
            for index in range(0,len(preds_router)):
                pred_router = preds_router[index]
                target_router = targets_router[index]

                router_label = ROUTER[index]
                router_weights = [torch.sum(target_router == router_label[i], dtype=torch.float)
                        for i in range(len(router_label))]
                router_weights = [1- router_weights[i] / sum(router_weights) if router_weights[i] != 0 else 0
                        for i in range(len(router_weights))]
                router_weights = torch.tensor(router_weights)
                router_weights = router_weights.cuda()

                pred_router = F.interpolate(input=pred_router, size=(h, w),
                                           mode='bilinear', align_corners=True)
                pred_router = torch.log(pred_router)
                loss_router = F.nll_loss(pred_router, target_router, weight=router_weights, ignore_index=self.ignore_index)
                loss += loss_router * WEIGHT_ROUTER
                #print('{} router loss is {}'.format(index, loss_router))
        else:
            loss += F.cross_entropy(preds_router, targets_router, ignore_index=self.ignore_index)

        # loss for leaf
        preds_leaf = preds[2]
        targets_leaf = targets[2]
        if isinstance(preds_leaf, list):
            for index in range(0, len(preds_leaf)):
                pred_leaf = preds_leaf[index]
                target_leaf = targets_leaf[index]

                leaf_label = LEAF[index]
                leaf_weights = [torch.sum(target_leaf == leaf_label[i], dtype=torch.float) for i in range(len(leaf_label))]
                leaf_weights = [1- leaf_weights[i] / sum(leaf_weights) if leaf_weights[i] != 0 else 0
                        for i in range(len(leaf_weights))]
                leaf_weights = torch.tensor(leaf_weights)
                leaf_weights = leaf_weights.cuda()

                pred_leaf = F.interpolate(input=pred_leaf, size=(h, w),
                                           mode='bilinear', align_corners=True)
                pred_leaf = torch.log(pred_leaf)
                loss_leaf = F.nll_loss(pred_leaf, target_leaf, weight=leaf_weights, ignore_index=self.ignore_index)

                loss += loss_leaf * WEIGHT_LEAF
                #print('{} leaf loss is {}'.format(index, loss_leaf))
        else:
            loss += F.cross_entropy(preds_leaf, targets_leaf,
                                    ignore_index=self.ignore_index)


        return loss

    def forward(self, preds, targets):

        loss = self.parsing_loss(preds, targets)
        return loss



if __name__ == '__main__':
    pass
