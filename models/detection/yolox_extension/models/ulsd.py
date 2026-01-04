import torch.nn as nn
import torch.nn.functional as F
import ipdb

import numpy as np
import matplotlib.pyplot as plt


def weighted_l1_loss(logits, target, mask=None):
    loss = F.l1_loss(logits, target, reduction='none')
    if mask is not None:
        w = mask.mean(3, True).mean(2, True)
        w[w == 0] = 1
        loss = loss * (mask / w)
    return loss.mean()


def weighted_smooth_l1_loss(logits, target, mask=None):
    loss = F.smooth_l1_loss(logits, target, reduction='none')
    if mask is not None:
        loss = loss * mask
    return loss.mean()


def lpn_loss_func(outputs, labels):
    """
    LPN Loss

    """
    n_eoff = outputs['eoff'].shape[1] // 2
    lmap_loss = F.binary_cross_entropy(outputs['lmap'], labels['lmap'])
    jmap_loss = F.binary_cross_entropy(outputs['jmap'], labels['jmap'])
    joff_loss = weighted_l1_loss(outputs['joff'], labels['joff'], labels['jmap'])
    cmap_loss = F.binary_cross_entropy(outputs['cmap'], labels['cmap'])
    coff_loss = weighted_l1_loss(outputs['coff'], labels['coff'], labels['cmap'])
    eoff_loss = n_eoff * weighted_smooth_l1_loss(outputs['eoff'], labels['eoff'], labels['cmap'])

    return lmap_loss, jmap_loss, joff_loss, cmap_loss, coff_loss, eoff_loss


def loi_loss_func(outputs, labels):
    """
    LoI Head Loss

    """
    pos_loss, neg_loss = 0.0, 0.0
    batch_size = len(outputs)
    for output, label in zip(outputs, labels):
        loss = F.binary_cross_entropy(output, label, reduction='none')
        pos_loss += loss[label == 1].mean() / batch_size
        neg_loss += loss[label == 0].mean() / batch_size

    return pos_loss, neg_loss
