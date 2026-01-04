import sys
sys.path.append('..')
import os
import glob
import numpy as np
import time
import torch
import ipdb
from .eval_metric import calc_mAPJ


def non_maximum_suppression(heatmap):
    max_heatmap = torch.nn.functional.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    mask = (heatmap == max_heatmap)
    weight = torch.ones_like(mask) * 0.6
    weight[mask] = 1.0
    heatmap = weight * heatmap
    return heatmap


def calc_junction(jmap, joff, thresh=1e-2, top_K=1000):
    jmap = torch.from_numpy(jmap)
    joff = torch.from_numpy(joff)

    jmap = non_maximum_suppression(jmap)

    h, w = jmap.shape[-2], jmap.shape[-1]
    score = jmap.flatten()
    joff = joff.reshape(2, -1).t()

    num = min(int((score >= thresh).sum().item()), top_K)
    indices = torch.argsort(score, descending=True)[:num]
    score = score[indices]
    y, x = indices // w, indices % w
    junc = torch.cat((x[:, None], y[:, None]), dim=1) + joff[indices] + 0.5

    junc[:, 0] = junc[:, 0].clamp(min=0, max=w - 1e-4)
    junc[:, 1] = junc[:, 1].clamp(min=0, max=h - 1e-4)

    junc = junc.numpy()
    score = score.numpy()

    return junc, score


def eval_mAPJ(groudtruth, prediction):

    junc_gts, junc_preds, junc_scores, im_ids = [], [], [], []
    for i, (gt, pred) in enumerate(zip(groudtruth, prediction)):
        # 得到 gt junc
        junc_gt = torch.cat([gt[:, 0, :], gt[:, 1, :]], dim=0)
        if junc_gt.is_cuda:
            junc_gt = junc_gt.detach().cpu()
        junc_gt_numpy = np.array(junc_gt)
        junc_gts.append(junc_gt_numpy)

        #  pred  junc
        jmap = pred['jmap']
        joff = pred['joff']
        junc_pred, junc_score = calc_junction(jmap, joff)
        
        junc_pred_numpy = np.array(junc_pred)
        junc_preds.append(junc_pred_numpy)

        junc_score_numpy = np.array(junc_score)
        junc_scores.append(junc_score_numpy)
        
        im_ids.append(np.array([i] * junc_pred.shape[0], dtype=np.int32))

    junc_preds = np.concatenate(junc_preds)
    junc_scores = np.concatenate(junc_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-junc_scores)
    junc_preds = junc_preds[indices]
    im_ids = im_ids[indices]


    mAPJ, P, R = calc_mAPJ(junc_gts, junc_preds, im_ids, [0.5, 1.0, 2.0])
    return mAPJ, P, R


if __name__ == '__main__':
    print('1')