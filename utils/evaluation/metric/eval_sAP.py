import sys
sys.path.append('..')
import os
import glob
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
from .eval_metric import calc_msAP, calc_sAP, plot_pr_curve
import util.bezier as bez
import ipdb

def eval_sAP(groudtruth, prediction, save_path=None, order=None):
    line_gts, line_preds, line_scores, im_ids = [], [], [], []
    for i, (gt, pred) in enumerate(zip(groudtruth, prediction)):
        if gt.is_cuda:
            gt = gt.detach().cpu()
        line_gt_numpy = np.array(gt)
        line_gts.append(line_gt_numpy)

        line_pred = pred['line_pred']
        line_score = pred['line_score']

        line_preds.append(line_pred)
        line_scores.append(line_score)
        im_ids.append(np.array([i] * line_pred.shape[0], dtype=np.int32))

    line_preds = np.concatenate(line_preds)
    line_scores = np.concatenate(line_scores)
    im_ids = np.concatenate(im_ids)
    indices = np.argsort(-line_scores)
    line_scores = line_scores[indices]
    line_preds = line_preds[indices]
    im_ids = im_ids[indices]

    n_pts = line_gts[0].shape[1]
    line_preds = np.asarray(bez.interp_line(line_preds, num=n_pts))

    msAP, P, R, sAP = calc_msAP(line_gts, line_preds, im_ids, [5.0, 10.0, 15.0])

    if save_path is not None:
        sAP10, _, _, rcs, prs = calc_sAP(line_gts, line_preds, im_ids, 10.0)
        figure = plot_pr_curve(rcs, prs, title='sAP${^{10}}$', legend=['ULSD'],)
        figure.savefig(os.path.join(save_path, f'sAP10_{cfg.version}.pdf'), format='pdf', bbox_inches='tight')
        sio.savemat(os.path.join(save_path, f'sAP10_{cfg.version}.mat'), {'rcs': rcs, 'prs': prs, 'AP': sAP10})
        plt.show()

    return msAP, P, R, sAP


