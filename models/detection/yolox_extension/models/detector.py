from typing import Dict, Optional, Tuple

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_yolox_head

from .line_proposal_network import LineProposalNetwork
from .ulsd import lpn_loss_func, loi_loss_func
from utils.timers import TimerDummy as CudaTimer
import util.camera as cam
import util.bezier as bez

from .loi_head import LoIHead

from data.utils.types import BackboneFeatures, LstmStates
import os
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt 
import cv2


def save_npz(lines, centers):
    order = 1 
    heatmap_size = [128, 128]
    heatmap_size = tuple(heatmap_size)
    n_pts = order + 1
    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]
    lines[:, :, 0] = np.clip(lines[:, :, 0] , 0, 128 - 1e-4)
    lines[:, :, 1] = np.clip(lines[:, :, 1] , 0, 128 - 1e-4)
    centers[:, 0] = np.clip(centers[:, 0] , 0, 128 - 1e-4)
    centers[:, 1] = np.clip(centers[:, 1] , 0, 128 - 1e-4)

    jmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    joff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    cmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)
    coff = np.zeros((2,) + heatmap_size[::-1], dtype=np.float32)
    eoff = np.zeros(((n_pts // 2) * 2, 2,) + heatmap_size[::-1], dtype=np.float32)
    lmap = np.zeros((1,) + heatmap_size[::-1], dtype=np.float32)

    juncs = np.concatenate((lines[:, 0], lines[:, -1]))
    juncs = np.round(juncs, 3)
    juncs = np.unique(juncs, axis=0)
    lpos = lines.copy()
    lneg = []

    if n_pts % 2 == 1:
        lines = np.delete(lines, n_pts // 2, axis=1)

    def to_int(x):
        return tuple(map(int, x))

    for c, pts in zip(centers, lines):
        v0, v1 = pts[0], pts[-1]

        cint = to_int(c)
        vint0 = to_int(v0)
        vint1 = to_int(v1)
        jmap[0, vint0[1], vint0[0]] = 1
        jmap[0, vint1[1], vint1[0]] = 1
        joff[:, vint0[1], vint0[0]] = v0 - vint0 - 0.5
        joff[:, vint1[1], vint1[0]] = v1 - vint1 - 0.5
        cmap[0, cint[1], cint[0]] = 1
        coff[:, cint[1], cint[0]] = c - cint - 0.5
        eoff[:, :, cint[1], cint[0]] = pts - c

    eoff = eoff.reshape((-1,) + heatmap_size[::-1])
    lmap[0] = bez.insert_line(lmap[0], lpos, color=255) / 255.0

    label_dict = {
    'junc': juncs,
    'lpos': lpos,
    'lneg': lneg,
    'jmap': jmap,
    'joff': joff,
    'cmap': cmap,
    'coff': coff,
    'eoff': eoff,
    'lmap': lmap
}
    return label_dict


def transform(label):
    jmap = torch.from_numpy(label['jmap']).float()
    joff = torch.from_numpy(label['joff']).float()
    cmap = torch.from_numpy(label['cmap']).float()
    coff = torch.from_numpy(label['coff']).float()
    eoff = torch.from_numpy(label['eoff']).float()
    lmap = torch.from_numpy(label['lmap']).float()
    line = torch.from_numpy(label['lpos']).float()
    lpos = np.random.permutation(label['lpos'])[: 300]
    lneg = np.random.permutation(label['lneg'])[: 300]
    if len(lneg) == 0:
        lneg = np.zeros((0, lpos.shape[1], 2))
    npos, nneg = len(lpos), len(lneg)
    lpre = np.concatenate((lpos, lneg))
    for i in range(len(lpre)):
        if random.random() > 0.5:
            lpre[i] = lpre[i, ::-1]
    lpre = torch.from_numpy(lpre).float()
    lpre_label = torch.cat([torch.ones(npos), torch.zeros(nneg)]).float()

    map = {'jmap': jmap, 'joff': joff, 'cmap': cmap, 'coff': coff, 'eoff': eoff, 'lmap': lmap}
    meta = {'line': line, 'lpre': lpre, 'lpre_label': lpre_label}

    return map, meta


def bezier_label_to_tensor(lines, centers):
    order = 1
    heatmap_size = (128, 128)
    n_pts = order + 1

    lines_mask = lines[:, 0, 1] > lines[:, -1, 1]
    lines[lines_mask] = lines[lines_mask, ::-1]

    np.clip(lines[:, :, 0], 0, heatmap_size[1] - 1e-4, out=lines[:, :, 0])
    np.clip(lines[:, :, 1], 0, heatmap_size[0] - 1e-4, out=lines[:, :, 1])
    np.clip(centers[:, 0], 0, heatmap_size[1] - 1e-4, out=centers[:, 0])
    np.clip(centers[:, 1], 0, heatmap_size[0] - 1e-4, out=centers[:, 1])

    jmap_np = np.zeros((1, *heatmap_size), dtype=np.float32)
    joff_np = np.zeros((2, *heatmap_size), dtype=np.float32)
    cmap_np = np.zeros((1, *heatmap_size), dtype=np.float32)
    coff_np = np.zeros((2, *heatmap_size), dtype=np.float32)
    eoff_np = np.zeros(((n_pts // 2) * 2, 2, *heatmap_size), dtype=np.float32)
    lmap_np = np.zeros((1, *heatmap_size), dtype=np.float32)

    juncs = np.unique(np.round(np.concatenate((lines[:, 0], lines[:, -1])), 3), axis=0)
    lpos = lines.copy()
    lneg = []

    if n_pts % 2 == 1:
        lines = np.delete(lines, n_pts // 2, axis=1)

    def to_int(pt):
        return tuple(map(int, pt))

    for c, pts in zip(centers, lines):
        v0, v1 = pts[0], pts[-1]
        cint = to_int(c)
        vint0 = to_int(v0)
        vint1 = to_int(v1)

        jmap_np[0, vint0[1], vint0[0]] = 1
        jmap_np[0, vint1[1], vint1[0]] = 1
        joff_np[:, vint0[1], vint0[0]] = v0 - vint0 - 0.5
        joff_np[:, vint1[1], vint1[0]] = v1 - vint1 - 0.5
        cmap_np[0, cint[1], cint[0]] = 1
        coff_np[:, cint[1], cint[0]] = c - cint - 0.5
        eoff_np[:, :, cint[1], cint[0]] = pts - c

    eoff_np = eoff_np.reshape((-1, *heatmap_size))
    lmap_np[0] = bez.insert_line(lmap_np[0], lpos, color=255) / 255.0

    # 转 Tensor
    map_data = {
        'jmap': torch.from_numpy(jmap_np).float(),
        'joff': torch.from_numpy(joff_np).float(),
        'cmap': torch.from_numpy(cmap_np).float(),
        'coff': torch.from_numpy(coff_np).float(),
        'eoff': torch.from_numpy(eoff_np).float(),
        'lmap': torch.from_numpy(lmap_np).float()
    }

    line_tensor = torch.from_numpy(lpos).float()
    lpos_sampled = np.random.permutation(lpos)[:300]
    lneg_sampled = np.random.permutation(lneg)[:300] if len(lneg) > 0 else np.zeros((0, lpos.shape[1], 2))

    npos, nneg = len(lpos_sampled), len(lneg_sampled)
    lpre = np.concatenate((lpos_sampled, lneg_sampled))
    for i in range(len(lpre)):
        if random.random() > 0.5:
            lpre[i] = lpre[i, ::-1]

    meta_data = {
        'line': line_tensor,
        'lpre': torch.from_numpy(lpre).float(),
        'lpre_label': torch.cat([torch.ones(npos), torch.zeros(nneg)]).float()
    }

    return map_data, meta_data


def line2meta(label):
    label = label.detach().cpu()
    lines = label[:, 1:].view(-1, 2, 2)
    lines = np.array(lines)
    camera = cam.Pinhole()
    pts_list = camera.interp_line(lines)
    lines = bez.fit_line(pts_list, order=2)[0]
    centers = lines[:, 1]
    lines = bez.fit_line(pts_list, order=1)[0]

    # label_dict = save_npz(lines.copy(), centers)
    # map, meta = transform(label_dict)
    map, meta = bezier_label_to_tensor(lines.copy(), centers)

    return map, meta


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)
    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])
    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        w = mask.mean(3, keepdim=True).mean(2, keepdim=True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean()

def get_junctions(jloc, joff, topk=300, th=0.0):
    jloc = jloc.reshape(-1) #torch.Size([16384])
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // 128).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % 128).float() + torch.gather(joff[0], 0, index) + 0.5
    junctions = torch.stack((x, y)).t()

    return junctions[scores > th]

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


def supervise_featuremap(pred_feature_map, gt_blurred_map):
    pred_feature_map = pred_feature_map.squeeze()                          

    if not isinstance(gt_blurred_map, torch.Tensor):
        gt_blurred_map = torch.from_numpy(gt_blurred_map)
    gt_blurred_map = gt_blurred_map.to(pred_feature_map.device).float()

    gt_blurred_map = torch.sigmoid(gt_blurred_map)
    loss = torch.mean(gt_blurred_map * (pred_feature_map - gt_blurred_map) ** 2)

    return loss


def ann2mask(cfg, ann):
    ann = ann.detach().cpu().numpy()

    x1_y1 = ann[:, 1:3].astype(int)  # [N, 2]
    x2_y2 = ann[:, 3:5].astype(int)  # [N, 2]

    mask = np.full((128, 128), 255, dtype=np.uint8)

    for p1, p2 in zip(x1_y1, x2_y2):
        cv2.line(mask, tuple(p1), tuple(p2), color=0, thickness=3)

    return mask
    

class Conv_BN_Sigm(nn.Module):
    def __init__(self, input_channels, output_channles,
                 kernel_size=3, stride=1, padding=1,
                 bias=True):
        super().__init__()
        self.CBR = nn.Sequential(
            nn.Conv2d(input_channels, output_channles,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias),
            nn.BatchNorm2d(output_channles),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.CBR(x) 
    

class YoloXDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        
        output_channels_list = [1, 1, 2, 1, 2, ] + [4 * ((model_cfg.lpn.order + 1) // 2)]
        self.mdl_config = model_cfg

        self.encoder_cfg = model_cfg.ENCODER

           
        self.n_dyn_junc = self.mdl_config.backbone.n_dyn_junc
        self.junc_thresh = self.mdl_config.backbone.junc_thresh
        self.n_dyn_posl = self.mdl_config.backbone.n_dyn_posl
        self.n_dyn_negl = self.mdl_config.backbone.n_dyn_negl
        self.n_pts0 = self.mdl_config.backbone.n_pts0
        self.n_pts1 = self.mdl_config.backbone.n_pts1
        
        
        self.dim_feat = self.mdl_config.backbone.num_feats
        self.dim_loi = self.mdl_config.backbone.dim_loi
        self.dim_fc = self.mdl_config.backbone.dim_fc
        self.scale =  self.mdl_config.backbone.dis_th

        backbone_cfg = model_cfg.backbone

        self.with_mask = model_cfg.with_mask
        lpn_cfg = model_cfg.lpn

        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :])
        self.backbone = build_recurrent_backbone(backbone_cfg)
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.conv = nn.Conv2d(self.dim_feat, self.dim_loi, 1)
        self.pool1d = nn.MaxPool1d(self.n_pts0 // self.n_pts1, stride=self.n_pts0 // self.n_pts1)
        self.fc = nn.Sequential(
            nn.Linear(self.dim_loi * self.n_pts1, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, self.dim_fc),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim_fc, 1)
        )

        self.use_residual = True
        self.lpn = LineProposalNetwork(
            input_channels=lpn_cfg.num_feats,
            output_channels_list=output_channels_list,
            order=lpn_cfg.order,
            junc_score_thresh=lpn_cfg.junc_score_thresh,
            line_score_thresh=lpn_cfg.line_score_thresh,
            junc_max_num=lpn_cfg.junc_max_num,
            line_max_num=lpn_cfg.line_max_num,
            num_pos_proposals=lpn_cfg.num_pos_proposals,
            num_neg_proposals=lpn_cfg.num_neg_proposals,
            nms_size=lpn_cfg.nms_size
        )

        self.head = LoIHead(num_feats=lpn_cfg.num_feats, order=lpn_cfg.order, n_pts=lpn_cfg.n_pts)

        self.reducer = Conv_BN_Sigm(input_channels=256, output_channles=1)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:\

        with CudaTimer(device=x.device, timer_name="Backbone"):
            features, backbone_features, states = self.backbone(x, previous_states)
        return features, backbone_features, states

    # Forward train
    def forward(self, x, previous_states, targets):
        device = x.device
        metas = { 
                'lpre': [],
                'lpre_label': [],
                'line': []
            }

        map_labels = {
            'jmap':[],
            'joff':[],
            'cmap':[],
            'coff':[],
            'eoff':[],
            'lmap':[],
        }

        for i in targets:
            label_single = i.object_labels   # [N, 5]
            map_single, meta_single = line2meta(label_single)
            map_single = {name: map_single[name].to(device) for name in map_single.keys()}
            for key in metas:
                metas[key].append(meta_single[key].to(device))

            assert map_single is not None
            for key in map_labels:
                map_labels[key].append(map_single[key].to(device))

        map_labels = {k: torch.stack(v, dim=0) for k, v in map_labels.items()}
        
        features, backbone_features, states = self.forward_backbone(x, previous_states)

        if self.with_mask:
            for i in range(len(x)):
                label_single = targets[i].object_labels

                line_mask = ann2mask(self.encoder_cfg, label_single)
                line_mask = torch.tensor(line_mask).to(device)

                # hourglass 后面的特征图
                loss1 = supervise_featuremap(self.reducer(backbone_features[1][i].unsqueeze(0)), line_mask)
                loss2 = supervise_featuremap(self.reducer(backbone_features[2][i].unsqueeze(0)), line_mask)

                backbone_feature_loss = loss1 + loss2

        map_preds, loi_preds, loi_scores = self.lpn(features)
        loi_preds, loi_labels = self.lpn.sample_lines(loi_preds, loi_scores, metas)

        loi_scores = self.head(features, loi_preds)
  
        lmap_loss, jmap_loss, joff_loss, eoff, coff_loss, eoff_loss = lpn_loss_func(map_preds, map_labels)
        pos_loss, neg_loss = loi_loss_func(loi_scores, loi_labels)

        if self.with_mask:
            losses = [lmap_loss, jmap_loss, joff_loss, eoff, coff_loss, eoff_loss, pos_loss, neg_loss, backbone_feature_loss]
            weights = [0.5, 8.0, 0.25, 8.0, 0.25, 1.0, 1.0, 1.0, 0.1]

            loss_names = ['lmap', 'jmap', 'joff', 'eoff', 'coff', 'eoff', 'pos', 'neg', 'backbone_feature_loss']

        else:
            losses = [lmap_loss, jmap_loss, joff_loss, eoff, coff_loss, eoff_loss, pos_loss, neg_loss]
            weights = [0.5, 8.0, 0.25, 8.0, 0.25, 1.0, 1.0, 1.0]

            loss_names = ['lmap', 'jmap', 'joff', 'eoff', 'coff', 'eoff', 'pos', 'neg']

        loss = sum([weight * loss for weight, loss in zip(weights, losses)])
        loss_dict = {name: weight * loss for name, weight, loss in zip(loss_names, weights, losses)}
        
        return loss, loss_dict, states
 
    # Forward test
    def forward_test(self, x, previous_states):
        features, backbone_features, states = self.forward_backbone(x, previous_states)
        maps, loi_preds, loi_scores = self.lpn(features)
        loi_scores = self.head(features, loi_preds)

        jmaps = maps['jmap']
        joffs = maps['joff']
        line_preds = loi_preds
        line_scores = loi_scores
        jmaps = jmaps.detach().cpu()
        joffs = joffs.detach().cpu()
        line_preds = [line_pred.detach().cpu() for line_pred in line_preds]
        line_scores = [line_score.detach().cpu() for line_score in line_scores]

        return jmaps, joffs, line_preds, line_scores, states
    
    def proposal_lines(self, md_maps, dis_maps, residual_maps, scale):
        """

        :param md_maps: 3 x H x W, the range should be (0,1) for every element
        :param dis_maps: 1 x H x W
        :return:
        """
        device = md_maps.device
        sign_pad = torch.tensor([-1, 0, 1], device=device, dtype=torch.float32).reshape(3, 1, 1)

        if residual_maps is None:
            dis_maps_new = dis_maps.repeat((1, 1, 1))
        else:
            dis_maps_new = dis_maps.repeat((3, 1, 1)) + sign_pad * residual_maps.repeat((3, 1, 1))

        height, width = md_maps.size(1), md_maps.size(2)
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x)
        md_ = (md_maps[0] - 0.5) * np.pi * 2
        st_ = md_maps[1] * np.pi / 2
        ed_ = -md_maps[2] * np.pi / 2

        cs_md = torch.cos(md_)
        ss_md = torch.sin(md_)

        cs_st = torch.cos(st_).clamp(min=1e-3)
        ss_st = torch.sin(st_).clamp(min=1e-3)
        cs_ed = torch.cos(ed_).clamp(min=1e-3)
        ss_ed = torch.sin(ed_).clamp(max=-1e-3)

        y_st = ss_st / cs_st
        y_ed = ss_ed / cs_ed

        x_st_rotated = (cs_md - ss_md * y_st)[None] * dis_maps_new * scale
        y_st_rotated = (ss_md + cs_md * y_st)[None] * dis_maps_new * scale
        x_ed_rotated = (cs_md - ss_md * y_ed)[None] * dis_maps_new * scale
        y_ed_rotated = (ss_md + cs_md * y_ed)[None] * dis_maps_new * scale

        x_st_final = (x_st_rotated + x0[None]).clamp(min=0, max=width-1)  #3,128,128
        y_st_final = (y_st_rotated + y0[None]).clamp(min=0, max=height-1)  #3,128,128
        x_ed_final = (x_ed_rotated + x0[None]).clamp(min=0, max=width-1)
        y_ed_final = (y_ed_rotated + y0[None]).clamp(min=0, max=height-1)
        

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final)).permute((1, 2, 3, 0))

        return lines
    
    def pooling(self, loi_feature, lines):
        C, H, W = loi_feature.shape
        start_points, end_points = lines[:, :2], lines[:, 2:]

        sampled_points = start_points[:, :, None] * self.tspan + end_points[:, :, None] * (1 - self.tspan) - 0.5
        sampled_points = sampled_points.transpose(1, 2).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=W - 1)
        py0 = py.floor().clamp(min=0, max=H - 1)
        px1 = (px0 + 1).clamp(min=0, max=W - 1)
        py1 = (py0 + 1).clamp(min=0, max=H - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = ((loi_feature[:, py0l, px0l] * (py1 - py) * (px1 - px) +
                loi_feature[:, py1l, px0l] * (py - py0) * (px1 - px) +
                loi_feature[:, py0l, px1l] * (py1 - py) * (px - px0) +
                loi_feature[:, py1l, px1l] * (py - py0) * (px - px0)).view(self.dim_loi, -1, self.n_pts0)
                ).transpose(0, 1).contiguous()

        xp = self.pool1d(xp)
        xp = xp.view(-1, self.n_pts1 * self.dim_loi)

        logits = self.fc(xp).flatten() 

        return logits



