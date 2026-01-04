from typing import Any, List, Optional, Dict
from warnings import warn

import numpy as np
import os
# from utils.evaluation.prophesee.evaluation import evaluate_list
from utils.evaluation.metric.eval_mAPJ import eval_mAPJ
from utils.evaluation.metric.eval_sAP import eval_sAP

from utils.helpers import *
from termcolor import colored
from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
import torch
from matplotlib import pyplot as plt
import wandb, ipdb, h5py

def save_dict_list_to_h5(data_list, filename):
    """保存字典列表到HDF5文件"""
    with h5py.File(filename, 'w') as f:
        # 添加整体属性说明
        f.attrs['num_items'] = len(data_list)
        f.attrs['data_structure'] = 'list of dicts with ev_frame, pred_lines, sequence_name, item'
        
        for i, data_dict in enumerate(data_list):
            group = f.create_group(f'item_{i:06d}')  # 用6位数字编号
            
            # 存储numpy数组
            group.create_dataset('ev_frame', data=data_dict['ev_frame'])
            group.create_dataset('pred_lines', data=data_dict['pred_lines'])
            
            # 存储标量数据作为属性
            group.attrs['sequence_name'] = data_dict['sequence_name']
            group.attrs['item'] = data_dict['item']
            
            # 可选：保存数组的形状信息作为属性
            group.attrs['ev_frame_shape'] = data_dict['ev_frame'].shape
            group.attrs['pred_lines_shape'] = data_dict['pred_lines'].shape


def visualize_and_save_lines(gt_lines, pred_lines, img_name=None):
    """
    可视化 ground-truth 和预测的线段，并保存为图片
    :param label: torch.Tensor, 形状为 [M, 2, 2]
    :param line_pred: torch.Tensor, 形状为 [M, 2, 2]
    :param save_path: 保存路径，默认 "prediction.png"
    """
    if not hasattr(visualize_and_save_lines, "epoch_counter"):
        visualize_and_save_lines.epoch_counter = 0

    img_name = f'gt_vs_pred_epoch_{visualize_and_save_lines.epoch_counter}' if img_name is None else img_name
    visualize_and_save_lines.epoch_counter += 1

    label = gt_lines.detach().cpu().numpy()
    line_pred = pred_lines.detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 如果是图像坐标系

    # 画 label（绿色）
    for line in label:
        x = [line[0, 0], line[1, 0]]
        y = [line[0, 1], line[1, 1]]
        plt.plot(x, y, color='green', linewidth=2, label='Label')

    # 画预测线（红色虚线）
    for line in line_pred:
        x = [line[0, 0], line[1, 0]]
        y = [line[0, 1], line[1, 1]]
        plt.plot(x, y, color='red', linewidth=1.5, linestyle='--', label='Prediction')

    # 避免图例重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()

    # 保存目录
    save_dir = './vis_results'
    os.makedirs(save_dir, exist_ok=True)

    # 图像文件名
    save_path = os.path.join(save_dir, f"{img_name}.png")

    # 保存图像
    plt.savefig(save_path)
    plt.close()



def check_metric_cal(labels, predictions):
    #可视化第一张图片的标签和预测结果
    gt_for_vis = labels[0] # N, 2, 2

    pred_for_vis = predictions[0]['line_pred']
    pred_score = predictions[0]['line_score'] # [N,]

    mask = pred_score > 0.8
    pred_for_vis = pred_for_vis[mask]

    if len(pred_for_vis) > 0:
        visualize_and_save_lines(gt_lines=gt_for_vis, pred_lines=pred_for_vis, img_name=None)

            

class PropheseeEvaluator:
    LABELS = 'lables'
    PREDICTIONS = 'predictions'
    PREDICTIONS_WITH_EVENT = 'predictions_with_event'

    def __init__(self, dataset: str, downsample_by_2: bool):
        super().__init__()
       
        self.dataset = dataset
        self.downsample_by_2 = downsample_by_2

        self._buffer = None
        self._buffer_empty = True
        self._reset_buffer()
        self.ignored = True

    def _reset_buffer(self):
        self._buffer_empty = True
        self._buffer = {
            self.LABELS: list(),
            self.PREDICTIONS: list(),
            self.PREDICTIONS_WITH_EVENT: list(),
        }

    def set_ignored_to_False(self):
        self.ignored = False

    def _add_to_buffer(self, key: str, value: List[np.ndarray]):
        assert isinstance(value, list)
      
        self._buffer_empty = False
        assert self._buffer is not None
        self._buffer[key].extend(value)

    def _get_from_buffer(self, key: str) -> List[np.ndarray]:
        assert not self._buffer_empty
        assert self._buffer is not None
        return self._buffer[key]

    def add_predictions(self, predictions: List[dict]):

        self._add_to_buffer(self.PREDICTIONS, predictions)


    def add_prediction_with_event(self, predictions_with_event: List[dict]):

        self._add_to_buffer(self.PREDICTIONS_WITH_EVENT, predictions_with_event)

    def add_labels(self, labels: List[np.ndarray]): 
        label_list = []
        for label in labels:
            if isinstance(label, ObjectLabels):
                label = label.object_labels
            if label is None:
                continue  
            label_list.append(label)
        self._add_to_buffer(self.LABELS, label_list)

    def reset_buffer(self) -> None:
        # E.g. call in on_validation_epoch_start
        self._reset_buffer()

    def has_data(self):
        return not self._buffer_empty

    def evaluate_buffer(self, img_height: int, img_width: int) -> Optional[Dict[str, Any]]:
        # e.g call in on_validation_epoch_end
        if self._buffer_empty:
            warn("Attempt to use prophesee evaluation buffer, but it is empty", UserWarning, stacklevel=2)
            return

        groudtruth = self._get_from_buffer(self.LABELS)
        predictions = self._get_from_buffer(self.PREDICTIONS)

        result =  self._get_from_buffer(self.PREDICTIONS_WITH_EVENT)

        print(colored(len(result), 'green'))
        save_dict_list_to_h5(result, 'prediction_ours.h5')

        ipdb.set_trace()


        assert len(groudtruth) == len(predictions)
        print(colored('groudtruth length:{}'.format(len(groudtruth)), 'yellow'))

        # 检查预测结果和gt是否对应
        check_metric_cal(groudtruth, predictions)
    
        mAPJ, P, R = eval_mAPJ(groudtruth, predictions)
        msAP, P, R, sAP = eval_sAP(groudtruth, predictions)
    
        metric = {
            'sAP5': sAP[0],
            'sAP10':sAP[1],
            'sAP15':sAP[2],
            'msAP': msAP,
            'mAPJ': mAPJ,
        }
        return metric
