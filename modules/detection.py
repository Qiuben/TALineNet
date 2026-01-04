from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, time, ipdb, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode

from models.detection.yolox_extension.models.detector import YoloXDetector

from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches

from termcolor import colored


### 黑底 ###
def save_lines_on_ev_frame(lines: np.ndarray,
                           ev_pre_vis: np.ndarray,
                           sequence_name: str,
                           item: int,
                           dataset_name: str,
                           target_size_wh=(346, 260),
                           line_color="orange",
                           line_width=1.0,
                           point_color=None,
                           point_size=None,
                           point_zorder=5,
                           plot=False):
    """
    将 lines(128x128 坐标) 与 ev_pre_vis[item](512x512) 一起缩放到 (W,H)=(346,260)，
    用 matplotlib 将线段画到事件图上并保存。

    Params
    ------
    lines: (N,2,2) 线段端点 [[x1,y1],[x2,y2]]，原坐标范围 128x128
    ev_pre_vis: (T,512,512) 灰度事件帧
    sequence_name: 序列名
    args: 包含 "<dataname>" 键
    item: 第几帧（0..T-1）
    target_size_wh: (W,H) 目标分辨率，默认 (346,260)
    line_color: 线条颜色（字符串或RGB元组）
    line_width: 线条粗细
    point_color: 端点颜色；默认与 line_color 一致
    point_size: 端点大小；默认与线宽相关（2*line_width）
    plot: 是否 plt.show()
    """
    assert lines.ndim == 3 and lines.shape[1:] == (2, 2), "lines 需为 (N,2,2)"
    assert ev_pre_vis.ndim == 3 and ev_pre_vis.shape[1:] == (512, 512), "ev_pre_vis 需为 (T,512,512)"

    W_tgt, H_tgt = target_size_wh

    # 1) 缩放事件帧到目标大小 (H_tgt, W_tgt)
    pos = ev_pre_vis[:10].sum(axis=0).astype(np.float32)
    neg = ev_pre_vis[10:20].sum(axis=0).astype(np.float32)

    # 先转成uint8（需要先归一化或截断）
    pos_u8 = np.clip(pos, 0, 255).astype(np.uint8)
    neg_u8 = np.clip(neg, 0, 255).astype(np.uint8)

    # 对每个通道分别进行直方图均衡化
    pos_equalized = cv2.equalizeHist(pos_u8)
    neg_equalized = cv2.equalizeHist(neg_u8)


    frame = np.zeros((512, 512, 3), dtype=np.uint8)
    frame[..., 0] = pos_equalized   # R
    frame[..., 1] = 0        # G 全黑
    frame[..., 2] = neg_equalized

    frame_resized = cv2.resize(frame, (W_tgt, H_tgt), interpolation=cv2.INTER_AREA)

    # 2) 缩放 lines（原 128x128 -> 346x260）
    sx = W_tgt / 128.0
    sy = H_tgt / 128.0
    lines_scaled = lines.astype(np.float32).copy()
    lines_scaled[:, 0, 0] *= sx  # x1
    lines_scaled[:, 0, 1] *= sy  # y1
    lines_scaled[:, 1, 0] *= sx  # x2
    lines_scaled[:, 1, 1] *= sy  # y2

    # 3) 用 matplotlib 可视化（坐标/边距与你提供的 save_lines 一致）
    if point_color is None:
        point_color = line_color
    if point_size is None:
        point_size = max(0.5, line_width * 2.0)

    fig = plt.figure()
    # 与原函数一致：宽高比 = W/H；无边框，铺满
    fig.set_size_inches(W_tgt / H_tgt, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    # 坐标范围与方向（与原代码一致）
    plt.xlim([-0.5, W_tgt - 0.5])
    plt.ylim([H_tgt - 0.5, -0.5])

    # 单通道灰度显示
    plt.imshow(frame_resized, cmap="gray")

    # 线段与端点（颜色/粗细保持一致）
    for pts in lines_scaled:
        pts = pts - 0.5  # 与你提供的函数一致的 -0.5 偏移
        plt.plot(pts[:, 0], pts[:, 1],
                 color="orange",
                 linewidth=0.5,
                 )
        plt.scatter(pts[:, 0], pts[:, 1],
                    color="#33FFFF",
                    s=1.2,
                    edgecolors="none",
                    zorder=point_zorder)

    # 4) 保存
    save_dir = os.path.join('vis_ours_result', dataset_name, sequence_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{item}.png")

    # dpi=H_tgt 使导出像素≈(W_tgt,H_tgt)；无边距
    plt.savefig(save_path, dpi=H_tgt, bbox_inches="tight", pad_inches=0)
    print(colored(f"save {save_path}", "yellow"))
    if plot:
        plt.show()
    plt.close()

    return save_path


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

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def visualize_ev_label(batch_size, ev_tensors, current_labels, tidx):
    # 确保保存图像的文件夹存在
    os.makedirs('vis_input', exist_ok=True)

    # 可视化事件和标签
    for i in range(batch_size):
        if current_labels[i] is None:
            continue

        # 提取事件和标签信息
        ev_tensor_single = ev_tensors[i].detach().cpu().numpy()
        ObjectLabels_item = current_labels[i]
        label = ObjectLabels_item.object_labels.detach().cpu().numpy()

        # 计算正负事件图像
        pos_image = np.sum(ev_tensor_single[:10, :, :], axis=0)
        neg_image = np.sum(ev_tensor_single[10:20, :, :], axis=0)
        pos_image = cv2.equalizeHist(pos_image.astype(np.uint8))
        neg_image = cv2.equalizeHist(neg_image.astype(np.uint8))

        # 组合正负事件图像
        image = np.concatenate((neg_image[..., None], np.zeros((512, 512, 1)), pos_image[..., None]), axis=-1) / 255.0

        # 创建一个包含两个子图的图像（1行2列）
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        # **第一张图**：显示事件和标签
        axs[0].imshow(image)
        axs[0].set_xlim(0, 512)
        axs[0].set_ylim(0, 512)
        axs[0].set_title("Events and Labels")
        sca = 512 / 128

        # 绘制标签
        for lb in label:
            t, x1, y1, x2, y2 = lb
            # 将标签坐标缩放到 (256, 256)
            x1, y1, x2, y2 = x1 * sca, y1 * sca, x2 * sca, y2 * sca
            axs[0].plot([x1, x2], [y1, y2], color='yellow', linewidth=1, linestyle='dashed')

        # **第二张图**：仅显示事件
        axs[1].imshow(image)
        axs[1].set_xlim(0, 512)
        axs[1].set_ylim(0, 512)
        axs[1].set_title("Events Only")

        # 保存图像为文件
        image_filename = f'vis_input/{i}.png'  # 根据索引修改文件名
        plt.savefig(image_filename)
        plt.close()


class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model.backbone

        self.with_lstm = self.mdl_config.with_lstm
        self.n_dyn_junc = self.mdl_config.n_dyn_junc
        self.junc_thresh = self.mdl_config.junc_thresh
        self.n_dyn_posl = self.mdl_config.n_dyn_posl
        self.n_dyn_negl = self.mdl_config.n_dyn_negl
        self.n_pts0 = self.mdl_config.n_pts0
        self.n_pts1 = self.mdl_config.n_pts1
  
        self.mdl = YoloXDetector(full_config.model)
        self.register_buffer('tspan', torch.linspace(0, 1, self.n_pts0)[None, None, :])
        
        
        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)

        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets=None) \
            -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        return self.mdl(x=event_tensor,
                        previous_states=previous_states,
                        retrieve_detections=retrieve_detections,
                        targets=targets)

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]  #  is a list
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ] # is a list, 每一个都是一个 'SparselyBatchedObjectLabels' object
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)

        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

        total_loss = 0.0
        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            batch_size = ev_tensors.size(0)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]
                
            # 可视化训练的事件和标签
            # visualize_ev_label(batch_size, ev_tensors, sparse_obj_labels[tidx], tidx)
            if any(ann is None for ann in sparse_obj_labels[tidx]):
                continue
            loss, loss_dict, states = self.mdl.forward(x=ev_tensors, previous_states=prev_states, targets=sparse_obj_labels[tidx])

            total_loss += loss

            prev_states = states
      
        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        average_loss = total_loss / sequence_len 

        return {ObjDetOutput.SKIP_VIZ: False,
                'loss': average_loss}

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        dataset_name = self.full_config.dataset.name
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])

        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

        groudtruth = list()
        prediction = list()
        prediction_with_event = list()

        for tidx in range(sequence_len):
            ev_tensors = ev_tensor_sequence[tidx]
            gt_labels = sparse_obj_labels[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            batch_size = len(ev_tensors)

            B = ev_tensors.size(0)
          
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            jmaps, joffs, line_preds, line_scores, states = self.mdl.forward_test(x=ev_tensors,  previous_states=prev_states)

            none_indices = [i for i, ann in enumerate(sparse_obj_labels[tidx]) if ann is None]
            
            if none_indices : 
                valid_label = [t for i, t in enumerate(sparse_obj_labels[tidx]) if i not in none_indices]
                B = jmaps.size(0)
                device = jmaps.device
                valid_idx = [i for i in range(B) if i not in none_indices]
                idx_t = torch.tensor(valid_idx, device=device, dtype=torch.long)
                jmaps = jmaps.index_select(0, idx_t)
                joffs = joffs.index_select(0, idx_t)
                # 过滤 list
                line_preds  = [line_preds[i]  for i in valid_idx]
                line_scores = [line_scores[i] for i in valid_idx]

                ev_tensors = torch.tensor(ev_tensors, device=device)
                ev_tensors = ev_tensors.index_select(0, idx_t)
            else:
                valid_label = gt_labels

            # valid_label = gt_labels
            assert len(valid_label) == len(ev_tensors)
            seq_name_list, item_list = [], []
           
            prev_states = states
            idx_in_batch = 0
            for i in range(len(valid_label)):
                label_one_image = valid_label[i]
                seq_name = label_one_image.seq_name
                seq_name_list.append(seq_name)
                item_list.append(label_one_image.item)

                jmap = jmaps[i].numpy()
                joff = joffs[i].numpy()
                line_pred = line_preds[i] # N, 2, 2
                line_score = line_scores[i]
                gt_single = valid_label[i]

                labels = gt_single.object_labels[:, 1:].view(-1, 2, 2) 

                ev_pre_vis = ev_tensors[i].detach().cpu().numpy()  # (20,512,512)
                mask = line_score.detach().cpu().numpy() > 0.88
                vis_lines = line_pred.detach().cpu().numpy()[mask]

                groudtruth.append(labels)

                out = {
                        'ev_frame': ev_pre_vis,
                        'pred_lines': vis_lines,
                        'sequence_name': seq_name_list[idx_in_batch],
                        'item': item_list[idx_in_batch],
                        }
                        
                prediction_with_event.append(out)

                # label   torch.Size([M, 2, 2])  
                # line_pred   torch.Size([M, 2, 2])  
                output = {}
                output['jmap'] = jmap
                output['joff'] = joff
                output['line_pred'] = line_pred  # range  (128, 128)
                output['line_score'] = line_score

                prediction.append(output)

                idx_in_batch += 1

        assert len(groudtruth) == len(prediction)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)

        if len(groudtruth) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}
        
        output = {
            ObjDetOutput.SKIP_VIZ: True
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(groudtruth)
            self.mode_2_psee_evaluator[mode].add_predictions(prediction)

            self.mode_2_psee_evaluator[mode].add_prediction_with_event(prediction_with_event)
        return output
    
    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)
    
    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn('psee_evaluator is None in mode={}'.format(mode), UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metric = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])

            assert metric is not None

            prefix = '{}/'.format(mode_2_string[mode])   #'val/'
            step = self.trainer.global_step

            log_dict = {}
            for k, v in metric.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, 'tensor must be a scalar.\n{}=\n{}=\n{}=\n{}='.format(
                    v, type(v), value, type(value))
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict['{}{}'.format(prefix, k)] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                # self.logger.log_metrics(metrics=log_dict)
                log_str = " | ".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
                print(log_str)
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            psee_evaluator.reset_buffer()
        else:
            warn('psee_evaluator has not data in mode={}'.format(mode), UserWarning, stacklevel=2)

        return {'val/msAP': metric['msAP'], 'cal/mAPJ': metric['mAPJ']}

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
         # 获取当前 epoch 信息
        current_epoch = self.current_epoch

    
    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            print(colored('start run validate ...', 'yellow'))
            self.run_psee_evaluator(mode=mode)

        
    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            print(colored('start run test ...', 'yellow'))
            self.run_psee_evaluator(mode=mode)
        
    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    