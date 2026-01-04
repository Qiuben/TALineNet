from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from data.utils.types import FeatureMap, BackboneFeatures, LstmState, LstmStates
import ipdb
from models.layers.rnn import DWSConvLSTM2d
from models.layers.maxvit.maxvit import (
    PartitionAttentionCl,
    nhwC_2_nChw,
    get_downsample_layer_Cf2Cl,
    PartitionType)

from models.layers.hourglass.hourglass import Hourglass

from .base import BaseDetector

from termcolor import colored

class ResNetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes=None, stride=1, downsample=None):
        super().__init__()
        planes = planes or inplanes // self.expansion

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        return out
    
class MultiTaskHead(nn.Module):
    def __init__(self, inplanes, num_classes, head_size):
        super().__init__()
        assert num_classes == sum(head_size)
        planes = inplanes // 4

        heads = []
        for outplanes in head_size:
            heads.append(
                nn.Sequential(
                    nn.Conv2d(inplanes, planes, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(planes, outplanes, 1)
                )
            )
        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        return th.cat([head(x) for head in self.heads], dim=1)
    

    
class RNNDetector(BaseDetector):
    def __init__(self, mdl_config: DictConfig):
        super().__init__()
        self.with_lstm = mdl_config.with_lstm
        ###### Config ######
        in_channels = mdl_config.input_channels
        self.num_stages = mdl_config.hourglass_stack_num

        inplanes = mdl_config.inplanes
        num_feats = mdl_config.num_feats
        depth = mdl_config.depth
        num_blocks_hg = mdl_config.num_blocks_hg
        self.num_stacks = mdl_config.num_stacks
        head_size = mdl_config.head_size
        num_classes = sum(head_size)

        # config in RVT
        num_blocks = mdl_config.num_blocks

        ###### Compile if requested ######
        compile_cfg = mdl_config.get('compile', None)
        if compile_cfg is not None:
            compile_mdl = compile_cfg.enable
            if compile_mdl and th_compile is not None:
                compile_args = OmegaConf.to_container(compile_cfg.args, resolve=True, throw_on_missing=True)
                self.forward = th_compile(self.forward, **compile_args)
            elif compile_mdl:
                print('Could not compile backbone because torch.compile is not available')


        res_block = ResNetBottleneck
        head_block = lambda c_in, c_out: MultiTaskHead(c_in, c_out, head_size=head_size)

        # initial layers
        self.shallow_conv, self.shallow_res = self._make_shallow_layer(res_block, in_channels, inplanes, num_feats) 

        # Hourglass modules
        self.res = nn.ModuleList([self._make_residual(res_block, num_blocks_hg, num_feats) for _ in range(self.num_stacks)])
        self.fcs = nn.ModuleList([self._make_fc(num_feats, num_feats) for _ in range(self.num_stacks)])


        self.scores = nn.ModuleList([head_block(num_feats, num_classes) for _ in range(self.num_stacks)])
        self.fcs_ = nn.ModuleList([nn.Conv2d(num_feats, num_feats, 1) for _ in range(self.num_stacks - 1)])
        self.scores_ = nn.ModuleList([nn.Conv2d(num_classes, num_feats, 1) for _ in range(self.num_stacks - 1)])

        ##################################

        input_dim = in_channels
        self.stages = nn.ModuleList()

        for stage_idx in range(self.num_stages):
            #hourglass + lstm
            stage = RNNDetectorStage(with_lstm=self.with_lstm,
                                    dim_in = num_feats, 
                                    res_block = res_block,
                                    num_blocks = num_blocks_hg,
                                    depth = depth,
                                    stage_cfg = mdl_config.stage)
            
            self.stages.append(stage)

    def _make_residual(self, block, num_blocks_hg, inplanes, planes=None, stride=1):
        planes = planes or inplanes // block.expansion
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks_hg):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    

    def _make_shallow_layer(self, block, in_channels, inplanes, num_feats):
        shallow_conv = nn.Sequential(
            nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
        )

        shallow_res = nn.Sequential(
            self._make_residual(block, 1, inplanes, inplanes),
            nn.MaxPool2d(2, stride=2),
            self._make_residual(block, 1, inplanes * block.expansion, inplanes * block.expansion),
            self._make_residual(block, 1, inplanes * block.expansion ** 2, num_feats // block.expansion)
        )

        return shallow_conv, shallow_res
    
    def _make_fc(self, inplanes, outplanes):
        fc = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

        return fc

    def forward(self, x: th.Tensor, prev_states: Optional[LstmStates] = None)\
            -> Tuple[BackboneFeatures, LstmStates]:
        
        if prev_states is None:
            prev_states = [None] * self.num_stages

        assert len(prev_states) == self.num_stages, print(len(prev_states), self.num_stages)
        
        states: LstmStates = list()
        # 特征图
        output: Dict[int, FeatureMap] = {}

        x = self.shallow_conv(x)
        x = self.shallow_res(x)

        for i, stage in enumerate(self.stages):   
            y, state = stage(x, prev_states[i])
            states.append(state)
            y = self.res[i](y)
            y = self.fcs[i](y) # bs, 256, 128, 128

            stage_number = i + 1
            output[stage_number] = y

            if i < self.num_stacks - 1:
                fc_ = self.fcs_[i](y)
                x = x + fc_
        feature = y

        return feature, output, states


class RNNDetectorStage(nn.Module):
    """Operates with NCHW [channel-first] format as input and output.
    """

    def __init__(self,
                 with_lstm: bool,
                 dim_in: int, 
                 res_block: int,
                 num_blocks: int,
                 depth: int,
                 stage_cfg: DictConfig):
        super().__init__()
        assert isinstance(num_blocks, int) and num_blocks > 0

        lstm_cfg = stage_cfg.lstm
        self.hourglass = Hourglass(res_block, num_blocks, dim_in, depth)

        self.with_lstm = with_lstm
        if self.with_lstm:
            self.lstm = DWSConvLSTM2d(dim=dim_in,
                                    dws_conv=lstm_cfg.dws_conv,
                                    dws_conv_only_hidden=lstm_cfg.dws_conv_only_hidden,
                                    dws_conv_kernel_size=lstm_cfg.dws_conv_kernel_size,
                                    cell_update_dropout=lstm_cfg.get('drop_cell_update', 0))

     
    def forward(self, x: th.Tensor,
                h_and_c_previous: Optional[LstmState] = None,
                token_mask: Optional[th.Tensor] = None) \
            -> Tuple[FeatureMap, LstmState]:
        
        x = self.hourglass(x)
        if self.with_lstm:
            h_c_tuple = self.lstm(x, h_and_c_previous)
            x = h_c_tuple[0]
        else:
            h_c_tuple = [th.zeros_like(x), th.zeros_like(x)]
        return x, h_c_tuple
