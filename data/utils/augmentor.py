import collections.abc as abc
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
from warnings import filterwarnings, warn

import torch as th
import torch.distributions.categorical
from omegaconf import DictConfig
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.types import DataType, LoaderDataDictGenX
from utils.helpers import torch_uniform_sample_scalar

from termcolor import colored
import numpy as np
from scipy.ndimage import zoom


NO_LABEL_WARN_MSG = 'No Labels found. This can lead to a crash and should not happen often.'
filterwarnings('always', message=NO_LABEL_WARN_MSG)


@dataclass
class ZoomOutState:
    active: bool
    x0: int
    y0: int
    zoom_out_factor: float


@dataclass
class AugmentationState:
    apply_h_flip: bool
    apply_v_flip: bool
    clockwise_rotation: bool
    counterclockwise_rotation: bool
    # rotation: RotationState
    apply_zoom_in: bool
    zoom_out: ZoomOutState


class RandomSpatialAugmentorGenX:
    def __init__(self,
                 dataset_hw: Tuple[int, int],
                 automatic_randomization: bool,
                 augm_config: DictConfig):
        
        assert isinstance(dataset_hw, tuple)
        assert len(dataset_hw) == 2
        assert all(x > 0 for x in dataset_hw)
        assert isinstance(automatic_randomization, bool)

        self.hw_tuple = dataset_hw
        self.automatic_randomization = automatic_randomization
        self.h_flip_prob = augm_config.prob_hflip
        self.v_flip_prob = augm_config.prob_hflip

        self.rot_clockwise_prob = augm_config.rotate.clockwise_prob
        self.rot_counterclockwise_prob = augm_config.rotate.counterclockwise_prob

        assert 0 <= self.h_flip_prob <= 1
        assert 0 <= self.rot_clockwise_prob <= 1
        assert 0 <= self.rot_counterclockwise_prob <= 1

        self.augm_state = AugmentationState(
            apply_h_flip=False,
            apply_v_flip=False,
            # rotation=RotationState(active=False, angle_deg=0.0),
            clockwise_rotation= False,
            counterclockwise_rotation=False,
            apply_zoom_in=False,
            zoom_out=ZoomOutState(active=False, x0=0, y0=0, zoom_out_factor=1.0))

    def randomize_augmentation(self):
        """Sample new augmentation parameters that will be consistently applied among the items.

        This function only works with augmentations that are input-independent.
        E.g. The zoom-in augmentation parameters depend on the labels and cannot be sampled in this function.
        For the same reason, it is not a very reasonable augmentation for the streaming scenario.
        """
        self.augm_state.apply_h_flip = self.h_flip_prob > th.rand(1).item()

        self.augm_state.apply_v_flip = self.v_flip_prob > th.rand(1).item()

        self.augm_state.clockwise_rotation = self.rot_clockwise_prob > th.rand(1).item()

        self.augm_state.counterclockwise_rotation = self.rot_counterclockwise_prob > th.rand(1).item()

    def _rotate(self, data_dict: LoaderDataDictGenX, type_: str) -> LoaderDataDictGenX:
        # angle_deg = self.augm_state.rotation.angle_deg
        return {k: RandomSpatialAugmentorGenX._rotate_recursive(v,  rotate_type=type_, datatype=k)
                for k, v in data_dict.items()}

    @staticmethod
    def _rotate_tensor(input_: Any, rotate_type: str, datatype: DataType):
        assert isinstance(input_, th.Tensor)
        assert rotate_type in ['clockwise', 'counterclockwise']

        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:

            input_ = input_.detach().cpu().numpy()
            
            if rotate_type == 'clockwise':

                # 顺时针旋转图像 90°
                img_rotated = np.transpose(input_, (0, 2, 1))  # 先将宽和高维度互换
                img_rotated = np.flip(img_rotated, axis=2).copy()  # 然后沿宽度方向翻转 

            if rotate_type == 'counterclockwise':
                img_rotated = np.transpose(input_, (0, 2, 1)) 
                img_rotated = np.flip(img_rotated, axis=1).copy()
        output = th.tensor(img_rotated)
        return output
    
       
    @classmethod
    def _rotate_recursive(cls, input_: Any, rotate_type : str, datatype: DataType):

        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        
        # 事件
        if isinstance(input_, th.Tensor):
            # return input_
            return cls._rotate_tensor(input_=input_, rotate_type= rotate_type, datatype=datatype)
        
        #标签
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ
            assert rotate_type in ['clockwise', 'counterclockwise']

            if rotate_type == 'clockwise':
                input_.rotate_clockwise_()

            if rotate_type == 'counterclockwise':
                input_.rotate_counterclockwise_()
            
            return input_
        
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._rotate_recursive(x, rotate_type = rotate_type, datatype = datatype) \
                    for x in input_]
        
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._rotate_recursive(value, rotate_type = rotate_type, datatype=datatype) \
                    for key, value in input_.items()}
        
        else: 
            print('error')

    @staticmethod
    def _flip(data_dict: LoaderDataDictGenX, type_: str) -> LoaderDataDictGenX:
        assert type_ in {'h', 'v'}
        return {k: RandomSpatialAugmentorGenX._flip_recursive(v, flip_type=type_, datatype=k) \
                for k, v in data_dict.items()}

    @staticmethod
    def _flip_tensor(input_: Any, flip_type: str, datatype: DataType):
        assert isinstance(input_, th.Tensor)
        flip_axis = -1 if flip_type == 'h' else -2
        if datatype == DataType.IMAGE or datatype == DataType.EV_REPR:
            return th.flip(input_, dims=[flip_axis])
        
        raise NotImplementedError

    @classmethod
    def _flip_recursive(cls, input_: Any, flip_type: str, datatype: DataType):
        if datatype in (DataType.IS_PADDED_MASK, DataType.IS_FIRST_SAMPLE):
            return input_
        
        # 事件
        if isinstance(input_, th.Tensor):

            return cls._flip_tensor(input_=input_, flip_type=flip_type, datatype=datatype)
        
        # Label
        if isinstance(input_, ObjectLabels) or isinstance(input_, SparselyBatchedObjectLabels):
            assert datatype == DataType.OBJLABELS or datatype == DataType.OBJLABELS_SEQ

            label_tensor = input_[0].object_labels
            
            if flip_type == 'h':
                # in-place modification
                input_.flip_h_()
            if flip_type == 'v':
                input_.flip_v_()

            return input_
            
            
        if isinstance(input_, abc.Sequence):
            return [RandomSpatialAugmentorGenX._flip_recursive(x, flip_type=flip_type, datatype=datatype) \
                    for x in input_]
        if isinstance(input_, abc.Mapping):
            return {key: RandomSpatialAugmentorGenX._flip_recursive(value, flip_type=flip_type, datatype=datatype) \
                    for key, value in input_.items()}
        

    @staticmethod
    def _hw_from_data(data_dict: LoaderDataDictGenX) -> Tuple[int, int]:
        height = None
        width = None
        for k, v in data_dict.items():
            _hw = None
            if k == DataType.OBJLABELS or k == DataType.OBJLABELS_SEQ:
                hw = v.input_size_hw
                if hw is not None:
                    _hw = v.input_size_hw
            elif k in (DataType.IMAGE, DataType.FLOW, DataType.EV_REPR):
                _hw = v[0].shape[-2:]
            if _hw is not None:
                _height, _width = _hw
                if height is None:
                    assert width is None
                    height, width = _height, _width
                else:
                    assert height == _height and width == _width
        assert height is not None
        assert width is not None
        return height, width

    def __call__(self, data_dict: LoaderDataDictGenX):
        """
        :param data_dict: LoaderDataDictGenX type, image-based tensors must have (*, h, w) shape.
        :return: map with same keys but spatially augmented values.
        """
        if self.automatic_randomization:
            self.randomize_augmentation()

        label = data_dict[DataType.OBJLABELS_SEQ]
      
        # 这里标签是 128， 128

        # 上下翻转
        if self.augm_state.apply_v_flip:
            data_dict = self._flip(data_dict, type_='v')

        if self.augm_state.apply_h_flip:
            data_dict = self._flip(data_dict, type_='h')

        if self.augm_state.clockwise_rotation:
            data_dict = self._rotate(data_dict, type_ = 'clockwise')

        if self.augm_state.counterclockwise_rotation:
            data_dict = self._rotate(data_dict, type_ = 'counterclockwise')

        return data_dict

