from pathlib import Path
from typing import Any, List, Optional

import h5py, ipdb
try:
    import hdf5plugin
except ImportError:
    pass
import numpy as np
import torch, random
from torchdata.datapipes.map import MapDataPipe

from data.genx_utils.labels import ObjectLabelFactory, ObjectLabels
from data.utils.spatial import get_original_hw
from data.utils.types import DatasetType
from utils.timers import TimerDummy as Timer

from termcolor import colored
import cv2
import util.camera as cam
import util.bezier as bez

def get_event_representation_dir(path: Path, ev_representation_name: str) -> Path:
    ev_repr_dir = path / 'event_representations_v2' / ev_representation_name
    assert ev_repr_dir.is_dir(), f'{ev_repr_dir}'
    return ev_repr_dir


def get_objframe_idx_2_repr_idx(path: Path, ev_representation_name: str) -> np.ndarray:
    ev_repr_dir = get_event_representation_dir(path=path, ev_representation_name=ev_representation_name)
    objframe_idx_2_repr_idx = np.load(str(ev_repr_dir / 'objframe_idx_2_repr_idx.npy'))
    return objframe_idx_2_repr_idx

def resize_batch(input_array, target_height, target_width):
    batch_size, num_frames, original_height, original_width = input_array.shape
    # 创建新的数组以保存结果
    resized_array = np.zeros((batch_size, num_frames, target_height, target_width), dtype=input_array.dtype)
    
    # 遍历 batch 和帧，逐个进行 resize
    for b in range(batch_size):
        for f in range(num_frames):
            resized_array[b, f] = cv2.resize(input_array[b, f], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    return resized_array

def line2meta(label):
    label = label.detach().cpu()
    lines = label[:, 1:].view(-1, 2, 2)
    lines = np.array(lines)
    camera = cam.Pinhole()
    pts_list = camera.interp_line(lines)
    lines = bez.fit_line(pts_list, order=2)[0]
    centers = lines[:, 1]
    lines = bez.fit_line(pts_list, order=1)[0]

    label_dict = save_npz(lines.copy(), centers)
    map, meta = transform(label_dict)
    return map, meta

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


class SequenceBase(MapDataPipe):
    """
    Structure example of a sequence:
    .
    ├── event_representations_v2
    │ └── ev_representation_name
    │     ├── event_representations.h5
    │     ├── objframe_idx_2_repr_idx.npy
    │     └── timestamps_us.npy
    └── labels_v2
        ├── labels.npz
        └── timestamps_us.npy
    """

    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool):
        assert sequence_length >= 1
        assert path.is_dir()
        seq_name = path.name
        self.only_load_end_labels = only_load_end_labels

        ev_repr_dir = get_event_representation_dir(path=path, ev_representation_name=ev_representation_name)

        labels_dir = path / 'labels_v2'
        assert labels_dir.is_dir()

        self.seq_len = sequence_length

        ds_factor_str = '_ds2_nearest' if downsample_by_factor_2 else ''
        self.ev_repr_file = ev_repr_dir / 'event_representations{}.h5'.format(ds_factor_str)
        assert self.ev_repr_file.exists(), 'self.ev_repr_file=' + str(self.ev_repr_file)

        height, width = get_original_hw(dataset_type)

        with Timer(timer_name='prepare labels'):
            label_data = np.load(str(labels_dir / 'labels.npz'))
            objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
            labels = label_data['labels']

            label_factory = ObjectLabelFactory.from_structured_array(
                object_labels=labels,
                objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                input_size_hw=(128, 128),
                original_size = (height, width),
                downsample_factor=2 if downsample_by_factor_2 else None,
                seq_name = seq_name)
            
            
            self.label_factory = label_factory

        with Timer(timer_name='load objframe_idx_2_repr_idx'):
            self.objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
                path=path, ev_representation_name=ev_representation_name)
            
        with Timer(timer_name='construct repr_idx_2_objframe_idx'):
            self.repr_idx_2_objframe_idx = dict(zip(self.objframe_idx_2_repr_idx,
                                                    range(len(self.objframe_idx_2_repr_idx))))

    def _get_labels_from_repr_idx(self, repr_idx: int) -> Optional[ObjectLabels]:

        objframe_idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        
        # 发现在 self.label_factory 中只有第一项的形状是[5， N], 其它都是 [0, N]
        return None if objframe_idx is None else self.label_factory[objframe_idx]

    def _get_event_repr_torch(self, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        assert end_idx > start_idx
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            ev_repr = h5f['data'][start_idx:end_idx]

        ev_repr = resize_batch(ev_repr, 512, 512)

        ev_repr = torch.from_numpy(ev_repr)
        if ev_repr.dtype != torch.uint8:
            ev_repr = torch.asarray(ev_repr, dtype=torch.float32)

        ev_repr = torch.split(ev_repr, 1, dim=0)
        # remove first dim that is always 1 due to how torch.split works
        ev_repr = [x[0] for x in ev_repr]
        return ev_repr

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError
