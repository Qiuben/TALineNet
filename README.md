# TALineNet: Temporally-Aware Continuous Line Segment Detection in Event Streams
This repository contains the official PyTorch implementation of the paper: “TALineNet: Temporally-Aware Continuous Line Segment
Detection in Event Streams”

## Table of Contents
- [Dataset](#dataset)
- [Installation](#installatuion)
- [Training and Testing](#train-test)

## Dataset
You can download the synthetic dataset Ev-WireframeSeq as well as the real-scene datset Ev-LineSeq from [OneDrive](https://1drv.ms/f/c/93289205239bc375/IgC0F7BmFkD3RqQo51EgklVGAVn3aSoHt5aeoidAgLjtyMM?e=ANqbof). 
This link provides the event representations, which is ready for direct network processing. If you need the raw event data, please contact us via email.


## Installation
For ease of reproducibility, you are suggested to install miniconda (or anaconda if you prefer) before following executing the following commands.

`git clone https://github.com/Qiuben/TALineNet`

`cd TALineNet`

`conda create -y -n TALineNet python=3.9`

`pip install torch==2.1.1 torchvision==0.16.1 torchdata==0.7.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118`

`pip install wandb pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn ipdb timm opencv-python-headless pytorch_lightning==1.8.6 numpy==1.26.3`



## Training and Testing

