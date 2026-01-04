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
We have trained and tested our models on CUDA 11.8, Python 3.10.14, torch 1.10.1.

For ease of reproducibility, you are suggested to install miniconda (or anaconda if you prefer) before following executing the following commands.

`git clone https://github.com/Qiuben/EvLSD-IED`

`cd EvLSD-IED`

`pip install -r requirements.txt`


## Testing Pre-trained Models
You can download the pretrained model on E-wirferame 
from [OneDrive](https://1drv.ms/f/c/93289205239bc375/EoSWLjyUd4JDgzARyahZtTcBjfqtTmDchmW_w_GWYltV8A?e=vkLnVt).

