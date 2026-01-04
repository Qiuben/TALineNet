import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch, time
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra, cv2
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    # print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus

    print(colored('use gpu {}'.format(gpus), 'cyan'))
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = WandbLogger(project="RVT")
    ckpt_path = Path(config.checkpoint)

    print(colored('load checkpoint from {}'.format(ckpt_path), 'cyan'))
    # ---------------------
    # Model
    # ---------------------

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config})

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )

    start = time.time()
    with torch.inference_mode():
        if config.use_test_set:
            print(colored('run trainer.test!', 'cyan'))
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            print(colored('run trainer.validate!', 'cyan'))
            trainer.validate(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))

    end = time.time()
    total_time = end - start

    print(colored('total_time:{} s'.format(total_time), 'yellow'))
    FPS = 2400 / total_time
    print(colored('FPS is {}'.format(FPS)))


if __name__ == '__main__':
    main()
