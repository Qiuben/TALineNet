import torch.nn as nn
import torch.nn.functional as F

import wandb
# import matplotlib.pyplot as plt
from termcolor import colored
import numpy as np
import cv2
import ipdb


class Hourglass(nn.Module):
    def __init__(self, res_block, num_blocks, inplanes, depth):
        super().__init__()
        self.depth = depth

        self.encoders = self._make_encoder(res_block, num_blocks, inplanes, depth)
        self.decoders = self._make_decoder(res_block, num_blocks, inplanes, depth)

    def _make_residual(self, block, num_blocks, inplanes):
        layers = [block(inplanes) for _ in range(num_blocks)]

        return nn.Sequential(*layers)

    def _make_encoder(self, block, num_blocks, inplanes, depth):
        encoders = []
        for i in range(depth):
            res = [self._make_residual(block, num_blocks, inplanes) for _ in range(2)]
            if i == depth - 1:
                res.append(self._make_residual(block, num_blocks, inplanes))
            encoders.append(nn.ModuleList(res))

        return nn.ModuleList(encoders)

    def _make_decoder(self, block, num_blocks, inplanes, depth):
        decoders = [self._make_residual(block, num_blocks, inplanes) for _ in range(depth)]

        return nn.ModuleList(decoders)

    def _encoder_forward(self, x):
        out = []
        for i in range(self.depth):
            out.append(self.encoders[i][0](x))
            x = self.encoders[i][1](F.max_pool2d(x, 2, stride=2))

            if i == self.depth - 1:
                out.append(self.encoders[i][2](x))

        return out[::-1]

    def _decoder_forward(self, x):
        out = x[0]

        for i in range(self.depth):
            up = x[i + 1]
            low = self.decoders[i](out)
            low = F.interpolate(low, scale_factor=2)
            out = low + up

        return out

    def forward(self, x):
        x = self._encoder_forward(x)  # 5个值的list， 8， 16， 32， 64， 128
        out = self._decoder_forward(x)
        
        return out
