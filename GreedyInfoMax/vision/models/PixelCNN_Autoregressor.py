import torch
import torch.nn as nn

from GreedyInfoMax.vision.models import PixelCNN, InfoNCE_Loss
from GreedyInfoMax.utils import model_utils

class PixelCNN_Autoregressor(torch.nn.Module):
    def __init__(self, opt, in_channels, pixelcnn_layers=4, calc_loss=True, **kwargs):
        super().__init__()
        self.opt = opt
        self.calc_loss = calc_loss

        layer_objs = [
            PixelCNN.PixelCNNGatedLayer.primary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
        ]
        layer_objs = layer_objs + [
            PixelCNN.PixelCNNGatedLayer.secondary(
                in_channels, in_channels, 3, mask_mode="only_vert", **kwargs
            )
            for _ in range(1, pixelcnn_layers)
        ]

        self.stack = PixelCNN.PixelCNNGatedStack(*layer_objs)
        self.stack_out = nn.Conv2d(in_channels, in_channels, 1)

        self.loss = InfoNCE_Loss.InfoNCE_Loss(
            opt, in_channels=in_channels, out_channels=in_channels
        )

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                if m is self.stack_out:
                    # nn.init.kaiming_normal_(
                    #     m.weight, mode="fan_in", nonlinearity="relu"
                    # )
                    model_utils.makeDeltaOrthogonal(
                        m.weight, nn.init.calculate_gain("relu")
                    )
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_in", nonlinearity="tanh"
                    )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def forward(self, input):
        _, c_out, _ = self.stack(input, input)  # Bc, C, H, W
        c_out = self.stack_out(c_out)

        assert c_out.shape[1] == input.shape[1]

        if self.calc_loss:
            loss = self.loss(input, c_out)
        else:
            loss = None

        return c_out, loss
