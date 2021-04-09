import torch
import torch.nn as nn
import os

from GreedyInfoMax.audio.models import full_model
from GreedyInfoMax.utils import model_utils


def load_model_and_optimizer(
    opt, reload_model=False, calc_accuracy=False, num_GPU=None
):

    # Original dimensions given in CPC paper (Oord et al.).
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    padding = [2, 2, 2, 2, 1]
    enc_hidden = 512
    reg_hidden = 256

    # Initialize model.
    model = full_model.FullModel(
        opt,
        kernel_sizes=kernel_sizes,
        strides=strides,
        padding=padding,
        enc_hidden=enc_hidden,
        reg_hidden=reg_hidden,
        calc_accuracy=calc_accuracy,
    )

    # Run on only one GPU for supervised losses.
    if opt.loss == 2 or opt.loss == 1:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    model, optimizer = model_utils.reload_weights(opt, model, optimizer, reload_model)

    model.train()
    print(model)

    return model, optimizer
