from optparse import OptionParser
import time
import os
import torch
import numpy as np

from GreedyInfoMax.audio.arg_parser import (
    reload_args,
    architecture_args,
    GIM_args,
    general_args,
)


def parse_args():
    # load parameters and options
    parser = OptionParser()

    parser = general_args.parse_general_args(parser)
    parser = GIM_args.parse_GIM_args(parser)
    parser = architecture_args.parse_architecture_args(parser)
    parser = reload_args.parser_reload_args(parser)

    (opt, _) = parser.parse_args()

    opt.time = time.ctime()

    # Device configuration
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opt.experiment = "audio"

    return opt


def create_log_path(opt, add_path_var=None):
    unique_path = False

    if opt.save_dir != "":
        opt.log_path = os.path.join(opt.data_output_dir, "logs", opt.save_dir)
        unique_path = True
    elif add_path_var is not None:
        opt.log_path = os.path.join(
            opt.data_output_dir, "logs", add_path_var, opt.time
        )
    else:
        opt.log_path = os.path.join(opt.data_output_dir, "logs", opt.time)

    ### very hacky way to avoid overwriting of log-files in case scripts are started at the same time
    while os.path.exists(opt.log_path) and not unique_path:
        opt.log_path += "_" + str(np.random.randint(100))

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)

    opt.log_path_latent = os.path.join(opt.log_path, "latent_space")

    if not os.path.exists(opt.log_path_latent):
        os.makedirs(opt.log_path_latent)
