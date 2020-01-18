from optparse import OptionGroup

def parser_reload_args(parser):
    group = OptionGroup(parser, "Reloading pretrained model options")

    ### Options to load pretrained models
    group.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start GIM training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to pre-trained model needs to be specified in opt.model_path)",
    )
    group.add_option(
        "--model_path",
        type="string",
        default=".",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    group.add_option(
        "--model_num",
        type="string",
        default="100",
        help="Number of the saved model to be used for training the linear classifier"
        "(loaded using model_path + model_X.ckpt, where X is the model_num passed here)",
    )
    group.add_option(
        "--model_type",
        type="int",
        default=0,
        help="Which type of model to use for training of linear classifier on downstream task:"
        "0 - pretrained GreedyInfoMax/CPC model"
        "1 - randomly initialized model"
        "2 - fully supervised model",
    )
    parser.add_option_group(group)
    return parser