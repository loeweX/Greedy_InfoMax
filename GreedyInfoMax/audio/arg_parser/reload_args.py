from optparse import OptionGroup

def parser_reload_args(parser):
    group = OptionGroup(parser, "Reloading pretrained model options")

    # Options to load pretrained audio_models
    group.add_option(
        "--start_epoch",
        type="int",
        default=0,
        help="Epoch to start CPC training from: "
        "v=0 - start training from scratch, "
        "v>0 - load pre-trained model that was trained for v epochs and continue training "
        "(path to model is specified in opt.model_path)",
    )
    group.add_option(
        "--model_path",
        type="string",
        default="",
        help="Directory of the saved model (path within --data_input_dir)",
    )
    group.add_option(
        "--model_num",
        type="string",
        default="",
        help="Number of the saved model to be used for testing pre-trained models"
        "(loaded using model_path + model_X.ckpt, where X is the model_num passed here)",
    )
    group.add_option(
        "--model_type",
        type="int",
        default=0,
        help="Which type of model to use for additional trainings of linear classifiers:"
        "0 - pretrained CPC model"
        "1 - randomly initialized model"
        "2 - fully supervised training model",
    )
    parser.add_option_group(group)
    return parser
