from optparse import OptionGroup

def parse_architecture_args(parser):
    group = OptionGroup(parser, "Architecture options")

    group.add_option(
        "--loss",
        type="int",
        default=0,
        help="Choose between different loss functions to be used for training:"
        "0 - InfoNCE loss"
        "1 - supervised training using the phone labels"
        "2 - supervised training using the speaker labels",
    )
    group.add_option(
        "--model_splits",
        type="int",
        default=6,
        help="Number of individually trained 'layers' that the original model should be split into "
             "(options: "
             "1 - corresponds to the CPC architecture from Oord et al.,"
             "5 - training the last convolutional and the autoregressive layer together,"
             "6 - Greedy InfoMax as presented in the paper, every layer trained individually)",
    )
    group.add_option(
        "--use_autoregressive",
        action="store_true",
        default=False,
        help="Boolean to decide whether to use autoregressive audio_models in the lower modules. "
        "If set to false, only the final module has an autoregressive model on top.",
    )
    group.add_option(
        "--remove_BPTT",
        action="store_true",
        default=False,
        help="Boolean to decide whether to use BPTT (=False)"
        "or detach gradients for every time-step (=True)"
        "(only relevant for the training of the autoregressive layer)",
    )
    parser.add_option_group(group)
    return parser