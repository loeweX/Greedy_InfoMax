from optparse import OptionGroup

def parse_GIM_args(parser):
    group = OptionGroup(parser, "Greedy InfoMax training options")
    group.add_option(
        "--learning_rate", type="float", default=2e-4, help="Learning rate"
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=5,
        help="Time steps to predict into future",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=16,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--model_splits",
        type="int",
        default=3,
        help="Number of individually trained modules that the original model should be split into "
             "options: 1 (normal end-to-end backprop) or 3 (default used in experiments of paper)",
    )
    group.add_option(
        "--train_module",
        type="int",
        default=3,
        help="Index of the module to be trained individually (0-2), "
        "or training network as one (3)",
    )

    parser.add_option_group(group)
    return parser
