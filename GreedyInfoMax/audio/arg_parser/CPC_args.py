from optparse import OptionGroup

def parse_CPC_args(parser):
    group = OptionGroup(parser, "CPC training options")
    group.add_option(
        "--learning_rate", type="float", default=1.5e-4, help="Learning rate"
    )
    group.add_option(
        "--prediction_step",
        type="int",
        default=12,
        help="Time steps k to predict into future",
    )
    group.add_option(
        "--negative_samples",
        type="int",
        default=10,
        help="Number of negative samples to be used for training",
    )
    group.add_option(
        "--sampling_method",
        type="int",
        default=1,
        help="Which type of method to use for negative sampling: \n"
        "0 - inside the loop for the prediction time-steps. Slow, but samples from all but the current pos sample \n"
        "1 - outside the loop for prediction time-steps, "
             "Low probability (<0.1%) of sampling the positive sample as well. \n"
        "2 - outside the loop for prediction time-steps. Sampling only within the current sequence"
             "Low probability of sampling the positive sample as well. \n"
    )
    group.add_option(
        "--train_layer",
        type="int",
        default=6,
        help="Index of the layer to be trained individually (0-5), "
        "or training network as one (6)",
    )
    parser.add_option(
        "--subsample",
        action="store_true",
        default=False,
        help="Boolean to decide whether to subsample from the total sequence lengh within intermediate layers",
    )
    parser.add_option_group(group)
    return parser
