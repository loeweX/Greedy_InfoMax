import torch

from GreedyInfoMax.vision.models import FullModel, ClassificationModel
from GreedyInfoMax.utils import model_utils


def load_model_and_optimizer(opt, num_GPU=None, reload_model=False, calc_loss=True):

    model = FullModel.FullVisionModel(
        opt, calc_loss
    )

    optimizer = []
    if opt.model_splits == 1:
        optimizer.append(
            torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
        )
    elif opt.model_splits >= 3:
        # use separate optimizer for each module, so gradients don't get mixed up
        for idx, layer in enumerate(model.encoder):
            optimizer.append(torch.optim.Adam(layer.parameters(), lr=opt.learning_rate))
    else:
        raise NotImplementedError

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model, optimizer = model_utils.reload_weights(
        opt, model, optimizer, reload_model=reload_model
    )

    return model, optimizer


def load_classification_model(opt):

    if opt.resnet == 34:
        in_channels = 256
    else:
        in_channels = 1024

    if opt.dataset == "stl10":
        num_classes = 10
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model