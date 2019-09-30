import torch
import torch.nn as nn
import os


def distribute_over_GPUs(opt, model, num_GPU):
    ## distribute over GPUs
    if opt.device.type != "cpu":
        if num_GPU is None:
            model = nn.DataParallel(model)
            num_GPU = torch.cuda.device_count()
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
        else:
            assert (
                num_GPU <= torch.cuda.device_count()
            ), "You cant use more GPUs than you have."
            model = nn.DataParallel(model, device_ids=list(range(num_GPU)))
            opt.batch_size_multiGPU = opt.batch_size * num_GPU
    else:
        model = nn.DataParallel(model)
        opt.batch_size_multiGPU = opt.batch_size

    model = model.to(opt.device)
    print("Using", num_GPU, "GPUs")

    return model, num_GPU


def reload_weights(opt, model, optimizer, reload_model):
    # reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:
        print("Loading weights from ", opt.model_path)
        if opt.device.type != "cpu":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num))
                )
            )
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location="cpu"
                )
            )
    # reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)
        if opt.device.type != "cpu":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.start_epoch))
                )
            )
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.start_epoch)),
                    map_location="cpu"
                )
            )

        for i, optim in enumerate(optimizer):
            if opt.device.type != "cpu":
                optim.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                        )
                    )
                )
            else:
                optim.load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                        ),
                        map_location="cpu"
                    )
                )
    else:
        print("Randomly initialized model")

    return model, optimizer
