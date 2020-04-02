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
    print("Let's use", num_GPU, "GPUs!")

    return model, num_GPU


def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)


def reload_weights(opt, model, optimizer, reload_model):
    ## reload weights for training of the linear classifier
    if (opt.model_type == 0) and reload_model:  # or opt.model_type == 2)
        print("Loading weights from ", opt.model_path)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(opt.model_path, "model_{}.ckpt".format(opt.model_num)),
                    map_location=opt.device.type,
                )
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.model_num),
                        ),
                         map_location=opt.device.type,
                    )
                )

    ## reload weights and optimizers for continuing training
    elif opt.start_epoch > 0:
        print("Continuing training from epoch ", opt.start_epoch)

        if opt.experiment == "audio":
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path, "model_{}.ckpt".format(opt.start_epoch)
                    ),
                    map_location=opt.device.type,
                ),
                strict=False,
            )
        else:
            for idx, layer in enumerate(model.module.encoder):
                model.module.encoder[idx].load_state_dict(
                    torch.load(
                        os.path.join(
                            opt.model_path,
                            "model_{}_{}.ckpt".format(idx, opt.start_epoch),
                        ),
                        map_location=opt.device.type,
                    )
                )

        for i, optim in enumerate(optimizer):
            if opt.model_splits > 3 and i > 2:
                break
            optim.load_state_dict(
                torch.load(
                    os.path.join(
                        opt.model_path,
                        "optim_{}_{}.ckpt".format(str(i), opt.start_epoch),
                    ),
                    map_location=opt.device.type,
                )
            )
    else:
        print("Randomly initialized model")

    return model, optimizer