import torch
import torch.nn as nn

from GreedyInfoMax.audio.models import independent_module
from GreedyInfoMax.utils import utils


class FullModel(nn.Module):
    def __init__(
        self,
        opt,
        kernel_sizes,
        strides,
        padding,
        enc_hidden,
        reg_hidden,
        calc_accuracy=False,
    ):
        """
        Entire CPC model that can be split into smaller chunks for training
        """
        super(FullModel, self).__init__()

        self.opt = opt
        self.reg_hidden = reg_hidden
        self.enc_hidden = enc_hidden

        # load model
        self.fullmodel = nn.ModuleList([])

        if self.opt.model_splits == 1:
            # CPC model
            self.fullmodel.append(
                independent_module.IndependentModule(
                    opt,
                    enc_kernel_sizes=kernel_sizes,
                    enc_strides=strides,
                    enc_padding=padding,
                    enc_hidden=enc_hidden,
                    reg_hidden=reg_hidden,
                    calc_accuracy=calc_accuracy,
                )
            )
        elif self.opt.model_splits == 5:
            # GIM model, where the last encoding layer is trained together with the autoregressor
            enc_input = 1
            last_idx = len(kernel_sizes) - 1

            for i in range(last_idx):
                self.fullmodel.append(
                    independent_module.IndependentModule(
                        opt,
                        enc_input=enc_input,
                        enc_kernel_sizes=[kernel_sizes[i]],
                        enc_strides=[strides[i]],
                        enc_padding=[padding[i]],
                        enc_hidden=enc_hidden,
                        reg_hidden=reg_hidden,
                        use_autoregressive=self.opt.use_autoregressive,
                        calc_accuracy=calc_accuracy,
                    )
                )
                enc_input = enc_hidden

            self.fullmodel.append(
                independent_module.IndependentModule(
                    opt,
                    enc_input=enc_input,
                    enc_kernel_sizes=[kernel_sizes[last_idx]],
                    enc_strides=[strides[last_idx]],
                    enc_padding=[padding[last_idx]],
                    enc_hidden=enc_hidden,
                    reg_hidden=reg_hidden,
                    use_autoregressive=True,
                    calc_accuracy=calc_accuracy,
                )
            )
        elif (
            self.opt.model_splits == 6
        ):  # GIM model in which the last autoregressive layer is trained independently
            enc_input = 1

            for i in range(len(kernel_sizes)):
                self.fullmodel.append(
                    independent_module.IndependentModule(
                        opt,
                        enc_input=enc_input,
                        enc_kernel_sizes=[kernel_sizes[i]],
                        enc_strides=[strides[i]],
                        enc_padding=[padding[i]],
                        enc_hidden=enc_hidden,
                        reg_hidden=reg_hidden,
                        use_autoregressive=self.opt.use_autoregressive,
                        calc_accuracy=calc_accuracy,
                    )
                )
                enc_input = enc_hidden

            if not self.opt.use_autoregressive:
                # append separate autoregressive layer
                self.fullmodel.append(
                    independent_module.IndependentModule(
                        opt,
                        enc_input=enc_input,
                        enc_hidden=enc_hidden,
                        reg_hidden=reg_hidden,
                        use_encoder=False,
                        enc_kernel_sizes=None,
                        enc_strides=None,
                        enc_padding=None,
                        use_autoregressive=True,
                        calc_accuracy=calc_accuracy,
                    )
                )
        else:
            raise Exception("Invalid option for opt.model_splits")

    def forward(self, x, filename=None, start_idx=None, n=6):
        model_input = x

        cur_device = utils.get_device(self.opt, x)

        # first dimension is used for concatenating results from different GPUs
        loss = torch.zeros(1, len(self.fullmodel), device=cur_device)
        accuracy = torch.zeros(1, len(self.fullmodel), device=cur_device)

        if n == 6:  # train all layers at once
            for idx, layer in enumerate(self.fullmodel):
                loss[:, idx], accuracy[:, idx], _, z = layer(
                    model_input, filename, start_idx
                )
                model_input = z.permute(0, 2, 1).detach()
        else:
            """
            forward to the layer that we want to train and only output that layer's loss
            (all other values stay at zero initialization)
            This does not reap the memory benefits that would be possible if we trained layers completely separately 
            (by training a layer and saving its output as the dataset to train the next layer on), but enables us 
            to test the behaviour of the model for greedy iterative training
            """
            assert (
                self.opt.model_splits == 5 or self.opt.model_splits == 6
            ), "Works only for GIM model training"

            for idx, layer in enumerate(self.fullmodel[: n + 1]):
                if idx == n:
                    loss[:, idx], accuracy[:, idx], _, _ = layer(
                        model_input, filename, start_idx
                    )
                else:
                    _, z = layer.get_latents(model_input)
                    model_input = z.permute(0, 2, 1).detach()

        return loss

    def forward_through_n_layers(self, x, n):
        if self.opt.model_splits == 1:
            if n > 4:
                model_input = x
                for idx, layer in enumerate(self.fullmodel):
                    c, z = layer.get_latents(model_input)
                    model_input = z.permute(0, 2, 1).detach()
                x = c
            else:
                x = self.fullmodel[0].encoder.forward_through_n_layers(
                    x, n+1
                )
                x = x.permute(0, 2, 1)
        elif self.opt.model_splits == 6 or self.opt.model_splits == 5:
            model_input = x
            for idx, layer in enumerate(self.fullmodel[: n + 1]):
                c, z = layer.get_latents(model_input)
                model_input = z.permute(0, 2, 1).detach()
            if n < 5:
                x = z
            else:
                x = c

        return x