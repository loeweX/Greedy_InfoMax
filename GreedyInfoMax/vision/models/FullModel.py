import torch
import torch.nn as nn

from GreedyInfoMax.vision.models import PixelCNN_Autoregressor, Resnet_Encoder


class FullVisionModel(torch.nn.Module):
    def __init__(self, opt, calc_loss):
        super().__init__()
        self.opt = opt
        self.contrastive_samples = self.opt.negative_samples
        print("Contrasting against ", self.contrastive_samples, " negative samples")
        self.calc_loss = calc_loss

        if self.opt.model_splits == 1 and not self.opt.loss == 1:
            # building the CPC model including the autoregressive PixelCNN on top of the ResNet
            self.employ_autoregressive = True
        else:
            self.employ_autoregressive = False

        self.model, self.encoder, self.autoregressor = self._create_full_model(opt)

        print(self.model)

    def _create_full_model(self, opt):

        block_dims = [3, 4, 6]
        num_channels = [64, 128, 256]

        full_model = nn.ModuleList([])
        encoder = nn.ModuleList([])

        if opt.resnet == 34:
            self.block = Resnet_Encoder.PreActBlockNoBN
        elif opt.resnet == 50:
            self.block = Resnet_Encoder.PreActBottleneckNoBN
        else:
            raise Exception("Undefined parameter choice")

        if opt.grayscale:
            input_dims = 1
        else:
            input_dims = 3

        output_dims = num_channels[-1] * self.block.expansion

        if opt.model_splits == 1:
            encoder.append(
                Resnet_Encoder.ResNet_Encoder(
                    opt,
                    self.block,
                    block_dims,
                    num_channels,
                    0,
                    calc_loss=False,
                    input_dims=input_dims,
                )
            )
        elif opt.model_splits == 3:
            for idx, _ in enumerate(block_dims):
                encoder.append(
                    Resnet_Encoder.ResNet_Encoder(
                        opt,
                        self.block,
                        [block_dims[idx]],
                        [num_channels[idx]],
                        idx,
                        calc_loss=False,
                        input_dims=input_dims,
                    )
                )
        else:
            raise NotImplementedError

        full_model.append(encoder)

        if self.employ_autoregressive:
            autoregressor = PixelCNN_Autoregressor.PixelCNN_Autoregressor(
                opt, in_channels=output_dims, calc_loss=True
            )

            full_model.append(autoregressor)
        else:
            autoregressor = None

        return full_model, encoder, autoregressor


    def forward(self, x, label, n=3):
        model_input = x

        if self.opt.device.type != "cpu":
            cur_device = x.get_device()
        else:
            cur_device = self.opt.device

        n_patches_x, n_patches_y = None, None

        loss = torch.zeros(1, self.opt.model_splits, device=cur_device) #first dimension for multi-GPU training
        accuracies = torch.zeros(1, self.opt.model_splits, device=cur_device) #first dimension for multi-GPU training

        for idx, module in enumerate(self.encoder[: n+1]):
            h, z, cur_loss, cur_accuracy, n_patches_x, n_patches_y = module(
                model_input, n_patches_x, n_patches_y, label
            )
            # Detach z to make sure no gradients are flowing in between modules
            # we can detach z here, as for the CPC model the loop is only called once and h is forward-propagated
            model_input = z.detach()

            if cur_loss is not None:
                loss[:, idx] = cur_loss
                accuracies[:, idx] = cur_accuracy

        if self.employ_autoregressive and self.calc_loss:
            c, loss[:, -1] = self.autoregressor(h)
        else:
            c = None

            if self.opt.model_splits == 1 and cur_loss is not None:
                loss[:, -1] = cur_loss
                accuracies[:, -1] = cur_accuracy

        return loss, c, h, accuracies


    def switch_calc_loss(self, calc_loss):
        ## by default models are set to not calculate the loss as it is costly
        ## this function can enable the calculation of the loss for training
        self.calc_loss = calc_loss
        if self.opt.model_splits == 1 and self.opt.loss == 0:
            self.autoregressor.calc_loss = calc_loss

        if self.opt.model_splits == 1 and self.opt.loss == 1:
            self.encoder[-1].calc_loss = calc_loss

        if self.opt.model_splits > 1:
            if self.opt.train_module == self.opt.model_splits:
                for i, layer in enumerate(self.encoder):
                    layer.calc_loss = calc_loss
            else:
                self.encoder[self.opt.train_module].calc_loss = calc_loss
