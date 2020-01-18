# Loosely derived from https://github.com/jzbontar/pixelcnn-pytorch/blob/master/main.py
#  and moreso derived from https://github.com/rampage644/wavenet/blob/master/wavenet/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#owidth  = floor((width  + 2*padW - kW) / dW + 1)
#oheight = floor((height + 2*padH - kH) / dH + 1)
#dW is stride, assuming 1:
# kW // 2 = padW
def same_padding(kernel_size):
    # assumming stride 1
    if isinstance(kernel_size, int):
        return kernel_size // 2
    else:
        return (kernel_size[0] // 2, kernel_size[1] // 2)

# PyTorch port of
class MaskedConvolution2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
            *args, mask='B', vertical=False, mask_mode="noblind", **kwargs):
        if "padding" not in kwargs:
            assert "stride" not in kwargs
            kwargs["padding"] = same_padding(kernel_size)
        remove = {"conditional_features", "conditional_image_channels"}
        for feature in remove:
            if feature in kwargs:
                del kwargs[feature]
        super(MaskedConvolution2D, self).__init__(in_channels,
                out_channels, kernel_size, *args, **kwargs)
        Cout, Cin, kh, kw = self.weight.size()
        pre_mask = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        yc, xc = kh // 2, kw // 2

        assert mask_mode in {"noblind", "turukin", "fig1-van-den-oord", "none", "only_vert"}

        if mask_mode == "none":
            pass
        elif mask_mode == "only_vert":
            pre_mask[:, :, yc + 1:, :] = 0.0
        elif mask_mode == "noblind":
            # context masking - subsequent pixels won't have access
            # to next pixels (spatial dim)
            if vertical:
                if mask == 'A':
                    # In the first layer, can ONLY access pixels above it
                    pre_mask[:, :, yc:, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    pre_mask[:, :, yc+1:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
            # same pixel masking - pixel won't access next color (conv filter dim)
            #def bmask(i_out, i_in):
            #    cout_idx = np.expand_dims(np.arange(Cout) % 3 == i_out, 1)
            #    cin_idx = np.expand_dims(np.arange(Cin) % 3 == i_in, 0)
            #    a1, a2 = np.broadcast_arrays(cout_idx, cin_idx)
            #    return a1 * a2

            #for j in range(3):
            #    pre_mask[bmask(j, j), yc, xc] = 0.0 if mask == 'A' else 1.0

            #pre_mask[bmask(0, 1), yc, xc] = 0.0
            #pre_mask[bmask(0, 2), yc, xc] = 0.0
            #pre_mask[bmask(1, 2), yc, xc] = 0.0
        elif mask_mode == "fig1-van-den-oord":
            if vertical:
                pre_mask[:, :, yc:, :] = 0.0
            else:
                # All rows after center must be zero
                pre_mask[:, :, yc+1:, :] = 0.0
                ### All rows before center must be zero # XXX: not actually necessary
                ##pre_mask[:, :, :yc, :] = 0.0
                # All columns after center in center row must be zero
                pre_mask[:, :, yc, xc+1:] = 0.0

            if mask == 'A':
                # Center must be zero in first layer
                pre_mask[:, :, yc, xc] = 0.0
        elif mask_mode == "turukin":
            pre_mask[:, :, yc+1:, :] = 0.0
            pre_mask[:, :, yc, xc+1:] = 0.0
            if mask == 'A':
                pre_mask[:, :, yc, xc] = 0.0

        print("%s %s MASKED CONV: %d x %d. Mask:" % (mask, "VERTICAL" if vertical else "HORIZONTAL", kh, kw))
        print(pre_mask[0, 0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution2D, self).forward(x)


class PixelCNNGatedLayer(nn.Module):
    def __init__(self, primary, in_channels, out_channels, filter_size,
            mask='B', nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=False,
            residual_horizontal=True, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        super().__init__()
        self.primary = primary
        if primary:
            assert mask == 'A'
            assert not residual_vertical
            assert not residual_horizontal
        else:
            assert mask == 'B'
        self.out_channels = out_channels
        self.gated = gated
        gm = 2 if gated else 1
        self.vertical_conv = MaskedConvolution2D(
            in_channels, gm * out_channels, (filter_size, filter_size),
            mask=mask, vertical=True, mask_mode=mask_mode, groups=groups)
        self.v_to_h_conv = nn.Conv2d(gm * out_channels, gm * out_channels, 1, groups=groups)

        self.horizontal_conv = MaskedConvolution2D(
            in_channels, gm * out_channels,
            (filter_size if horizontal_2d_convs else 1, filter_size), # XXX: traditionally (1, filter_size),
            mask=mask, vertical=False, mask_mode=mask_mode, groups=groups)

        self.residual_vertical = None
        if residual_vertical:
            self.residual_vertical = nn.Conv2d(in_channels, gm * out_channels, 1, groups=groups)

        self.horizontal_output = nn.Conv2d(out_channels, out_channels, 1, groups=groups)
        self.horizontal_skip = None
        if skips:
            self.horizontal_skip = nn.Conv2d(out_channels, out_channels, 1, groups=groups)
        self.conditional_vector = conditional_features is not None
        self.conditional_image = conditional_image_channels is not None
        if self.conditional_image:
            self.cond_conv_h = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False, groups=groups)
            self.cond_conv_v = nn.Conv2d(conditional_image_channels, gm * out_channels, 1, bias=False, groups=groups)
        if self.conditional_vector:
            self.cond_fc_h = nn.Linear(conditional_features, gm * out_channels, bias=False)
            self.cond_fc_v = nn.Linear(conditional_features, gm * out_channels, bias=False)
        self.residual_horizontal = residual_horizontal
        self.relu_out = relu_out

    @classmethod
    def primary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None,
            skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        return cls(True, in_channels, out_channels, filter_size, nobias=nobias,
                mask='A', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=False, residual_horizontal=False,
                skips=skips, gated=gated,
                relu_out=relu_out, horizontal_2d_convs=horizontal_2d_convs,
                mask_mode=mask_mode, groups=groups)

    @classmethod
    def secondary(cls, in_channels, out_channels, filter_size,
            nobias=False, conditional_features=None,
            conditional_image_channels=None, residual_vertical=True,
            residual_horizontal=True, skips=False, gated=True,
            relu_out=False, horizontal_2d_convs=False, mask_mode="noblind", groups=1):
        return cls(False, in_channels, out_channels, filter_size, nobias=nobias,
                mask='B', conditional_features=conditional_features,
                conditional_image_channels=conditional_image_channels,
                residual_vertical=residual_vertical, residual_horizontal=residual_horizontal,
                skips=skips, gated=gated, relu_out=relu_out,
                horizontal_2d_convs=horizontal_2d_convs, mask_mode=mask_mode, groups=groups)

    def _gate(self, x):
        if self.gated:
            return torch.tanh(x[:,:self.out_channels]) * torch.sigmoid(x[:,self.out_channels:])
        else:
            return x

    def __call__(self, v, h, conditional_image=None, conditional_vector=None):
        horizontal_preactivation = self.horizontal_conv(h) # 1xN
        vertical_preactivation = self.vertical_conv(v) # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation) # 1x1
        if self.residual_vertical is not None:
            vertical_preactivation = vertical_preactivation + self.residual_vertical(v) # 1x1 to residual
        horizontal_preactivation = horizontal_preactivation + v_to_h
        if self.conditional_image and conditional_image is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_conv_h(conditional_image)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_conv_v(conditional_image)
        if self.conditional_vector and conditional_vector is not None:
            horizontal_preactivation = horizontal_preactivation + \
                    self.cond_fc_h(conditional_vector).unsqueeze(-1).unsqueeze(-1)
            vertical_preactivation = vertical_preactivation + \
                    self.cond_fc_v(conditional_vector).unsqueeze(-1).unsqueeze(-1)
        v_out = self._gate(vertical_preactivation)
        h_activated = self._gate(horizontal_preactivation)
        h_skip = None
        if self.horizontal_skip is not None:
            h_skip = self.horizontal_skip(h_activated)
        h_preres = self.horizontal_output(h_activated)
        if self.residual_horizontal:
            h_out = h + h_preres
        else:
            h_out = h_preres
        if self.relu_out:
            v_out = F.relu(v_out)
            h_out = F.relu(h_out)
            if h_skip is not None:
                h_skip = F.relu(h_skip)
        return v_out, h_out, h_skip

class PixelCNNGatedStack(nn.Module):
    def __init__(self, *args):
        super().__init__()
        layers = list(args)
        for i, layer in enumerate(layers):
            assert isinstance(layer, PixelCNNGatedLayer)
            if i == 0:
                assert layer.primary
            else:
                assert not layer.primary
        self.layers = nn.ModuleList(layers)

    def __call__(self, v, h, skips=None, conditional_image=None, conditional_vector=None):
        if skips is None:
            skips = []
        else:
            skips = [skips]
        for layer in self.layers:
            v, h, skip = layer(v, h, conditional_image=conditional_image, conditional_vector=conditional_vector)
            if skip is not None:
                skips.append(skip)
        if len(skips) == 0:
            skips = None
        else:
            skips = torch.cat(skips, 1)
        return v, h, skips
