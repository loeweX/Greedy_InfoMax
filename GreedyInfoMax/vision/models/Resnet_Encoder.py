import torch.nn as nn
import torch.nn.functional as F
import torch

from GreedyInfoMax.vision.models import InfoNCE_Loss, Supervised_Loss
from GreedyInfoMax.utils import model_utils


class PreActBlockNoBN(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockNoBN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneckNoBN(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckNoBN, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride
                )
            )

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))
        out = self.conv3(F.relu(out))
        out += shortcut
        return out


class ResNet_Encoder(nn.Module):
    def __init__(
        self,
        opt,
        block,
        num_blocks,
        filter,
        encoder_num,
        patch_size=16,
        input_dims=3,
        calc_loss=False,
    ):
        super(ResNet_Encoder, self).__init__()
        self.encoder_num = encoder_num
        self.opt = opt

        self.patchify = True
        self.overlap = 2

        self.calc_loss = calc_loss
        self.patch_size = patch_size
        self.filter = filter

        self.model = nn.Sequential()

        if encoder_num == 0:
            self.model.add_module(
                "Conv1",
                nn.Conv2d(
                    input_dims, self.filter[0], kernel_size=5, stride=1, padding=2
                ),
            )
            self.in_planes = self.filter[0]
            self.first_stride = 1
        elif encoder_num > 2:
            self.in_planes = self.filter[0] * block.expansion
            self.first_stride = 2
        else:
            self.in_planes = (self.filter[0] // 2) * block.expansion
            self.first_stride = 2

        for idx in range(len(num_blocks)):
            self.model.add_module(
                "layer {}".format((idx)),
                self._make_layer(
                    block, self.filter[idx], num_blocks[idx], stride=self.first_stride
                ),
            )
            self.first_stride = 2

        ## loss module is always present, but only gets used when training GreedyInfoMax modules
        if self.opt.loss == 0:
            self.loss = InfoNCE_Loss.InfoNCE_Loss(
                opt,
                in_channels=self.in_planes,
                out_channels=self.in_planes
            )
        elif self.opt.loss == 1:
            self.loss = Supervised_Loss.Supervised_Loss(opt, self.in_planes, True)
        else:
            raise Exception("Invalid option")

        if self.opt.weight_init:
            self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                model_utils.makeDeltaOrthogonal(
                    m.weight, nn.init.calculate_gain("relu")
                )
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                m.momentum = 0.3

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, n_patches_x, n_patches_y, label, patchify_right_now=True):
        if self.patchify and self.encoder_num == 0 and patchify_right_now:
            x = (
                x.unfold(2, self.patch_size, self.patch_size // self.overlap)
                .unfold(3, self.patch_size, self.patch_size // self.overlap)
                .permute(0, 2, 3, 1, 4, 5)
            )
            n_patches_x = x.shape[1]
            n_patches_y = x.shape[2]
            x = x.reshape(
                x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
            )

        z = self.model(x)

        out = F.adaptive_avg_pool2d(z, 1)
        out = out.reshape(-1, n_patches_x, n_patches_y, out.shape[1])
        out = out.permute(0, 3, 1, 2)

        accuracy = torch.zeros(1)
        if self.calc_loss and self.opt.loss == 0:
            loss = self.loss(out, out)
        elif self.calc_loss and self.opt.loss == 1:
            loss, accuracy = self.loss(out, label)
        else:
            loss = None

        return out, z, loss, accuracy, n_patches_x, n_patches_y

