import torch.nn as nn
import torch

from GreedyInfoMax.audio.data import phone_dict
from GreedyInfoMax.audio.models import loss


class Phones_Loss(loss.Loss):
    def __init__(self, opt, hidden_dim, calc_accuracy):
        super(Phones_Loss, self).__init__()

        self.opt = opt

        self.phone_dict = phone_dict.load_phone_dict(opt)
        self.hidden_dim = hidden_dim
        self.calc_accuracy = calc_accuracy

        # create linear classifier
        self.linear_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 41)
        ).to(self.opt.device)  # 41 different phones to differentiate

        self.phones_loss = nn.CrossEntropyLoss()

        self.label_num = 128


    def get_loss(self, x, z, c, filename, start_idx):
        total_loss, accuracies = self.calc_supervised_phones_loss(
            c, filename, start_idx
        )
        return total_loss, accuracies

    def calc_supervised_phones_loss(self, c, filename, start_idx):
        """
        Calculates the loss for fully supervised training using the provided phones labels.
        Since there are labels for every 10ms of input, we need to downscale the output of
        the trained layer to 128 values first, which is done by maxpooling.
        :param c: output of the layer to be trained
        :param filename: filenames of the current files in the batch
        :param start_idx: idx within the audio-files for the current files in the batch
        :return: loss and accuracy
        """

        targets = torch.zeros(self.opt.batch_size, self.label_num ).long()
        for idx, cur_audio_idx in enumerate(start_idx):
            targets[idx, :] = torch.LongTensor(
                self.phone_dict[filename[idx]][
                    (cur_audio_idx - 80) // 160 : (cur_audio_idx - 80 + 20480) / 160
                ]
            )

        targets = targets.to(self.opt.device).reshape(-1)

        # forward pass
        c = c.permute(0, 2, 1)

        pooled_c = nn.functional.adaptive_avg_pool1d(c, self.label_num)
        pooled_c = pooled_c.permute(0, 2, 1).reshape(-1, self.hidden_dim)

        phones_out = self.linear_classifier(pooled_c)

        loss = self.phones_loss(phones_out, targets)

        accuracy = torch.zeros(1)
        # calculate accuracy
        if self.calc_accuracy:
            _, predicted = torch.max(phones_out.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            accuracy[0] = correct / total

        return loss, accuracy
