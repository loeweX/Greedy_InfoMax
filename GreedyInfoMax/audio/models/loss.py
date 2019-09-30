from abc import abstractmethod
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    @abstractmethod
    def get_loss(self, x, z, c, filename, start_idx):
        pass
