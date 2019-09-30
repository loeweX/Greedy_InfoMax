import torch
import torch.nn as nn

from GreedyInfoMax.utils import utils

class Autoregressor(nn.Module):
    def __init__(self, opt, input_size, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=True
        )

        self.opt = opt
        if self.opt.remove_BPTT:
            self.gru = nn.GRUCell(
                input_size=self.input_size, hidden_size=self.hidden_dim
            )

    def forward(self, input):

        cur_device = utils.get_device(self.opt, input)

        if self.opt.remove_BPTT:
            """
            For removing BPTT, we loop over the sequence manually and detach the hidden state 
            to restrict gradients to work only within the current time-step
            """

            input = input.permute(1, 0, 2)  # L, B, C

            regress_hidden_state = torch.zeros(
                input.size(1), self.hidden_dim, device=cur_device
            )
            output = torch.zeros(
                input.size(0), input.size(1), self.hidden_dim, device=cur_device
            )

            for i in range(len(input)):
                regress_hidden_state = self.gru(input[i], regress_hidden_state.detach())
                output[i] = regress_hidden_state

            output = output.permute(1, 0, 2)
        else:
            regress_hidden_state = torch.zeros(1, input.size(0), self.hidden_dim, device=cur_device)
            self.gru.flatten_parameters()
            output, regress_hidden_state = self.gru(input, regress_hidden_state)

        return output
