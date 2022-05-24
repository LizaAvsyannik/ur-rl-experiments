import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, act_fn=nn.ReLU, n_layers=4, n_inner=-1):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.act_fn = act_fn
        if n_inner == -1:
            n_inner = n_out
        self.block = nn.Sequential(nn.Linear(n_in, n_inner),
                                   act_fn(),
                                   *[nn.Sequential(nn.Linear(n_inner, n_inner),
                                                   act_fn())
                                     for _ in range(max(0, n_layers - 2))],
                                   nn.Linear(n_inner, n_out))

    def forward(self, inputs):
        return self.block(inputs)
