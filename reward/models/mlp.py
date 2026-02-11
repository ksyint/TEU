import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_dims=None, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        if hidden_dims is None:
            hidden_dims = [1024, 128, 64, 16]

        layers = []
        in_dim = input_size
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            if i < len(hidden_dims) - 2:
                layers.append(nn.Dropout(dropout))
            elif i == len(hidden_dims) - 2:
                layers.append(nn.Dropout(dropout * 0.5))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, x):
        return self.layers(x)
