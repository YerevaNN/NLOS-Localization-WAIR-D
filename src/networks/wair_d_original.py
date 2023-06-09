import torch
import torch.nn as nn


class WAIRDOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.__mlp = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )

    def forward(self, bs_location: torch.Tensor, angles: list[torch.Tensor], toa: torch.Tensor):

        x = torch.cat([toa[:, None],
                       bs_location[:, :2],
                       angles[0][:, None],
                       angles[1][:, None],
                       ], dim=1)

        out = self.__mlp(x)
        return out
