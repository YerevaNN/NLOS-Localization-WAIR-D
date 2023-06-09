import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, p, a, b):
        super().__init__()

        layers = []
        for i in range(3 * a):
            in_channels = p if i == 0 else 32 * b
            layers += [
                nn.Conv1d(in_channels=in_channels, out_channels=32 * b, kernel_size=p, padding='same'),
                nn.ReLU(),
                nn.BatchNorm1d(32 * b)
            ]

        self.__block1 = nn.Sequential(*layers)

        layers = []
        for i in range(3 * a):
            in_channels = 1 if i == 0 else 32 * b

            layers += [
                nn.Conv2d(in_channels=in_channels, out_channels=32 * b, kernel_size=(3, 3), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(32 * b)
            ]

        self.__block2 = nn.Sequential(*layers)

        layers = [
            # nn.AvgPool2d(kernel_size=(32 * b)),
            nn.Linear(32 * b * p, 2),
            nn.Sigmoid(),
        ]

        self.__block3 = nn.Sequential(*layers)

    def forward(self, bs_location: torch.Tensor, angles: list[torch.Tensor], toa: torch.Tensor):
        x = torch.cat([toa[:, :, None], bs_location[:, :, :2], *angles], dim=2)

        x = torch.permute(x, (0, 2, 1))

        x = x[:, None, :, :]

        x = self.__block2(x)

        x = torch.mean(x, dim=-1)
        x = torch.flatten(x, start_dim=1)

        out = self.__block3(x)

        return out
