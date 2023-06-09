import torch
import torch.nn as nn


class WAIRDNonML:
    def __call__(self, bs_location: torch.Tensor, angles: list[torch.Tensor], toa: torch.Tensor):
        d = torch.sqrt(0.36 * toa ** 2 - 2 * (bs_location[:, 2] * 0.75))

        d = torch.where(torch.isnan(d), 0.6 * toa, d)  # in wair_d bs_z = 6, ue_z=1.5, bs_z - ue_z = 0.75*bs_z

        us_location = torch.zeros_like(bs_location)
        us_location[:, 0] = bs_location[:, 0] + d * torch.cos(angles[0])
        us_location[:, 1] = bs_location[:, 1] + d * torch.sin(angles[0])
        us_location[:, 2] = bs_location[:, 2] * 0.25

        return us_location, d
