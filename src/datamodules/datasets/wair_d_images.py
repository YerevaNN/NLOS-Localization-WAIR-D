import os
import logging

import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter

from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class WAIRDDatasetImages(Dataset):

    def __init__(self, data_path: str, scenario: str, split: str, output_kernel_size: int, use_channels: list[int]):
        super().__init__()

        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')

        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')

        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__split: str = split
        self.__output_kernel_size: int = output_kernel_size
        self.__use_channels = use_channels

        self.__num_envs = 10000 - 1

        self.__num_train_envs_count = 10000 - 500 - 100 - 1
        self.__num_val_envs_count = 500
        self.__num_test_envs_count = 100

        self.__num_bss_per_env = 5
        self.__num_ues_per_env = 30
        self.__num_pairs_per_env = self.__num_bss_per_env * self.__num_ues_per_env

        self.__environments: list[str] = self.__prepare_environments()

    def __getitem__(self, pair_idx: int):
        environment_idx: int = pair_idx // self.__num_pairs_per_env

        local_pair_idx = pair_idx % self.__num_pairs_per_env
        bs_idx = local_pair_idx // self.__num_ues_per_env
        ue_idx = local_pair_idx % self.__num_ues_per_env

        environment: str = self.__environments[environment_idx]

        env_path = os.path.join(self.__scenario_path, environment)
        metadata = dict(np.load(os.path.join(env_path, "metadata.npz")))

        pair_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}")
        pair_data_path = os.path.join(pair_path, "data.npz")
        pair_data = dict(np.load(pair_data_path, allow_pickle=True))

        input_img = list(np.load(os.path.join(pair_path, "input_img.npz")).values())[0].astype(np.float32)
        ue_loc_img = list(np.load(os.path.join(pair_path, "ue_loc_img.npz")).values())[0].astype(np.float32)

        locations = pair_data["locations"].item()
        ue_location = locations['ue'].astype(np.float32)[:2].astype(np.float32) * max(input_img.shape)

        input_img = resize(input_img, (3, 224, 224))
        ue_loc_img = resize(ue_loc_img, (1, 224, 224))

        ue_loc_img = gaussian_filter(ue_loc_img, self.__output_kernel_size)
        ue_loc_img_max = ue_loc_img.max()

        if ue_loc_img_max != 0:
            ue_loc_img /= ue_loc_img_max

        if self.__use_channels is not None:
            input_img = input_img[self.__use_channels]

        return (input_img,
                ue_loc_img,
                ue_location,
                metadata["img_size"], pair_data["is_los"])

    def __len__(self):
        env_count = None
        if self.__split == "train":
            env_count = self.__num_train_envs_count
        elif self.__split == "val":
            env_count = self.__num_val_envs_count
        else:
            env_count = self.__num_test_envs_count

        return env_count * self.__num_bss_per_env * self.__num_ues_per_env

    def __prepare_environments(self) -> list[str]:
        environments = os.listdir(self.__scenario_path)
        environments = sorted(filter(lambda x: str.isnumeric(x), environments))

        if self.__split == "train":
            return environments[:900] + environments[1000: 9499]
        elif self.__split == "val":
            return environments[9499:]
        else:
            return environments[900:1000]
