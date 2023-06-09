import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class WAIRDDatasetFullAllLinks(Dataset):
    def __init__(self, data_path: str, scenario: str, split: str, n_links: int):
        super().__init__()

        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')

        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')

        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__split: str = split
        self.__n_links = n_links

        self.__num_envs = 10000

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

        pair_data_path = os.path.join(env_path, f"bs{bs_idx}_ue{ue_idx}", "data.npz")
        pair_data = dict(np.load(pair_data_path, allow_pickle=True))

        locations = pair_data["locations"].item()
        bs_location, ue_location = locations['bs'].astype(np.float32), locations['ue'].astype(np.float32)

        toa = pair_data['toa'].astype(np.float32)

        angles = pair_data['angles'].item()
        aod, aoa = angles['aod'].astype(np.float32), angles['aoa'].astype(np.float32)

        path_responses = pair_data['path_responses'].item()
        path_responses = np.stack([x for x in path_responses.values()], axis=1).astype(np.float32)

        sorted_toa_indices = np.argsort(toa)

        shortest_toa_idx = sorted_toa_indices[0]

        toa_shortest = toa[shortest_toa_idx]

        phi_diff = aod[shortest_toa_idx, 1] - aoa[shortest_toa_idx, 1]

        dis1 = toa_shortest * 0.3
        dis2 = ((ue_location[1] - bs_location[1]) ** 2 + (ue_location[0] - bs_location[0]) ** 2 + (
                2 * (ue_location[2] - bs_location[2])) ** 2) ** 0.5 * 0.5
        is_los = bool(np.round(phi_diff, 5) == np.round(np.pi, 5).astype(np.float32)
                      and np.round(dis1, 5) == np.round(dis2, 5))

        bs_location = bs_location
        ue_location = ue_location
        toa = toa[sorted_toa_indices[:self.__n_links]]
        aod = aod[sorted_toa_indices[:self.__n_links]]
        aoa = aoa[sorted_toa_indices[:self.__n_links]]
        path_responses = path_responses[sorted_toa_indices[:self.__n_links]]

        return (bs_location, ue_location,
                toa, aod, aoa, is_los, path_responses,
                metadata["img_size"] / 2,
                {"path": os.path.join(self.__scenario_path, environment)})

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
