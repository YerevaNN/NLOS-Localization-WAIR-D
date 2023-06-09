import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class WAIRDDatasetFullSingleLink(Dataset):
    def __init__(self, data_path: str, scenario: str, split: str):
        super().__init__()

        if scenario != "scenario_1":
            raise NotImplementedError(f'There is only an implementation for scenario_1. Given {scenario}')

        if split != "train" and split != "val" and split != "test":
            raise NotImplementedError(f'There is only an implementation for train and test. Given {split}')

        self.__scenario_path: str = os.path.join(data_path, scenario)
        self.__split: str = split

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
        bs_location, ue_location = locations['bs'], locations['ue']

        toa = pair_data['toa']

        angles = pair_data['angles'].item()
        aod, aoa = angles['aod'], angles['aoa']

        shortest_toa_idx = np.argmin(toa)

        toa, aod, aoa = toa[shortest_toa_idx], aod[shortest_toa_idx, 1], aoa[shortest_toa_idx, 1]

        phi_diff = aod - aoa

        dis1 = toa * 0.3
        dis2 = ((ue_location[1] - bs_location[1]) ** 2 + (ue_location[0] - bs_location[0]) ** 2 + (
                2 * (ue_location[2] - bs_location[2])) ** 2) ** 0.5 * 0.5
        is_los = bool(np.round(phi_diff, 5) == np.round(np.pi, 5)
                      and np.round(dis1, 5) == np.round(dis2, 5))

        return (bs_location.astype(np.float32), ue_location.astype(np.float32),
                toa.astype(np.float32), aod.astype(np.float32), aoa.astype(np.float32), is_los,
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

    def __get_path_data(self, environment: str, bs_idx: int, ue_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_path = os.path.join(self.__scenario_path, environment, "Path.npy")
        env_path_data: dict[str, dict] = np.load(path_path, allow_pickle=True, encoding='latin1').item()

        pair_path_data: dict[str, np.ndarray] = env_path_data[f"bs{bs_idx}_ue{ue_idx}"]

        return pair_path_data["taud"].astype(np.float32), \
               pair_path_data["dod"].astype(np.float32), \
               pair_path_data["doa"].astype(np.float32)

    def __get_locations(self, environment: str, bs_idx: int, ue_idx: int) -> tuple[torch.Tensor, torch.Tensor,
                                                                                   dict[str, float]]:
        env_bs_locations_df, env_ue_locations_df, metadata = self.__get_env_locations_data(environment)

        bs_location = env_bs_locations_df.loc[env_bs_locations_df["name"] == f"bs{bs_idx}"]
        ue_location = env_ue_locations_df.loc[env_ue_locations_df["name"] == f"ue{ue_idx}"]

        return (torch.tensor([bs_location["x"].item(), bs_location["y"].item(), bs_location["z"].item()]),
                torch.tensor([ue_location["x"].item(), ue_location["y"].item(), ue_location["z"].item()]),
                metadata)

    def __get_env_locations_data(self, environment: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        info_path = os.path.join(self.__scenario_path, environment, "Info.npy")
        info = np.load(info_path, allow_pickle=True, encoding='latin1').astype(np.float32)

        img_width, img_height = info[:2]
        locs = info[2:].reshape(-1, 4)

        # padding (making map shape square)
        if img_width > img_height:
            locs[:, 1] += (img_width - img_height) / 2  # y coordinates
            locs[:, 3] += (img_width - img_height) / 2  # y coordinates
        else:
            locs[:, 0] += (img_height - img_width) / 2  # x coordinates
            locs[:, 2] += (img_height - img_width) / 2  # x coordinates

        img_size = max(img_width, img_height)

        # scale
        locs /= img_size

        # locs = info[2:].reshape(-1, 4)

        locations_df = pd.DataFrame(dict(bs_x=locs[:, 0],
                                         bs_y=locs[:, 1],
                                         bs_z=[6.0 / img_size for _ in range(len(locs[:, 1]))],
                                         ue_x=locs[:, 2],
                                         ue_y=locs[:, 3],
                                         ue_z=[1.5 / img_size for _ in range(len(locs[:, 1]))]))

        # bs_locations
        bs_locations_df = locations_df[["bs_x", "bs_y", "bs_z"]].drop_duplicates()
        bs_locations_df = bs_locations_df.rename(columns={"bs_x": "x", "bs_y": "y", "bs_z": "z"})
        bs_locations_df = bs_locations_df.reset_index(drop=True)
        bs_locations_df["name"] = list(map(lambda x: f"bs{x}", bs_locations_df.index))

        # ue_locations
        ue_locations_df = locations_df[["ue_x", "ue_y", "ue_z"]].drop_duplicates()
        ue_locations_df = ue_locations_df.rename(columns={"ue_x": "x", "ue_y": "y", "ue_z": "z"})
        ue_locations_df = ue_locations_df.reset_index(drop=True)
        ue_locations_df["name"] = list(map(lambda x: f"ue{x}", ue_locations_df.index))

        return bs_locations_df, ue_locations_df, {"img_size": img_size}

    def __prepare_environments(self) -> list[str]:
        environments = os.listdir(self.__scenario_path)
        environments = sorted(filter(lambda x: str.isnumeric(x), environments))

        if self.__split == "train":
            return environments[:900] + environments[1000: 9499]
        elif self.__split == "val":
            return environments[9499:]
        else:
            return environments[900:1000]
