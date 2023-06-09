import os
import logging
import numpy as np
import pandas as pd
import shutil

from tqdm import tqdm
from omegaconf import DictConfig
from multiprocessing import Pool
from collections import ChainMap
from skimage import io
from skimage.transform import resize
from PIL import Image, ImageDraw
from src.utils import pad_to_square

log = logging.getLogger(__name__)


def __get_env_locations_data(env_path: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    info_path = os.path.join(env_path, "Info.npy")
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


def __get_env_pairs_data(env_path: str) -> dict[str, dict]:
    return np.load(os.path.join(env_path, "Path.npy"), allow_pickle=True, encoding='latin1').item()


def __get_env_path_responses_data(env_path: str):
    env_path_responses_data_files = {
        "2.6GHz": "H_2_6_G.npy",
        "6GHz": "H_6_0_G.npy",
        "28GHz": "H_28_0_G.npy",
        "60GHz": "H_60_0_G.npy",
        "100GHz": "H_100_0_G.npy"
    }

    env_path_responses_data_paths = {k: os.path.join(env_path, v) for k, v in
                                     env_path_responses_data_files.items()}

    env_path_responses_data = {k: np.load(v, allow_pickle=True, encoding='latin1').item() for k, v in
                               env_path_responses_data_paths.items()}

    return env_path_responses_data


def __check_los(pair_data):
    toa = pair_data['toa']
    aoa = pair_data['angles']['aoa']
    aod = pair_data['angles']['aod']

    bs_location = pair_data['locations']['bs']
    ue_location = pair_data['locations']['ue']

    # is_los
    sorted_toa_indices = np.argsort(toa)

    shortest_toa_idx = sorted_toa_indices[0]

    toa_shortest = toa[shortest_toa_idx]

    phi_diff = aod[shortest_toa_idx, 1] - aoa[shortest_toa_idx, 1]

    dis1 = toa_shortest * 0.3
    dis2 = ((ue_location[1] - bs_location[1]) ** 2 + (ue_location[0] - bs_location[0]) ** 2 + (
            2 * (ue_location[2] - bs_location[2])) ** 2) ** 0.5 * 0.5
    is_los = bool(np.round(phi_diff, 5) == np.round(np.pi, 5)
                  and np.round(dis1, 5) == np.round(dis2, 5))

    return is_los


def __get_images(pair_data, input_size, image_path):
    toa = pair_data['toa']
    aoa = pair_data['angles']['aoa']
    aod = pair_data['angles']['aod']

    bs_location = (pair_data['locations']['bs'][:2] * input_size).astype(np.int32)
    ue_location = (pair_data['locations']['ue'][:2] * input_size).astype(np.int32)

    img = io.imread(image_path, as_gray=True)
    img = pad_to_square(img < 0.9)[::-1]
    img = resize(img, (input_size, input_size))

    # aod_channel
    aod_channel = Image.new('F', (input_size, input_size), 0)
    draw = ImageDraw.Draw(aod_channel)

    for i in range(len(toa)):
        draw.line((bs_location[0],
                   bs_location[1],
                   int(bs_location[0] + input_size * np.cos(aod[i][1])),
                   int(bs_location[1] + input_size * np.sin(aod[i][1])),
                   ), fill=1 - toa[i])

    aod_channel_np = np.asarray(aod_channel)

    # aoa_channel
    aoa_channel = Image.new('F', (input_size, input_size), 0)

    draw = ImageDraw.Draw(aoa_channel)

    for i in range(len(toa)):
        draw.line((bs_location[0],
                   bs_location[1],
                   int(bs_location[0] - input_size * np.cos(aoa[i][1])),
                   int(bs_location[1] - input_size * np.sin(aoa[i][1])),
                   ), fill=1 - toa[i])

    aoa_channel_np = np.asarray(aoa_channel)

    # ue_loc_channel
    ue_loc_channel = Image.new('F', (input_size, input_size), 0)
    try:
        ue_loc_channel.putpixel(ue_location, 1)
    except:
        print(ue_loc_channel.size, ue_location, pair_data['locations']['ue'])

    ue_loc_channel = np.asarray(ue_loc_channel)

    # input
    input_img = np.stack([img, aod_channel_np, aoa_channel_np], axis=0)
    ue_loc_img = np.stack([ue_loc_channel], axis=0)

    return input_img, ue_loc_img


def prepare_env_data(args):
    env, raw_env_path, prepared_env_path, image_size = args

    if raw_env_path.endswith("9117"):
        return {}

    os.makedirs(prepared_env_path, exist_ok=True)
    shutil.copyfile(os.path.join(raw_env_path, "environment.png"),
                    os.path.join(prepared_env_path, "environment.png"))

    env_bs_locations_df, env_ue_locations_df, metadata = __get_env_locations_data(raw_env_path)
    env_pairs_data = __get_env_pairs_data(raw_env_path)

    env_path_responses_data = __get_env_path_responses_data(raw_env_path)

    np.savez_compressed(os.path.join(prepared_env_path, "metadata.npz"), **metadata)

    prepared_pairs_data = dict()
    for bs_idx in range(5):
        for ue_idx in range(30):
            pair_key = f'bs{bs_idx}_ue{ue_idx}'
            prepared_pair_path = os.path.join(prepared_env_path, pair_key)
            if os.path.exists(prepared_pair_path):
                shutil.rmtree(prepared_pair_path)

            os.makedirs(prepared_pair_path, exist_ok=True)

            if os.path.exists(os.path.join(prepared_pair_path, 'ue_loc_img.npz')):
                continue

            bs_location = env_bs_locations_df[env_bs_locations_df["name"] == f"bs{bs_idx}"] \
                [['x', 'y', 'z']].iloc[0].values

            ue_location = env_ue_locations_df[env_ue_locations_df["name"] == f"ue{ue_idx}"] \
                [['x', 'y', 'z']].iloc[0].values

            pair_data = env_pairs_data[pair_key]
            path_responses_data = {freq: freq_data[pair_key] for freq, freq_data in env_path_responses_data.items()}

            prepared_pair_data = dict(
                locations=dict(bs=bs_location, ue=ue_location),
                toa=pair_data['taud'] / metadata['img_size'],
                angles=dict(aod=pair_data['dod'], aoa=pair_data['doa']),
                path_responses=path_responses_data
            )

            prepared_pair_data["is_los"] = __check_los(prepared_pair_data)

            np.savez_compressed(os.path.join(prepared_pair_path, 'data.npz'), **prepared_pair_data)

            input_img, ue_loc_img = __get_images(prepared_pair_data, input_size=image_size,
                                                 image_path=os.path.join(raw_env_path, "environment.png"))

            np.savez_compressed(os.path.join(prepared_pair_path, 'input_img.npz'), input_img)
            np.savez_compressed(os.path.join(prepared_pair_path, 'ue_loc_img.npz'), ue_loc_img)

            prepared_pairs_data[pair_key] = prepared_pair_data

    np.savez_compressed(os.path.join(prepared_env_path, "pairs_data.npz"), **metadata)
    return {env: dict(pairs_data=prepared_pairs_data, metadata=metadata)}


def prepare_data(config: DictConfig) -> None:
    args = []
    for env in tqdm(sorted(os.listdir(config.raw_data_dir))):
        raw_env_path = os.path.join(config.raw_data_dir, env)
        prepared_env_path = os.path.join(config.prepared_data_dir, env)

        args.append((env, raw_env_path, prepared_env_path, config.image_size))

    with Pool(config.n_processes) as p:
        res = list(tqdm(p.imap(prepare_env_data, args), total=len(args)))

    data = dict(ChainMap(*res))

    np.savez_compressed(os.path.join(config.prepared_data_dir, "data.npz"), **data)
