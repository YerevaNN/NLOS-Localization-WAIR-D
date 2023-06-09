import logging
import hydra
import streamlit as st
import numpy as np
import torch
import plotly.express as px
import torch.nn as nn

from tqdm import tqdm
from PIL import Image, ImageFilter
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from scipy.special import softmax, expit

from omegaconf import DictConfig

log = logging.getLogger(__name__)


def unet_visualize(config: DictConfig) -> None:
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating algorithm {config.algorithm._target_} with checkpoint {config.checkpoint_path}")

    algorithm: LightningModule = hydra.utils.get_class(config.algorithm._target_) \
        .load_from_checkpoint(config.checkpoint_path, **config.algorithm,
                              network_conf=(OmegaConf.to_yaml(
                                  config.network) if "network" in config else None),
                              optimizer_conf=(OmegaConf.to_yaml(
                                  config.optimizer) if "optimizer" in config else None),
                              scheduler_conf=(OmegaConf.to_yaml(
                                  config.scheduler) if "scheduler" in config else None)
                              )

    algorithm.network.eval()

    datamodule.prepare_data()

    i = st.number_input("Example", value=0)

    batch = datamodule.test_set[i]

    input_image, supervision_image, ue_location, image_size, is_los = batch

    out = algorithm.network(torch.Tensor([input_image]))

    vv = (input_image.transpose((1, 2, 0)) * 255).astype(np.uint8)

    max_ind = out.flatten(1).argmax(dim=-1)
    ue_location_pred = torch.stack([max_ind % max(out.shape), max_ind // max(out.shape)], dim=1)  # .cuda()
    res = np.zeros_like((out.detach().numpy()[0][0] * 255).T.astype(np.uint8))
    res[ue_location_pred[0][1]][ue_location_pred[0][0]] = 255

    st.image((np.stack([input_image[0]]*3, axis=2)* 255).astype(np.uint8))
    st.image(supervision_image[0])

    vv = np.asarray(Image.fromarray(vv).filter(ImageFilter.MaxFilter(3)))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(vv)

    with col2:
        xk = supervision_image[0].copy()
        xk = np.asarray(Image.fromarray(xk).filter(ImageFilter.MaxFilter(3))) / xk.max()
        x = res.copy()
        x = np.asarray(Image.fromarray(x).filter(ImageFilter.MaxFilter(3))) / 255

        # st.write(xk.max(), x.max())
        st.image(np.stack([x, xk, np.zeros_like(x)], axis=2))

    st.write(f"Los: {is_los}")
    # col1, col2, col3 = st.columns(3)
    #
    # with col1:
    #     st.write("UE position")
    #     xk = supervision_image[0].copy()
    #     xk = np.asarray(Image.fromarray(x).filter(ImageFilter.MaxFilter(3)))
        # st.image((xk * 255).astype(np.uint8))
    # with col2:
    #     st.write("prediction")
    #
    #     out = out.detach().numpy()[0][0]
    #     l = out.max() - out.min()
    #
    #     st.image((out - out.min()) / l)
    #     # st.image((out / 255).astype(np.uint))
    #     # st.image(1-out.astype(np.uint))
    #     st.image((out > out.mean()) * 255)
    #
    # with col3:
    #     st.write("UE pred")
    #     x = res.copy()
    #     x = np.asarray(Image.fromarray(x).filter(ImageFilter.MaxFilter(3)))
    #     st.image(x)

    out = out.detach().numpy()[0][0]
    x = supervision_image[0].copy() == supervision_image[0].max()
    xk = np.asarray(Image.fromarray(x).filter(ImageFilter.MaxFilter(3))) > 0
    vv = expit(out)
    vv *= (1 - xk)
    vv += xk * np.max(vv) * 2
    st.plotly_chart(px.imshow(vv, color_continuous_scale="Blackbody"))
