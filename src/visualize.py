import logging
import hydra
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
import torch

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from src.networks import WAIRDNonML
from skimage import io
from skimage.transform import rescale

from src.utils import pad_to_square

log = logging.getLogger(__name__)


def visualize_pair(bs_location,
                   ue_location,
                   toa,
                   aod,
                   aoa,
                   is_los,
                   image_size,
                   path,
                   # ue_location_pred,
                   ue_location_pred_non_ml,
                   d_pred_non_ml):
    img = io.imread(os.path.join(path, "environment.png"))
    img = pad_to_square(img)
    img = rescale(img, image_size / max(img.shape), multichannel=True)

    fig = px.imshow(img[::-1])

    # Add traces
    fig.add_trace(go.Scatter(x=[bs_location[0]], y=[bs_location[1]],
                             mode='markers', marker=dict(size=5, color="blue"), name="BS location"))
    fig.add_trace(go.Scatter(x=[ue_location[0]], y=[ue_location[1]],
                             mode='markers', marker=dict(size=5, color="orange"), name="UE location"))

    # fig.add_trace(go.Scatter(x=[ue_location_pred[0]], y=[ue_location_pred[1]],
    #                          mode='markers', marker=dict(size=5, color="red"), name="UE location pred"))

    fig.add_trace(go.Scatter(x=[ue_location_pred_non_ml[0]], y=[ue_location_pred_non_ml[1]],
                             mode='markers', marker=dict(size=5, color="green"),
                             name="UE location pred (non ML)"))

    fig.add_shape(type="circle",
                  xref="x", yref="y",
                  x0=bs_location[0] - d_pred_non_ml,
                  y0=bs_location[1] - d_pred_non_ml,
                  x1=bs_location[0] + d_pred_non_ml,
                  y1=bs_location[1] + d_pred_non_ml,
                  line_color="green",
                  line_dash="dashdot",
                  line_width=1,
                  )

    fig.add_annotation(
        y=bs_location[1] + (d_pred_non_ml / 5) * torch.sin(aod),
        x=bs_location[0] + (d_pred_non_ml / 5) * torch.cos(aod),
        ay=bs_location[1],
        ax=bs_location[0],
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='blue',
    )

    fig.add_annotation(
        y=ue_location[1],
        x=ue_location[0],
        ay=ue_location[1] + (d_pred_non_ml / 5) * torch.sin(aoa),
        ax=ue_location[0] + (d_pred_non_ml / 5) * torch.cos(aoa),
        xref='x',
        yref='y',
        axref='x',
        ayref='y',
        text='',
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=1,
        arrowcolor='orange'
    )

    # fig.update_yaxes(autorange="reversed")

    fig.update_layout(title=("LOS" if is_los else "NLOS") + f" (Path delay = {toa:.4f})",
                      xaxis_range=[0, image_size], yaxis_range=[0, image_size])


    st.plotly_chart(fig)


def visualize(config: DictConfig) -> None:
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
    non_ml = WAIRDNonML()

    for batch in datamodule.val_dataloader():
        bs_location_, ue_location_, toa_, aod_, aoa_, is_los_, image_size_, meta_ = batch

        image_size_ /= 2
        ue_location_pred_non_ml_, d_pred_non_ml_ = non_ml(bs_location_, [aod_, aoa_], toa_)

        ue_location_pred_ = None #algorithm(bs_location_, [aod_, aoa_], toa_).to('cpu').detach()

        bs_location_ = bs_location_[:, :2]
        ue_location_ = ue_location_[:, :2]

        bs_location_ *= image_size_[:, None]
        ue_location_ *= image_size_[:, None]
        # ue_location_pred_ *= image_size_[:, None]
        ue_location_pred_non_ml_ *= image_size_[:, None]
        d_pred_non_ml_ *= image_size_

        for (bs_location, ue_location, toa, aod, aoa, is_los,
            image_size, path,
            # ue_location_pred,
            ue_location_pred_non_ml, d_pred_non_ml) in \
                zip(bs_location_, ue_location_, toa_, aod_, aoa_, is_los_,
                    image_size_, meta_["path"],
                    # ue_location_pred_,
                    ue_location_pred_non_ml_, d_pred_non_ml_):
                visualize_pair(bs_location, ue_location, toa, aod, aoa, is_los, image_size,
                               path,
                               # ue_location_pred,
                               ue_location_pred_non_ml, d_pred_non_ml)
        break
