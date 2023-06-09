import logging
import warnings
import hydra
import random
import torch

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from dotenv import load_dotenv


log = logging.getLogger(__name__)

load_dotenv('.env')

hydra.core.global_hydra.GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(config: DictConfig) -> None:
    from src import utils, train, visualize, prepare_data, test_cnn, unet_visualize

    if config.seed == -1:
        config.seed = random.randint(0, 10 ** 8)

    seed_everything(config.seed)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    if config.get("print_config"):
        utils.print_config(config, fields=tuple(config.keys()), resolve=True)

    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if config.name == "prepare_data":
        return prepare_data(config)

    if config.name == "train":
        return train(config)

    if config.name == "visualize":
        return visualize(config)

    if config.name == "test_cnn":
        return test_cnn(config)

    if config.name == "unet_visualize":
        return unet_visualize(config)


if __name__ == '__main__':
    main()
