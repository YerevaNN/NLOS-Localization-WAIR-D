import pytorch_lightning as pl

from torch.utils.data import DataLoader
from src.datamodules.datasets import WAIRDDatasetFullSingleLink


class WAIRDFullSingleLinkDatamodule(pl.LightningDataModule):
    def __init__(self, data_path: str, scenario: str, batch_size: int, num_workers: int, *args, **kwargs):
        super().__init__()
        self.__data_path = data_path
        self.__scenario = scenario
        self.__batch_size = batch_size
        self.__num_workers = num_workers

        self.__train_set = None
        self.__val_set = None
        self.__test_set = None

    def prepare_data(self) -> None:
        self.__train_set = WAIRDDatasetFullSingleLink(data_path=self.__data_path, scenario=self.__scenario, split="train")
        self.__val_set = WAIRDDatasetFullSingleLink(data_path=self.__data_path, scenario=self.__scenario, split="val")
        self.__test_set = WAIRDDatasetFullSingleLink(data_path=self.__data_path, scenario=self.__scenario, split="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.__train_set, batch_size=self.__batch_size, num_workers=self.__num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.__val_set, batch_size=self.__batch_size, num_workers=self.__num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.__test_set, batch_size=self.__batch_size, num_workers=self.__num_workers)
