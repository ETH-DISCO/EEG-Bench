from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    Cho2017,
)
import moabb.datasets.base as base
from moabb.paradigms import LeftRightImagery
import numpy as np
import logging

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_cho2017(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Cho2017MDataset(BaseBCIDataset):
    def __init__(
        self,
        target_classes: Sequence[BCIClasses],
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = True,
    ):
        # fmt: off
        super().__init__(
            name="Cho2017", # aka MI_LR 
            interval=(0, 3),
            target_classes=target_classes,
            available_classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],  
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=512,
            channel_names=["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Cho2017MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "labels_mapping": {
                "left_hand": 1,
                "right_hand": 2,
            },
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        Cho2017M = Cho2017()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = LeftRightImagery()
        elif set(self.target_classes) == set(
            [BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI]
        ):
            paradigm = LeftRightImagery()
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_cho2017)(
            paradigm, Cho2017M, self.subjects
        )  # type: ignore
