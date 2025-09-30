from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    BNCI2015_004,
)
from moabb.paradigms import MotorImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_scherer2015(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Scherer2015MDataset(BaseBCIDataset):
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
            name="Scherer2015", # aka MI II or BNCI 2015-004
            interval=(3, 10),
            target_classes=target_classes,
            available_classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=256,
            channel_names = ["AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3", "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Scherer2015MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "labels_mapping": {"right_hand": 4, "feet": 5},
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        Scherer2015 = BNCI2015_004()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery()
        elif self.target_classes == [BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI]:
            paradigm = MotorImagery(n_classes=2, events=["right_hand", "feet"])
        else:
            raise ValueError("Invalid target classes")
        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_scherer2015)(
            paradigm, Scherer2015, self.subjects
        )  # type: ignore
