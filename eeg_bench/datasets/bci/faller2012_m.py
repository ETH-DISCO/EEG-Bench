from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    BNCI2015_001,
)
from moabb.paradigms import MotorImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_faller2012(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Faller2012MDataset(BaseBCIDataset):
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
            name="Faller2012", # aka BNCI2015_001
            interval=(0, 5),
            target_classes=target_classes,
            available_classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=512,
            channel_names = ["FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4",],
            preload=preload,
        )
        # fmt: on
        logging.info("in Faller2012MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "labels_mapping": {"right_hand": 1, "feet": 2},
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        Faller2012 = BNCI2015_001()
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
        self.data, self.labels, _ = self.cache.cache(_load_data_faller2012)(
            paradigm, Faller2012, self.subjects
        )  # type: ignore
