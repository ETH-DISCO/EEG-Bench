from .base_bci_dataset import BaseBCIDataset
import warnings
import logging
from ...enums.bci_classes import BCIClasses
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    AlexMI,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

"""
Data: https://zenodo.org/record/806023/files/
"""

def _load_data_barachant2012(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Barachant2012MDataset(BaseBCIDataset):
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
            name="Barachant2012", # AlexMI
            interval=(0, 3),
            target_classes=target_classes,
            available_classes=[BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],  
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=512,
            channel_names=["Fpz", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8"],
            preload=preload,
        )
        # fmt: on
        logging.info("in Barachant2012MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        Barachant2012 = AlexMI()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery(n_classes=2, events=["right_hand", "feet"])
        elif set(self.target_classes) == set([BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI]):
            paradigm = MotorImagery(n_classes=2, events=["right_hand", "feet"])
        else:
            raise ValueError("Invalid target classes")

        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_barachant2012)(
            paradigm, Barachant2012, self.subjects
        )  # type: ignore
