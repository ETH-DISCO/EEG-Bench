from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    Zhou2016,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_zhou2016(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class Zhou2016MDataset(BaseBCIDataset):
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
            name="Zhou2016",
            interval=(0, 5),
            target_classes=target_classes,
            available_classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names=['Fp1', 'Fp2', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4', 'CP3', 'CPz', 'CP4', 'O1', 'Oz', 'O2'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Zhou2016MDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "labels_mapping": {
                "left_hand": 1,
                "right_hand": 2,
                "feet": 3,
            },
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        Zhou2016M = Zhou2016()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery(
                n_classes=3, events=["left_hand", "right_hand", "feet",]
            )
        elif set(self.target_classes) == set(
            [BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI]
        ):
            paradigm = MotorImagery(n_classes=2, events=["left_hand", "right_hand"])
        elif set(self.target_classes) == set([BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI]):
            paradigm = MotorImagery(n_classes=2, events=["right_hand", "feet"])
        else:
            raise ValueError("Invalid target classes")
        
        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
            return
        self.data, self.labels, _ = self.cache.cache(_load_data_zhou2016)(
            paradigm, Zhou2016M, self.subjects
        )  # type: ignore
