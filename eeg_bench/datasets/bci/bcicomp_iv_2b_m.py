from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
from typing import Optional, Sequence
import moabb
from moabb.datasets import (
    BNCI2014_004,
)
from moabb.paradigms import LeftRightImagery
import moabb.datasets.base as base
from moabb.paradigms.base import BaseParadigm
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def _load_data_bcicomp_iv_2b(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class BCICompIV2bMDataset(BaseBCIDataset):
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
            name="BCICompIV2b",
            interval=(3, 7.5),
            target_classes=target_classes,
            available_classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names = ["C3", "Cz", "C4"],
            preload=preload,
        )
        # fmt: on
        logging.info("in BCICompIV2bMDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2},
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        BCI_IV_2b = BNCI2014_004()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
        elif self.target_classes == [BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI]:
            paradigm = LeftRightImagery()
        else:
            raise ValueError("Invalid target classes")
        
        if (self.subjects is None) or (len(self.subjects) == 0):
            self.data = np.array([])
            self.labels = np.array([])
        else:
            self.data, self.labels, _ = self.cache.cache(_load_data_bcicomp_iv_2b)(
            paradigm, BCI_IV_2b, self.subjects
        )  # type: ignore
