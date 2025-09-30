from .base_bci_dataset import BaseBCIDataset
import warnings
from ...enums.bci_classes import BCIClasses
import moabb
from typing import Optional, Sequence
from moabb.paradigms.base import BaseParadigm
from moabb.datasets import (
    BNCI2014_001,
)
import moabb.datasets.base as base
from moabb.paradigms import MotorImagery
import logging
import numpy as np

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

"""
BCI IV 2a dataset aka BNCI2014_001
Four class motor imagery
Paper 1:    https://www.bbci.de/competition/iv/desc_2a.pdf
Paper 2:    https://lampx.tugraz.at/~bci/database/001-2014/description.pdf
Data 1:     https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit
Data 2:     https://bnci-horizon-2020.eu/database/data-sets/
"""

def _load_data_bcicomp_iv_2a(
    paradigm: BaseParadigm, dataset: base.BaseDataset, subjects: Sequence[int]
):
    return paradigm.get_data(dataset=dataset, subjects=subjects)


class BCICompIV2aMDataset(BaseBCIDataset):
    """
    - "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"
    """
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
            name="BCICompIV2a", # MI Limb
            interval=(2, 6),
            target_classes=target_classes,
            available_classes=[BCIClasses.LEFT_HAND_MI, BCIClasses.RIGHT_HAND_MI, BCIClasses.FEET_MI, BCIClasses.TONGUE_MI],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=250,
            channel_names=["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
            preload=preload,
        )
        # fmt: on
        logging.info("in BCICompIV2aMDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,  # check if correct or target frequency
            "channel_names": self._channel_names,  # check if correct or target channels
            "labels_mapping": {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            "name": self.name,
        }

        if preload:
            self.load_data()

    def load_data(self) -> None:
        BCI_IV_2a = BNCI2014_001()
        if self.target_classes is None:
            logging.warning("target_classes is None, loading all classes...")
            paradigm = MotorImagery(
                n_classes=4, events=["left_hand", "right_hand", "feet", "tongue"]
            )
        elif set(self.target_classes) == set(
            [
                BCIClasses.LEFT_HAND_MI,
                BCIClasses.RIGHT_HAND_MI,
                BCIClasses.FEET_MI,
                BCIClasses.TONGUE_MI,
            ]
        ):
            paradigm = MotorImagery(
                n_classes=4, events=["left_hand", "right_hand", "feet", "tongue"]
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
        else:
            self.data, self.labels, _ = self.cache.cache(_load_data_bcicomp_iv_2a)(
                paradigm, BCI_IV_2a, self.subjects
            )  # type: ignore
