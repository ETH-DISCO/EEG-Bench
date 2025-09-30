from abc import ABC, abstractmethod
import os
from joblib import Memory
from ..enums.bci_classes import BCIClasses
from ..enums.clinical_classes import ClinicalClasses
from typing import Dict, Sequence, Tuple, List
from mne.io import BaseRaw
import numpy as np
from ..config import get_config_value

ClassesType = BCIClasses | ClinicalClasses
DataType = np.ndarray | List[BaseRaw]
LabelsType = np.ndarray | List[str]

class AbstractDataset(ABC):
    def __init__(
        self,
        target_classes: Sequence[BCIClasses],
        subjects: Sequence[int],
    ):
        self.data: DataType
        self.labels: LabelsType
        self.meta: Dict

        self.target_classes = target_classes
        self.subjects = subjects
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def _download(self) -> None:
        pass

    def get_data(self) -> Tuple[DataType, LabelsType, Dict]:
        """
        Returns the data, labels and meta information of the dataset.
        For BCI it looks mostly like this:
        X is a list of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
        y is alist of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, ).
        meta is a list of dictionaries, for each dataset one dictionary.
            Each dictionary contains meta information about the samples.
            Such as the sampling frequency, the channel names, the labels mapping, etc.
        """
        if not self._check_data_loaded():
            self.load_data()
        return self.data, self.labels, self.meta

    def __getitem__(self, index) -> Tuple[DataType, LabelsType]:
        if not self._check_data_loaded():
            self.load_data()
        return self.data[index], self.labels[index]
    
    def __len__(self) -> int:
        if not self._check_data_loaded():
            self.load_data()
        return len(self.data)
    
    def _check_data_loaded(self) -> bool:
        if not hasattr(self, "data") or self.data is None:
            return False
        if not hasattr(self, "labels") or self.labels is None:
            return False
        return True