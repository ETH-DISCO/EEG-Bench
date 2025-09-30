from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np


class AbstractModel(ABC):

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def validate_meta(self, meta: Dict) -> None:
        """
        Validate the meta information.

        Parameters
        ----------
        meta : Dict
            Dictionary containing meta information about the samples.
            Such as the sampling frequency, the channel names, the labels mapping, etc.

        Raises
        ------
        ValueError
            If the meta information is not valid.
        """
        if "sampling_frequency" not in meta:
            raise ValueError("Meta information must contain the sampling frequency.")
        if "channel_names" not in meta:
            raise ValueError("Meta information must contain the channel names.")
    
    def _set_channels(self, task_name) -> None:
        if task_name == "Left Hand vs Right Hand MI":
            self.channels = ["C3", "Cz", "C4"]
        elif task_name == "Right Hand vs Feet MI":
            self.channels = ['C3', 'Cz', 'C4']
        elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
            self.channels = ["C3", "Cz", "C4", "Fz", "Pz"]
        elif task_name == "Five Fingers MI":
            self.channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
        else:
            raise ValueError("Invalid task name: ", task_name)

    @abstractmethod
    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        X : List[np.ndarray]
            List of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
        y : List[np.ndarray]
            List of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, ).
        meta : List[Dict]
            List of dictionaries, for each dataset one dictionary.
            Each dictionary contains meta information about the samples.
            Such as the sampling frequency, the channel names, the labels mapping, etc.
        """
        pass

    @abstractmethod
    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        """
        Predict the labels for the given data.

        Parameters
        ----------
        X : List[np.ndarray]
            List of numpy arrays, for each dataset one numpy array.
            Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
        meta : List[Dict]
            List of dictionaries, for each dataset one dictionary.
            Each dictionary contains meta information about the samples.
            Such as the sampling frequency, the channel names, the labels mapping, etc.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        pass
