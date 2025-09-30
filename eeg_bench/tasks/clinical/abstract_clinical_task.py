from abc import ABC, abstractmethod
from ...datasets.abstract_dataset import AbstractDataset
from ...enums.split import Split
from ...enums.clinical_classes import ClinicalClasses
from typing import List, Tuple, Dict, Type, Sequence, Optional
import numpy as np


class AbstractClinicalTask(ABC):
    def __init__(
        self,
        name: str,
        clinical_classes: Sequence[ClinicalClasses],
        datasets: Sequence[Type[AbstractDataset]],
        subjects_split: Dict[Type[AbstractDataset], Dict[Split, Sequence[int]]],
        event_map: Optional[Dict[int, int]] = None,
        chunk_len_s: Optional[int] = None,
        num_labels_per_chunk: Optional[int] = None,
    ):
        assert len(datasets) > 0, "At least one dataset is required"
        assert set(datasets).issubset(subjects_split.keys()), "Subjects split must match datasets"
        assert all(
            subjects_split[dataset].keys() == {Split.TRAIN, Split.TEST}
            for dataset in datasets
        ), "Subjects split must contain train and test splits"
        self.name = name
        self.clinical_classes = clinical_classes
        self.datasets = datasets
        self.subjects_split = subjects_split
        self.event_map = event_map # maps event types (e.g. "Sleep stage 2") to a class (e.g. 3)
        self.chunk_len_s = chunk_len_s
        self.num_labels_per_chunk = num_labels_per_chunk

    def get_data(
        self, split: Split
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Get the data for the given split.

        Parameters
        ----------
        split : Split
            The split for which to get the data.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]
            The data, labels and meta information for the given split.

            X is a list of numpy arrays, for each dataset one numpy array.
                Each numpy array has dimensions (n_samples, n_channels, n_timepoints).
            y is alist of numpy arrays, for each dataset one numpy array.
                Each numpy array has dimensions (n_samples, ).
            meta is a list of dictionaries, for each dataset one dictionary.
                Each dictionary contains meta information about the samples.
                Such as the sampling frequency, the channel names, the labels mapping, etc.
        """

        data = [
            dataset(
                target_classes=self.clinical_classes,
                subjects=self.subjects_split[dataset][split],
            ).get_data() # type: ignore
            for dataset in self.datasets
        ]

        X, y, meta = map(list, zip(*data))
        for m in meta:
            m['task_name'] = self.name
            if self.event_map is not None:
                m['event_map'] = self.event_map
        return X, y, meta

    def __str__(self):
        return self.name

    @abstractmethod
    def get_metrics(self):
        """
        Retrieve the scoring function associated with this task.

        Returns
        -------
        function : callable
            A function that accepts two parameters, y_true and y_pred, and returns a float
            representing the score. This function is used to evaluate the performance of
            predictions against the true labels.
        """
        pass
