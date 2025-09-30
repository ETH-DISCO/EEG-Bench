from ..abstract_model import AbstractModel
from typing import List, Dict, cast, Tuple, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.utils import shuffle
from joblib import Memory
from ...config import get_config_value
from .brainfeatures.feature_extraction_2 import _prepare_data_cached
from collections import Counter
import os

class BrainfeaturesLDAModel(AbstractModel):
    def __init__(
        self,
        resample_rate: int = 100,
        channels: List[str] = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'], # len:  49
        num_classes: int = 2,
        num_labels_per_chunk: Optional[int] = None,
    ):
        super().__init__('Brainfeatures-LDA')
        self.lda = LDA()

        self.is_multilabel_task = num_labels_per_chunk is not None
        if self.is_multilabel_task:
            self.lda = MultiOutputClassifier(self.lda, n_jobs=os.cpu_count() - 1)
        self.num_labels_per_chunk = num_labels_per_chunk
        self.num_classes = num_classes
        self.chunk_len_s = None if num_labels_per_chunk is None else 16

        self.resample_rate = resample_rate
        self.channels = channels
        self.cache = Memory(location=get_config_value("cache"), verbose=0)

    def fit(self, X: List[List[np.ndarray]], y: List[np.ndarray], meta: List[Dict]) -> None:
        if not "TUEG" in meta[0]["name"] and not self.is_multilabel_task:
            [self.validate_meta(m) for m in meta]
        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared, y_prepared = self._prepare_data(X, y, meta)
        del X, y, meta
        
        print(f"LDA fit X_prepared shape {X_prepared.shape}")
        print(f"LDA fit y_prepared shape {y_prepared.shape}")

        X_prepared, y_prepared = shuffle(X_prepared, y_prepared, random_state=42)
        # should be done by the benchmark and not by models

        # Fit LDA on extracted featues
        self.lda.fit(X_prepared, y_prepared)

    def predict(self, X: List[List[np.ndarray]], meta: List[Dict]) -> np.ndarray:
        if not "TUEG" in meta[0]["name"] and not self.is_multilabel_task:
            [self.validate_meta(m) for m in meta]

        X_prepared, _ = self._prepare_data(X, None, meta)

        print(f"LDA predict X_prepared shape {X_prepared.shape}")
        return self.lda.predict(X_prepared)

    def _prepare_data(self, X, y, meta) -> Tuple[np.ndarray, np.ndarray]:
        dataset_name = meta[0]["name"]
        
        feature_generation_params = {
            "domains": ["cwt", "dwt", "dft"], # or all
            "epoch_duration_s": 6,
            "max_abs_val": 800,
            "window_name": "blackmanharris",
            "band_limits": [[0, 2], [2, 4],  [4, 8], [8, 13],
                            [13, 18],  [18, 24], [24, 30], [30, 49.9]],
            "agg_mode": "mean",
            "discrete_wavelet": "db4",
            "continuous_wavelet": "morl",
            "band_overlap": True,
            "chunk_len_s": None if not self.is_multilabel_task else 16, # None, 10 or 60
            "resample_freq": 200, # self.resample_rate, # is 100
            "max_recording_len_min": 30 if not self.is_multilabel_task else False,
            "cutoff_start_s": 10,
            "cutoff_end_s": 5,
            "is_multilabel_task": self.is_multilabel_task,
        }
        print("Preparing data w/ ", feature_generation_params)

        all_data, all_labels = self.cache.cache(_prepare_data_cached)(X, y, meta, dataset_name, feature_generation_params) # type: ignore
        return all_data, all_labels