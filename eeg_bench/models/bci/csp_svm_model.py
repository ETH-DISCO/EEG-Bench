from ..abstract_model import AbstractModel
from sklearn.svm import SVC
from typing import List, Dict, cast, Literal
import numpy as np
from mne.decoding import CSP
from resampy import resample
from sklearn.utils import shuffle


class CSPSVMModel(AbstractModel):
    def __init__(
        self,
        kernel : Literal["linear"] = "linear",
        C=1.0,
        random_state=42,
        n_components=4,
        reg=None,
        log=True,
        resample_rate=200,
        channels=["C3", "Cz", "C4"],
    ):
        super().__init__("CSP-SVM")
        self.svm = SVC(kernel=kernel, C=C, random_state=random_state, class_weight="balanced")
        self.csp = CSP(n_components=n_components, reg=reg, log=log)
        self.resample_rate = resample_rate
        self.channels = channels

    def fit(self, X: List[np.ndarray], y: List[np.ndarray], meta: List[Dict]) -> None:
        [self.validate_meta(m) for m in meta]
        self._set_channels(meta[0]["task_name"])
        # bring data into the right shape, so resample if needed and only take the C3, Cz, C4 channels
        # TODO handle case if no channel names are provided
        X_prepared = self._prepare_data(X, meta)
        y_prepared = np.concatenate(y, axis=0)

        X_prepared, y_prepared = cast(tuple[np.ndarray, np.ndarray], shuffle(X_prepared, y_prepared, random_state=42)) # should be done by the benchmark and not by models

        # Transform both training and test data using the learned CSP filters
        X_csp = self.csp.fit_transform(X_prepared, y_prepared)

        # Fit LDA on CSP-transformed training data
        self.svm.fit(X_csp, y_prepared)

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)
        X_csp = self.csp.transform(X_prepared)
        return self.svm.predict(X_csp)

    def _prepare_data(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:

        X_resampled = []
        for data, m in zip(X, meta):
            if data.size == 0:
                continue
            # resample if needed
            channel_indices = [m["channel_names"].index(ch) for ch in self.channels]
            data = data[:, channel_indices, :]
            if m["sampling_frequency"] != self.resample_rate:
                data = resample(
                    data,
                    m["sampling_frequency"],
                    self.resample_rate,
                    axis=2,
                    filter="kaiser_best",
                )
            X_resampled.append(data)

        # check if all have the same duration i.e. n_timepoints
        durations = set([d.shape[2] for d in X_resampled])
        if not len(durations) == 1:
            min_duration = min(durations)
            X_resampled = [d[:, :, :min_duration] for d in X_resampled]

        X_resampled = np.concatenate(X_resampled, axis=0)
        X_resampled = X_resampled.astype(np.float64)

        return X_resampled
