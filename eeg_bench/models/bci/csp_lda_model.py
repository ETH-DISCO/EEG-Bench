from ..abstract_model import AbstractModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from typing import List, Dict, cast
import numpy as np
from mne.decoding import CSP
from resampy import resample
from sklearn.utils import shuffle


class CSPLDAModel(AbstractModel):
    def __init__(
        self,
        n_components: int = 4,
        reg=None,
        log=True,
        resample_rate: int = 200,
        channels: List[str] = ["C3", "Cz", "C4"], #["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "A1", "A2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"], #["C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "Fz", "Cz", "Pz"], # 'FC3', "FCz", 'FC4', 
    ):
        super().__init__("CSP-LDA")
        self.lda = LDA()
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
        print(f"fit X_prepared shape: {X_prepared.shape}")
        print(f"fit y_prepared shape: {y_prepared.shape}")

        X_prepared, y_prepared = cast(tuple[np.ndarray, np.ndarray], shuffle(X_prepared, y_prepared, random_state=42))
        # should be done by the benchmark and not by models

        # Transform both training and test data using the learned CSP filters
        X_csp = self.csp.fit_transform(X_prepared, y_prepared)

        # Fit LDA on CSP-transformed training data
        self.lda.fit(X_csp, y_prepared)

    def predict(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:
        [self.validate_meta(m) for m in meta]

        X_prepared = self._prepare_data(X, meta)
        print(f"predict X_prepared shape: {X_prepared.shape}")
        X_csp = self.csp.transform(X_prepared)
        return self.lda.predict(X_csp)

    def _prepare_data(self, X: List[np.ndarray], meta: List[Dict]) -> np.ndarray:

        X_resampled = []
        for data, m in zip(X, meta):
            if data.size == 0:
                continue
            # resample if needed
            # only take the C3, Cz, C4 channels
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

        # normalize data / standard reference
        #X_resampled = X_resampled - np.mean(X_resampled, axis=1, keepdims=True)

        return X_resampled
