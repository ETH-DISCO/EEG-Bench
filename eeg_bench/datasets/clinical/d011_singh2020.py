from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
import numpy as np
import pandas as pd
from mne.io import read_raw_brainvision
import warnings
from tqdm import tqdm
import random
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download


def _load_data_singh2020(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    ctr_subjects = ['Control1179', 'Control1199', 'Control1159', 'Control1369', 'Control1229', 'Control1249', 'Control1139', 'Control1239', 'Control1149', 'Control1129', 'Control1209', 'Control1359', 'Control1169']
    pd_subjects = ['PD1169', 'PD1329', 'PD1149', 'PD1389', 'PD1469', 'PD1339', 'PD1359', 'PD1219', 'PD1259', 'PD1349', 'PD1279', 'PD1309', 'PD1299', 'PD1539', 'PD1199', 'PD1559', 'PD1129', 'PD1089', 'PD1159', 'PD1319', 'PD1099', 'PD1229', 'PD1249', 'PD1369', 'PD1209', 'PD1239']
    all_subjects = ctr_subjects + pd_subjects
    this_subjects = [all_subjects[index] for index in subjects]

    df_vars = pd.read_csv(os.path.join(data_path, 'ALL_data_Modeling.csv'), sep='\t')
    df_vars['id_unique'] = 'PD' + df_vars['Pedal_ID'].astype(str)
    df_vars.loc[df_vars['Group']=='Control', ['id_unique']] = 'Control' + df_vars['Pedal_ID'].astype(str)

    rng = random.Random(42)
    rng.shuffle(this_subjects)

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Singh2020"):
        if subject in ctr_subjects:
            file_path = os.path.join(data_path, "raw_data", f"{subject}.vhdr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            signals = raw.get_data()*1e5
            if np.max(signals) > 1000 or np.min(signals) < -1000:
                print(f"Large values in subject {subject}: signal range out of bounds (min={np.min(signals)}, max={np.max(signals)})")
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject, ['Age']].values[0][0])
        else:
            file_path = os.path.join(data_path, "raw_data", f"{subject}.vhdr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
            raw.pick(['eeg'])
            signals = raw.get_data()*1e5
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['id_unique']==subject, ['Age']].values[0][0])
    labels = np.array(labels)

    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class Singh2020Dataset(BaseClinicalDataset):
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = 250,
        preload: bool = False,
    ):
        # fmt: off
        super().__init__(
            name="Singh2020", # d011
            target_classes=[target_class],
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.AGE],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'Aa', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'Bb', 'POz', 'Cc', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Singh2020Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }

        self.data_path = get_data_path("singh2020", "singh2020")
        self.data_path.mkdir(parents=True, exist_ok=True)
        if preload:
            self.load_data(split=Split.TRAIN)

    def _download(self):
        if os.path.exists(os.path.join(self.data_path, ".download_complete")):
            # It appears the dataset is already downloaded
            return
        print(f"===== Downloading Dataset {self.name} =====")
        snapshot_download("jalauer/" + self.name, repo_type="dataset", local_dir=self.data_path, local_dir_use_symlinks=False, resume_download=True)
        print(f"===== Dataset {self.name} download complete. Files stored at {self.data_path} =====")
        with open(os.path.join(self.data_path, ".download_complete"), "w") as file:
            file.write("This file tells the benchmarking code that the download of this dataset has completed, in order to avoid repeated downloads.")

    def load_data(self, split) -> None:
        self._download()

        self.data, self.labels = self.cache.cache(_load_data_singh2020)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
    
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta