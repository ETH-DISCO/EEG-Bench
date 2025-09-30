from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
import numpy as np
import pandas as pd
from mne.io import read_raw_cnt
import warnings
from tqdm import tqdm
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download


def _load_data_gruendler2009(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    #intersection = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3']
    channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    df_vars = pd.read_excel(os.path.join(data_path, 'Info.xlsx'), sheet_name='SELECT', skiprows=[47, 48, 49, 50])
    ocd_list = df_vars.loc[df_vars['OCI'] >= 21.0, ['ID']].values.flatten().astype(int)
    selected_columns = ['ID', 'OCI', 'Sex', 'Age', 'BDI']
    values_df = df_vars[selected_columns]

    all_subjects = ['901', '904', '905', '906', '907', '908', '911', '912', '915', '916', '917', '919', '920', '921', '922', '924', '926', '927', '929', '933', '934', '935', '936', '937', '939', '940', '941', '945', '946', '948', '952', '953', '958', '959', '960', '903', '909', '914', '925', '930', '931', '932', '938', '950', '956', '957']
    this_subjects = [all_subjects[index] for index in subjects]

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Gruendler2009"):
        subject_id = int(subject)
        file_path = os.path.join(data_path, "data", f"{subject_id}flankers{'' if subject_id < 945 else '_ready'}.cnt")
        targets_df = values_df.loc[values_df['ID'] == subject_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw = read_raw_cnt(file_path, preload=True)
        cols = [ch['ch_name'] for ch in raw.info['chs'] if ch['ch_name'].upper() in channels]
        raw = raw.reorder_channels(cols)
        raw.set_eeg_reference("average")
        signals = raw.get_data(units='uV')
        if np.max(np.abs(signals)) > 10000000:
            print(f"Subject {subject_id} has signals with max value {np.max(np.abs(signals))} uV")
            signals = signals / 1000000
        if np.max(np.abs(signals)) > 10000:
            print(f"Subject {subject_id} has signals with max value {np.max(np.abs(signals))} uV")
            signals = signals / 1000
        data.append(signals)
        if target_class == ClinicalClasses.OCD:
            labels.append("ocd" if subject_id in ocd_list else "no_ocd")
        elif target_class == ClinicalClasses.OCI:
            labels.append(targets_df['OCI'].values[0])
        elif target_class == ClinicalClasses.BDI:
            labels.append(targets_df['BDI'].values[0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(targets_df['Age'].values[0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(targets_df['Sex'].values[0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class Gruendler2009Dataset(BaseClinicalDataset):
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
            name="Gruendler2009", # d008
            target_classes=[target_class],
            available_classes=[ClinicalClasses.OCD, ClinicalClasses.OCI, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'],
            #channel_names=['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Gruendler2009Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }

        self.data_path = get_data_path("gruendler2009", "gruendler2009")
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

        self.data, self.labels = self.cache.cache(_load_data_gruendler2009)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta