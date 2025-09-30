from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm
from resampy import resample
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download


def _load_data_albrecht2019(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    df_vars = pd.read_csv(os.path.join(data_path, 'DeID_Dems.csv'))
    sz_ids = df_vars.loc[df_vars['group'] == 'SZ', ['subno']].values

    all_subjects = ['CC_EEG_s155_N', 'CC_EEG_s154_N', 'CC_EEG_s138_P', 'CC_EEG_s101_P', 'CC_EEG_s158_N', 'CC_EEG_s156_N', 'CC_EEG_s117_P', 'CC_EEG_s146_P', 'CC_EEG_s162_N', 'CC_EEG_s176_N', 'CC_EEG_s167_N', 'CC_EEG_s141_P', 'CC_EEG_s133_P', 'CC_EEG_s174_N', 'CC_EEG_s172_N', 'CC_EEG_s163_N', 'CC_EEG_s170_N', 'CC_EEG_s112_P', 'CC_EEG_s173_N', 'CC_EEG_s128_P', 'CC_EEG_s125_P', 'CC_EEG_s111_P', 'CC_EEG_s108_P', 'CC_EEG_s161_N', 'CC_EEG_s151_N', 'CC_EEG_s153_N', 'CC_EEG_s168_N', 'CC_EEG_s131_P', 'CC_EEG_s139_P', 'CC_EEG_s119_P', 'CC_EEG_s160_N', 'CC_EEG_s140_P', 'CC_EEG_s169_N', 'CC_EEG_s109_P', 'CC_EEG_s136_P', 'CC_EEG_s178_N', 'CC_EEG_s164_N', 'CC_EEG_s107_P', 'CC_EEG_s171_N', 'CC_EEG_s165_N', 'CC_EEG_s143_P', 'CC_EEG_s130_P', 'CC_EEG_s105_P', 'CC_EEG_s166_N', 'CC_EEG_s150_N', 'CC_EEG_s114_P', 'CC_EEG_s127_P', 'CC_EEG_s126_P', 'CC_EEG_s124_P', 'CC_EEG_s135_P', 'CC_EEG_s157_N', 'CC_EEG_s145_P', 'CC_EEG_s132_P', 'CC_EEG_s120_P', 'CC_EEG_s123_P', 'CC_EEG_s144_P', 'CC_EEG_s110_P', 'CC_EEG_s113_P', 'CC_EEG_s137_P', 'CC_EEG_s102_P', 'CC_EEG_s116_P', 'CC_EEG_s115_P', 'CC_EEG_s134_P', "CC_EEG_s177_N", 'CC_EEG_s147_P', 'CC_EEG_s179_N', 'CC_EEG_s149_N', 'CC_EEG_s129_P', "CC_EEG_s152_N", "CC_EEG_s122_P", 'CC_EEG_s103_P', "CC_EEG_s180_N", "CC_EEG_s121_P", "CC_EEG_s159_N", 'CC_EEG_s104_P', "CC_EEG_s142_P"]

    this_subjects = [all_subjects[index] for index in subjects]

    # fmt: off
    channels = np.array(['Fp1', 'Fz', 'F3', 'F7', 'AFp9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'AFp10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'VEOGL', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'VEOGU', 'F6', 'F2', 'AF4', 'AF8'])
    non_eeg_channels = {'VEOGL', 'VEOGU'}
    eeg_indices = [i for i, ch in enumerate(channels) if ch not in non_eeg_channels]
    # fmt: on

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Albrecht2019"):
        file_path = os.path.join(data_path, "data", f"{subject}.mat")
        subject_id = int(subject.split('_')[2][1:])
        mat = loadmat(file_path, simplify_cells=True)
        signals = mat["EEG"]["data"]
        signals = signals[eeg_indices, :]
        data.append(signals)
        if target_class == ClinicalClasses.SCHIZOPHRENIA:
            labels.append("schizophrenia" if subject_id in sz_ids else "no_schizophrenia")
        elif target_class == ClinicalClasses.MEDICATION:
            assert subject_id in sz_ids, f"Subject {subject_id} is not in the schizophrenia group, so it has no medication label."
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Cloz']].values[0][0])
        elif target_class == ClinicalClasses.AGE:
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Age']].values[0][0])
        elif target_class == ClinicalClasses.SEX:
            # TODO have to check whether 0, 1 or 2 is male or female
            labels.append(df_vars.loc[df_vars['id']==subject_id, ['Sex']].values[0][0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class Albrecht2019Dataset(BaseClinicalDataset):
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
            name="Albrecht2019", # d004
            target_classes=[target_class],
            available_classes=[ClinicalClasses.SCHIZOPHRENIA, ClinicalClasses.MEDICATION, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=1000,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'AFp9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'AFp10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'F6', 'F2', 'AF4', 'AF8'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Albrecht2019Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }

        self.data_path = get_data_path("albrecht2019", "albrecht2019")
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

        self.data, self.labels = self.cache.cache(_load_data_albrecht2019)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta