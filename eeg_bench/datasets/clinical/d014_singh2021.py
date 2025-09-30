from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
from resampy import resample
import logging
from scipy.io import loadmat
import numpy as np
import pandas as pd
import glob
from mne.io import read_raw_brainvision
import warnings
from tqdm import tqdm
import random
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download


def _load_data_singh2021(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    ctr_subjects = ['Control1135', 'Control1195', 'Control1065', 'Control1275', 'Control1205', 'Control1025', 'Control1055', 'Control1225', 'Control1375', 'Control1295', 'Control1115', 'Control1405', 'Control1155', 'Control1265', 'Control1035', 'Control1385', 'Control1255', 'Control1335', 'Control1175', 'Control1305', 'Control1235', 'Control1415', 'Control1125', 'Control1095', 'Control1315', 'Control1395', 'Control1325', 'Control1185', 'Control1245', 'Control1085', 'Control1285', 'Control1215', 'Control1075']
    pd_subjects = ['PD1575', 'PD1525', 'PD1305', 'PD1445', 'PD1765', 'PD1025', 'PD1705', 'PD1625', 'PD1785', 'PD1615', 'PD1235', 'PD1555', 'PD1795', 'PD1045', 'PD1405', 'PD1265', 'PD1385', 'PD1325', 'PD1145', 'PD2515', 'PD1185', 'PD1055', 'PD1655', 'PD1375', 'PD2625', 'PD1175', 'PD1645', 'PD1245', 'PD1515', 'PD1585', 'PD1115', 'PD1125', 'PD1095', 'PD3625', 'PD1165', 'PD1535', 'PD1425', 'PD1485', 'PD1745', 'PD1635', 'PD1735', 'PD1335', 'PD1075', 'PD1215', 'PD1005', 'PD1565', 'PD1225', 'PD3515', 'PD1695', 'PD3565', 'PD1595', 'PD2855', 'PD1605', 'PD2815', 'PD2845', 'PD2565', 'PD1065', 'PD2445', 'PD1465', 'PD1775', 'PD1435', 'PD1365', 'PD1505', 'PD1725', 'PD1845', 'PD1035', 'PD1865', 'PD2835', 'PD1315', 'PD2865', 'PD1415', 'PD1855', 'PD1295', 'PD1715', 'PD1155', 'PD1675', 'PD1815', 'PD1135', 'PD1665']
    all_subjects = ctr_subjects + pd_subjects
    this_subjects = [all_subjects[index] for index in subjects]
    
    df_vars = pd.read_excel(os.path.join(data_path, 'Copy_of_IntervalTiming_Subj_Info_AIE.xlsx'), sheet_name='MAIN')

    rng.shuffle(this_subjects)

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Singh2021"):
        if subject in ctr_subjects:
            file_path = os.path.join(data_path, "data", f"{subject}.vhdr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)
                raw.pick(['eeg'])
            signals = raw.get_data(units='uV') / 10
            if np.min(signals) < -1000 or np.max(signals) > 1000:
                print(f"Skipping subject {subject} due to abnormal signal values: min={np.min(signals)}, max={np.max(signals)}")
                continue
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['Rest']==subject, ['Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                labels.append(df_vars.loc[df_vars['Rest']==subject, ['Gender']].values[0][0])
        else:
            file_path = os.path.join(data_path, "data", f"{subject}.vhdr")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = read_raw_brainvision(file_path, preload=True)     
                raw.pick(['eeg'])       
            signals = raw.get_data(units='uV') / 10
            if np.min(signals) < -1000 or np.max(signals) > 1000:
                print(f"Skipping subject {subject} due to abnormal signal values: min={np.min(signals)}, max={np.max(signals)}")
                continue
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['Rest']==subject, ['Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                labels.append(df_vars.loc[df_vars['Rest']==subject, ['Gender']].values[0][0])
    
    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class Singh2021Dataset(BaseClinicalDataset):
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
            name="Singh2021", # d014
            target_classes=[target_class],
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2', 'FCz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Singh2021Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }
        
        self.data_path = get_data_path("singh2021", "singh2021")
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

        self.data, self.labels = self.cache.cache(_load_data_singh2021)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
        
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta