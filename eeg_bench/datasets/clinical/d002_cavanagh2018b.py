from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple
import logging
from scipy.io import loadmat
import numpy as np
from resampy import resample
import pandas as pd
from tqdm import tqdm
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download


def _load_data_cavanagh2018b(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    
    pd_subjects = ["801", "802", "803", "804", "805", "806", "807", "808", "809", "810", "811", "813", "814", "815", "816", "817", "818", "819", "820", "821", "822", "823", "824", "825", "826", "827", "828", "829"]
    no_pd_subjects = ["890", "891", "892", "893", "894", "895", "896", "897", "898", "899", "900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911", "912", "913", "914", "8010", "8060", "8070"]
    all_subjects = pd_subjects + no_pd_subjects
    this_subjects = [all_subjects[index] for index in subjects]

    df_vars = pd.read_excel(os.path.join(data_path, "IMPORT_ME_REST.xlsx"))

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Cavanagh2018b"):
        if subject in pd_subjects:
            # Use os.path.join and a configurable file suffix to allow easier customization.
            file_name_1 = os.path.join(data_path, "data", f"{subject}_1_PD_REST.mat")
            if df_vars.loc[df_vars['PD_ID']==int(subject), ['1st Visit Meds Status']].values[0][0] == "OFF":
                mat_1 = loadmat(file_name_1, simplify_cells=True)
                signals = mat_1['EEG']['data']
                signals = signals[:63, :]
                data.append(signals)
                if target_class == ClinicalClasses.PARKINSONS:
                    labels.append("parkinsons")
                elif target_class == ClinicalClasses.AGE:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['PD_Age']].values[0][0])
                elif target_class == ClinicalClasses.SEX:
                    # have to check whether 0, 1 or 2 is men or women
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['PD_Sex']].values[0][0])
                elif target_class == ClinicalClasses.BDI:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['BDI']].values[0][0])
                elif target_class == ClinicalClasses.MEDICATION:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['1st Visit Meds Status']].values[0][0])
            else:
                file_name_2 = os.path.join(data_path, "data", f"{subject}_2_PD_REST.mat")
                mat_2 = loadmat(file_name_2, simplify_cells=True)
                signals = mat_2['EEG']['data']
                signals = signals[:63, :]
                data.append(signals)
                if target_class == ClinicalClasses.PARKINSONS:
                    labels.append("parkinsons")
                elif target_class == ClinicalClasses.AGE:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['PD_Age']].values[0][0])
                elif target_class == ClinicalClasses.SEX:
                    # have to check whether 0, 1 or 2 is men or women
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['PD_Sex']].values[0][0])
                elif target_class == ClinicalClasses.BDI:
                    labels.append(df_vars.loc[df_vars['PD_ID']==int(subject), ['BDI']].values[0][0])
                elif target_class == ClinicalClasses.MEDICATION:
                    labels.append("ON" if df_vars.loc[df_vars['PD_ID']==int(subject), ['1st Visit Meds Status']].values[0][0] == "OFF" else "OFF")
        
        else:
            assert not target_class in [ClinicalClasses.MEDICATION, ClinicalClasses.BDI], "no medication or bdi data for subjects without parkinsons"
            file_name = os.path.join(data_path, "data", f"{subject}_1_PD_REST.mat")
            mat = loadmat(file_name, simplify_cells=True)
            signals = mat['EEG']['data']
            signals = signals[:63, :]
            data.append(signals)
            if target_class == ClinicalClasses.PARKINSONS:
                labels.append("no_parkinsons")
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['MATCH CTL_ID']==int(subject), ['MATCH CTL_Age']].values[0][0])
            elif target_class == ClinicalClasses.SEX:
                # have to check whether 0, 1 or 2 is men or women
                labels.append(df_vars.loc[df_vars['MATCH CTL_ID']==int(subject), ['MATCH CTL_Sex']].values[0][0])
    labels = np.array(labels)

    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in data]
    return data, labels


class Cavanagh2018bDataset(BaseClinicalDataset):
    """
    - self.data: List of length n_subjects, where each element is a numpy array of shape (n_channels, n_samples)
    """
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
            name="Cavanagh2018b", # d002
            target_classes=[target_class],
            available_classes=[ClinicalClasses.PARKINSONS, ClinicalClasses.MEDICATION, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Cavanagh2018bDataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }
        
        self.data_path = get_data_path("cavanagh2018b", "cavanagh2018b")
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

        self.data, self.labels = self.cache.cache(_load_data_cavanagh2018b)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency
    
    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta