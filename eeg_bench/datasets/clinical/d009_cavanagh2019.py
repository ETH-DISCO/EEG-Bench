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
from tqdm import tqdm
from ...config import get_data_path
import os
from huggingface_hub import snapshot_download
from einops import rearrange


def _load_data_cavanagh2019(data_path, split: Split, subjects: Sequence[int], target_class: ClinicalClasses, sampling_frequency: int, resampling_frequency: Optional[int] = None) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    mat_raw = loadmat(os.path.join(data_path, "scripts/BigAgg_Data.mat"), simplify_cells=True)
    df_vars = pd.DataFrame()
    df_vars_q = pd.DataFrame()

    all_subjects = ['3001', '3002', '3003', '3004', '3005', '3007', '3009', '3010', '3013', '3062', '3014', '3015', '3016', '3018', '3019', '3020', '3021', '3023', '3024', '3025', '3026', '3027', '3028', '3029', '3030', '3032', '3033', '3034', '3035', '3036', '3037', '3038', '3039', '3041', '3043', '3044', '3045', '3046', '3047', '3049', '3050', '3051', '3052', '3054', '3056', '3058', '3060', '3068', '3070', '3072', '3076', '3078', '3082', '3084', '3086', '3088', '3090', '3092', '3006', '3008', '3011', '3012', '3017', '3031', '3040', '3042', '3048', '3074', '3080', '3064', '3066']
    this_subjects = [all_subjects[index] for index in subjects]

    df_vars['SubID'] = mat_raw['DEMO']['ID'][:, 0]
    df_vars['ControlEquals1'] = mat_raw['DEMO']['Group_CTL1'][:, 0]
    df_vars['FemaleEquals1'] = mat_raw['DEMO']['Sex_F1']
    df_vars['Age'] = mat_raw['DEMO']['Age']
    df_vars['BDItotal'] = mat_raw['QUEX']['BDI'][:, 0]

    df_vars_q['SubID'] = mat_raw['Q_DEMO']['URSI']
    df_vars_q['ControlEquals1'] = 0
    df_vars_q['FemaleEquals1'] = mat_raw['Q_DEMO']['Sex_F1']
    df_vars_q['Age'] = mat_raw['Q_DEMO']['Age']
    df_vars_q['BDItotal'] = mat_raw['Q_QUEX']['BDI']

    df_vars = pd.concat([df_vars, df_vars_q], ignore_index=True)
    mtbi_list = df_vars.loc[df_vars['ControlEquals1'] != 1, ['SubID']].values.astype(int).flatten()

    data = []
    labels = []
    for subject in tqdm(this_subjects, desc="Loading data from Cavanagh2019"):
        subject_id = int(subject)
        files = glob.glob(os.path.join(data_path, "data", f"{subject_id}_*_3AOB.mat"))
        for file_path in files:
            mat = loadmat(file_path, simplify_cells=True)
            signals = mat["EEG"]["data"]
            # signals = signals.reshape(60, -1)
            signals = rearrange(signals, "Channels Duration Trials -> Channels (Trials Duration)")
            data.append(signals)
            if target_class == ClinicalClasses.MTBI:
                labels.append(subject_id in mtbi_list)
            elif target_class == ClinicalClasses.BDI:
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'BDItotal'].values[0])
            elif target_class == ClinicalClasses.AGE:
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'Age'].values[0])
            elif target_class == ClinicalClasses.SEX:
                # TODO have to check whether 0, 1 or 2 is male or female
                labels.append(df_vars.loc[df_vars['SubID'] == subject_id, 'FemaleEquals1'].values[0])

    labels = np.array(labels)
    if resampling_frequency is not None:
        data = [resample(d, sampling_frequency, resampling_frequency, axis=-1, filter='kaiser_best', parallel=True) for d in tqdm(data, desc="Resampling data from Cavanagh2019")]
    return data, labels


class Cavanagh2019Dataset(BaseClinicalDataset):
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
            name="Cavanagh2019", # d009
            target_classes=[target_class],
            available_classes=[ClinicalClasses.MTBI, ClinicalClasses.BDI, ClinicalClasses.AGE, ClinicalClasses.SEX],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=500,
            channel_names=['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz'],
            preload=preload,
        )
        # fmt: on
        logging.info("in Cavanagh2019Dataset.__init__")
        self.meta = {
            "sampling_frequency": self._sampling_frequency,
            "channel_names": self._channel_names,
            "name": self.name,
        }

        self.data_path = get_data_path("cavanagh2019", "cavanagh2019")
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

        self.data, self.labels = self.cache.cache(_load_data_cavanagh2019)(
            self.data_path, split, self.subjects, self.target_classes[0], self._sampling_frequency, self._target_frequency) # type: ignore
        if self._target_frequency is not None:
            self._sampling_frequency = self._target_frequency
            self.meta["sampling_frequency"] = self._sampling_frequency

    def get_data(self, split: Split):
        self.load_data(split)
        return self.data, self.labels, self.meta
