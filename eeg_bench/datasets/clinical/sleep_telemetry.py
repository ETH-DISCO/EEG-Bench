from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple, List, Dict
import logging
import glob
import os
from mne.io import read_raw_edf, Raw
from mne import rename_channels
import mne
import wfdb
import numpy as np
from tqdm import tqdm
from ...config import get_data_path
from pathlib import Path


DATA_PATH = get_data_path("sleep_telemetry", "sleep_telemetry")


def _load_data_sleep_telemetry(subjects: Sequence[int], preload: bool) -> Tuple[List[List[Raw]], List[str], List[List[str]]]:

    data = []
    labels = []
    montage_type = []
    channel_map = {
        'EEG Fpz-Cz': "FPZ-CZ",
        'EEG Pz-Oz': "PZ-OZ",
        'EOG horizontal': "EOGh"
    }
    # MUST BE INTERSECTION OF CHANNELS IN ALL RECORDS!
    channel_list = ['FPZ-CZ', 'PZ-OZ', 'EOGh']
    for subject in tqdm(subjects):
        prefix = "ST7{:02d}".format(subject)
        files = glob.glob(os.path.join(DATA_PATH, prefix + "*-PSG.edf"), recursive=True)
        for file in files:
            raw = read_raw_edf(file, verbose="error", preload=preload)
            if raw.times[-1] < 60:
                raw.close()
                continue

            # rename channels and remove unneeded ones
            rename_channels(raw.info, lambda old_ch: channel_map[old_ch] if old_ch in channel_map.keys() else old_ch)
            raw = raw.reorder_channels(channel_list)
            data.append(raw)
            # process annotations
            label = []
            all_files = os.listdir(DATA_PATH)
            annotation_path = [f for f in all_files if f.startswith(file.split('/')[-1][:-10]) and f.endswith("-Hypnogram.edf")]
            assert len(annotation_path) == 1
            annotation_path = os.path.join(DATA_PATH, annotation_path[0])
            # Add annotations
            annotations = mne.read_annotations(annotation_path)
            assert len(annotations.onset) == len(annotations.duration)
            assert len(annotations.duration) == len(annotations.description)
            # sleep stage segments
            for i in range(len(annotations.onset)):
                if annotations.description[i] not in ["Sleep stage ?", "Movement time"]:
                    start = int(round(annotations.onset[i] * raw.info['sfreq']))
                    stop = int(round((annotations.onset[i] + annotations.duration[i]) * raw.info['sfreq']))
                    if annotations.description[i] in ["Sleep stage 3", "Sleep stage 4"]:
                        label.append(["Sleep stage 3/4", start, stop])
                    else:
                        label.append([annotations.description[i], start, stop])
            labels.append(label)
            montage_type.append("None")
        
    return data, labels, montage_type

class SleepTelemetryDataset(BaseClinicalDataset):
    def __init__(
        self,
        target_classes: Sequence[ClinicalClasses],
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = False,
    ):
        # fmt: off
        super().__init__(
            name="Sleep-Telemetry",
            target_classes=target_classes,
            available_classes=[ClinicalClasses.WAKE, ClinicalClasses.N1, ClinicalClasses.N2, ClinicalClasses.N34, ClinicalClasses.REM],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=0, # individual for each recording
            channel_names=[], # individual for each recording
            preload=preload,
        )
        # fmt: on
        logging.info("in SleepTelemetryDataset.__init__")
        self.data: List[List[Raw]]
        self.labels: List[str]
        self.meta = {
            "name": self.name,
            "montage_type": []
        }

        if preload:
            self.load_data()

    def _download(self):
        # check whether the path exists
        if DATA_PATH is None:
            raise ValueError("DATA_PATH is not set. Please set the variable \"data\" in the eeg_bench/config.json file.")
        
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH, exist_ok=True)

        # check whether the dataset is already downloaded
        data_files = list(Path(DATA_PATH).rglob("*.edf"))
        if len(data_files) == 88:
            # If .edf files are found, we assume the dataset is already downloaded
            return

        os.system("cd {:s} && wget -r -N -c -np -nH --cut-dirs=4 https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/".format(str(DATA_PATH)))
    
    def load_data(self) -> None:
        self._download()
        
        self.data, self.labels, montage_type = self.cache.cache(_load_data_sleep_telemetry)(self.subjects, self._preload) # type: ignore
        self.meta["montage_type"] = montage_type

    def get_data(self) -> Tuple[List[List[Raw]], List[str], Dict]:
        if not hasattr(self, "data") or self.data is None or not hasattr(self, "labels") or self.labels is None or self.meta is None:
            self.load_data()
        return self.data, self.labels, self.meta
