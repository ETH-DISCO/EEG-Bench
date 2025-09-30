from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple, List, Dict
import logging
import glob
import os
from mne.io import read_raw_edf, Raw
from mne import rename_channels
import wfdb
from tqdm import tqdm
from ...config import get_data_path
from pathlib import Path


DATA_PATH = get_data_path("chb_mit", "chb_mit")

def _load_data_chb_mit(subjects: Sequence[int], preload: bool) -> Tuple[List[List[Raw]], List[str], List[List[str]]]:

    data = []
    labels = []
    montage_type = []
    channel_map = {
        "T8-P8-0": "T8-P8"
    }
    # channel_list must be intersection of channels in all records
    channel_list = ["F3-C3", "F8-T8", "F4-C4", "FP1-F7", "P4-O2", "P8-O2", "T8-P8", "FZ-CZ", "P7-O1", "T7-P7", "FP2-F8", "CZ-PZ", "FP1-F3", "C3-P3", "C4-P4", "P3-O1", "F7-T7",  "FP2-F4"]
    for subject in tqdm(subjects):
        session_str = "chb{:02d}".format(subject)
        files = glob.glob(os.path.join(DATA_PATH, session_str, "*.edf"), recursive=True)
        if subject == 1:
            # session 21 is from the same subject as session 1
            files += glob.glob(os.path.join(DATA_PATH, "chb21", "*.edf"), recursive=True)
        for file in files:
            raw = read_raw_edf(file, verbose="error", preload=preload)
            if raw.times[-1] < 60:
                raw.close()
                continue

            # rename channels and remove unneeded ones
            rename_channels(raw.info, lambda old_ch: channel_map[old_ch] if old_ch in channel_map.keys() else old_ch)
            if all([ch not in raw.ch_names for ch in channel_list]):
                continue # happens only for 3 recordings
            raw = raw.reorder_channels(channel_list)

            data.append(raw)
            # process annotations
            label = []
            annotation_path = file + ".seizures"
            if os.path.exists(annotation_path):
                ann = wfdb.rdann(annotation_path[:-13], "edf.seizures")
                event_list = ann.__dict__["sample"]
                # event_list = [a,b,c,d,...], where (a,b), (c,d), ... are seizure segments
                assert len(event_list) % 2 == 0
                for i in range(0, len(event_list), 2):
                    start = event_list[i]
                    stop = event_list[i+1]
                    label.append(("Seizure", start, stop))
            labels.append(label)
            montage_type.append("None")
        
    return data, labels, montage_type

class CHBMITDataset(BaseClinicalDataset):
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
            name="CHB-MIT",
            target_classes=target_classes,
            available_classes=[ClinicalClasses.SEIZURE],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=0, # individual for each recording
            channel_names=[], # individual for each recording
            preload=preload,
        )
        # fmt: on
        logging.info("in CHBMITDataset.__init__")
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
        if len(data_files) == 686:
            # If .edf files are found, we assume the dataset is already downloaded
            return
        
        os.system("cd {:s} && wget -r -N -c -np -nH --cut-dirs=3 https://physionet.org/files/chbmit/1.0.0/".format(str(DATA_PATH)))

    def load_data(self) -> None:
        self._download()
        
        self.data, self.labels, montage_type = self.cache.cache(_load_data_chb_mit)(self.subjects, self._preload) # type: ignore
        self.meta["montage_type"] = montage_type

    def get_data(self) -> Tuple[List[List[Raw]], List[str], Dict]:

        if not hasattr(self, "data") or self.data is None or not hasattr(self, "labels") or self.labels is None or self.meta is None:
            self.load_data()
        return self.data, self.labels, self.meta

