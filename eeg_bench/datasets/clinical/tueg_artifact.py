from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from ...enums.split import Split
from typing import Optional, Sequence, Tuple, List, Dict
import logging
import glob
import os
from mne.io import read_raw_edf, Raw
from mne import rename_channels
from tqdm import tqdm
from ...config import get_data_path
import csv
import copy
from pathlib import Path

DATA_PATH = get_data_path("tuar", "tuar")

def _load_data_tuar(subjects: Sequence[int], preload: bool) -> Tuple[List[List[Raw]], List[str], List[List[str]]]:

    all_subjects = [
        'aaaaanzu', 'aaaaapks', 'aaaaagvx', 'aaaaamdu', 'aaaaafwz', 'aaaaabiw', 'aaaaajso', 'aaaaaobi', 'aaaaaofj', 'aaaaamxn', 'aaaaaluz', 'aaaaaobl', 'aaaaadjk', 'aaaaaogd', 'aaaaadao', 'aaaaanxz', 'aaaaanyb', 'aaaaakdo', 'aaaaaarq', 'aaaaafdh', 'aaaaakft', 'aaaaajjb', 'aaaaaoby', 'aaaaamzl', 'aaaaanzh', 'aaaaaelb', 'aaaaaayx', 'aaaaajys', 'aaaaalqs', 'aaaaaguk', 'aaaaaijh', 'aaaaakev', 'aaaaaiyu', 'aaaaapas', 'aaaaabuv', 'aaaaajfk', 'aaaaabdo', 'aaaaaksb', 'aaaaachj', 'aaaaanro', 'aaaaajoo', 'aaaaamdq', 'aaaaanvw', 'aaaaagcs', 'aaaaalth', 'aaaaanlp', 'aaaaaoxr', 'aaaaanhb', 'aaaaaayg', 'aaaaaiue', 'aaaaapnl', 'aaaaahte', 'aaaaajpf', 'aaaaanvg', 'aaaaajns', 'aaaaanpr', 'aaaaanvb', 'aaaaakqg', 'aaaaanwc', 'aaaaakeq', 'aaaaakcy', 'aaaaaomy', 'aaaaaezn', 'aaaaaphs', 'aaaaanbp', 'aaaaamhj', 'aaaaamrj', 'aaaaapqh', 'aaaaacyf', 'aaaaaosa', 'aaaaamsc', 'aaaaapre', 'aaaaakbz', 'aaaaalco', 'aaaaanta', 'aaaaanbq', 'aaaaalvw', 'aaaaaoeg', 'aaaaanyt', 'aaaaajoa', 'aaaaajrj', 'aaaaanqe', 'aaaaamtj', 'aaaaapkv', 'aaaaalqa', 'aaaaansj', 'aaaaaktz', 'aaaaajtu', 'aaaaaowf', 'aaaaahzm', 'aaaaalmh', 'aaaaapcr', 'aaaaaeab', 'aaaaaike', 'aaaaalfj', 'aaaaakka', 'aaaaaogk', 'aaaaaosh', 'aaaaaezj', 'aaaaagxr', 'aaaaamcr', 'aaaaanrc', 'aaaaahzs', 'aaaaaloy', 'aaaaanwj', 'aaaaagqb', 'aaaaanum', 'aaaaammu', 'aaaaajzm', 'aaaaaibx', 'aaaaampz', 'aaaaakck', 'aaaaaimu', 'aaaaanyw', 'aaaaaljj', 'aaaaaouk', 'aaaaaohw', 'aaaaaovl', 'aaaaacby', 'aaaaappv', 'aaaaabbn', 'aaaaamnq', 'aaaaallc', 'aaaaangp', 'aaaaapmb', 'aaaaamyb', 'aaaaajqh', 'aaaaatjz', 'aaaaakqr', 'aaaaajrf', 'aaaaaoda', 'aaaaakcx', 'aaaaajqk', 'aaaaamuc', 'aaaaamyc', 'aaaaalxk', 'aaaaamzf', 'aaaaadmi', 'aaaaamhb', 'aaaaaall', 'aaaaaozv', 'aaaaabnn', 'aaaaalnt', 'aaaaapkx', 'aaaaapex', 'aaaaakfq', 'aaaaajqb', 'aaaaaiby', 'aaaaamqc', 'aaaaalaf', 'aaaaakcv', 'aaaaaoav', 'aaaaakdj', 'aaaaajqu', 'aaaaaprj', 'aaaaaizu', 'aaaaaoiv', 'aaaaafhl', 'aaaaamon', 'aaaaadsm', 'aaaaamyy', 'aaaaamzi', 'aaaaaiat', 'aaaaalid', 'aaaaaenq', 'aaaaaout', 'aaaaambs', 'aaaaadeu', 'aaaaaksu', 'aaaaankf', 'aaaaajqo', 'aaaaanji', 'aaaaanxl', 'aaaaafcf', 'aaaaabms', 'aaaaajam', 'aaaaalox', 'aaaaajah', 'aaaaamzv', 'aaaaamob', 'aaaaamoa', 'aaaaahuy', 'aaaaaovn', 'aaaaajuh', 'aaaaanyf', 'aaaaaora', 'aaaaakxz', 'aaaaaovo', 'aaaaapmp', 'aaaaaohr', 'aaaaapar', 'aaaaabsk', 'aaaaalbt', 'aaaaakxo', 'aaaaanyc', 'aaaaakgp', 'aaaaafsb', 'aaaaalsq', 'aaaaanoz', 'aaaaaaju', 'aaaaankc', 'aaaaagsc', 'aaaaapxk', 'aaaaalzg', 'aaaaaicc', 'aaaaaons', 'aaaaaltw', 'aaaaaels', 'aaaaajoe', 'aaaaakmg', 'aaaaacad', 'aaaaapcu', 'aaaaaiae']

    class_map = {
        "musc": "Muscle Artifact",
        "eyem": "Eye Movement",
        "elec": "Electrode Artifact",
        "chew": "Chewing",
        "shiv": "Shivers",
        "elpp": "Electrode Pop",
    }

    data = []
    labels = []
    montage_type = []
    # channel_list must be intersection of channels in all records
    channel_list = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']

    all_files = glob.glob(os.path.join(DATA_PATH, "01_tcp_ar", "*.edf")) + glob.glob(os.path.join(DATA_PATH, "02_tcp_le", "*.edf")) + glob.glob(os.path.join(DATA_PATH, "03_tcp_ar_a", "*.edf"))
    for subject in tqdm(subjects):
        subject_name = all_subjects[subject]
        files = [file for file in all_files if file.split('/')[-1][:8] == subject_name]
        assert len(files) > 0
        for file in files:
            raw = read_raw_edf(file, verbose="error", preload=preload)
            if raw.times[-1] < 60:
                raw.close()
                continue

            # rename channels and remove unneeded ones
            raw_ch_copy = copy.deepcopy(raw.ch_names)
            rename_channels(raw.info, lambda old_ch: old_ch[4:-4] if old_ch.endswith("-REF") else (old_ch[4:-3] if old_ch.endswith("-LE") and (old_ch[:-3] + "-REF") not in raw_ch_copy else old_ch))
            raw = raw.reorder_channels(channel_list)
            data.append(raw)
            # process annotations
            label = []
            annotation_path = file[:-4] + ".csv"
            assert os.path.exists(annotation_path)
            with open(annotation_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if (not row[0].startswith("#")) and row[0] != "channel" and row[3] != "bckg":
                        assert len(row) == 5 and float(row[4]) == 1.0
                        # scale start and stop times from seconds to sampling_frequency Hz
                        start = int(round(float(row[1]) * raw.info['sfreq']))
                        stop = int(round(float(row[2]) * raw.info['sfreq']))
                        if row[3] in ["eyem_musc", "musc_elec", "eyem_elec", "eyem_chew", "chew_musc", "chew_elec", "eyem_shiv", "shiv_elec"]:
                            label.append([class_map[row[3][:4]], start, stop])
                            label.append([class_map[row[3][5:9]], start, stop])
                        else:
                            label.append([class_map[row[3]], start, stop])
            labels.append(label)
            montage_type.append("None")
        
    return data, labels, montage_type

class TUARDataset(BaseClinicalDataset):
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
            name="TUAR",
            target_classes=target_classes,
            available_classes=[ClinicalClasses.ARTIFACT, ClinicalClasses.EYE_MOVEMENT, ClinicalClasses.MUSCLE_ARTIFACT, ClinicalClasses.ELECTRODE_ARTIFACT, ClinicalClasses.CHEWING, ClinicalClasses.SHIVERS, ClinicalClasses.ELECTRODE_POP],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=0, # individual for each recording
            channel_names=[], # individual for each recording
            preload=preload,
        )
        # fmt: on
        logging.info("in TUARDataset.__init__")
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
        if len(data_files) == 310:
            # If .edf files are found, we assume the dataset is already downloaded
            return

        # prompt the user to get the password and username from the website
        print("You need to get a password to download TUEG Artefact from https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")

        print("\nPlease enter your password:")
        
        os.system(f"rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_artifact/v3.0.1/edf/ {str(DATA_PATH)}")
        

    def load_data(self) -> None:
        self._download()
        
        self.data, self.labels, montage_type = self.cache.cache(_load_data_tuar)(self.subjects, self._preload) # type: ignore
        self.meta["montage_type"] = montage_type

    def get_data(self) -> Tuple[List[List[Raw]], List[str], Dict]:
        if not hasattr(self, "data") or self.data is None or not hasattr(self, "labels") or self.labels is None or self.meta is None:
            self.load_data()
        return self.data, self.labels, self.meta
