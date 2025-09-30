from .base_clinical_dataset import BaseClinicalDataset
from ...enums.clinical_classes import ClinicalClasses
from typing import Optional, Sequence, Tuple, List, Dict
import logging
import glob
import os
from mne.io import read_raw_edf, Raw
from tqdm import tqdm
from ...config import get_data_path
from pathlib import Path


DATA_PATH = get_data_path("tuep", "tuep")


def _load_data_tueg_epilepsy(subjects: Sequence[int], preload: bool) -> Tuple[List[List[Raw]], List[str], List[List[str]]]:
    """Loads EEG data from the TUEG Epilepsy dataset.
    
    This function retrieves EEG recordings of subjects with and without epilepsy. 
    Subjects are mapped to dataset files based on their index:
    - Even-numbered subjects have epilepsy.
    - Odd-numbered subjects do not have epilepsy.

    Args:
        subjects (Sequence[int]): A list of subject indices (1-indexed) to load data for.
        preload (bool): If True, preloads EEG data into memory.

    Returns:
        Tuple:
            - List[Raw]: A list containing `Raw` EEG recordings.
            - List[str]: A list containing labels (`"epilepsy"` or `"no_epilepsy"`) for each EEG recording.
            - List[str]: A list containing montage types for each EEG recording.
    """
    
    # even subjects have epilepsy, odd subjects don't have epilepsy
    # fmt: off
    epilepsy_id = ['aaaaaanr', 'aaaaaawu', 'aaaaabdn', 'aaaaabhz', 'aaaaabju', 'aaaaacrz', 'aaaaadkv', 'aaaaaelp', 'aaaaaeph', 'aaaaaeqq', 'aaaaaewf', 'aaaaafif', 'aaaaaflb', 'aaaaagnh', 'aaaaagxr', 'aaaaaicb', 'aaaaaiek', 'aaaaaifn', 'aaaaaifp', 'aaaaaiiz', 'aaaaaimz', 'aaaaaint', 'aaaaaiud', 'aaaaajat', 'aaaaajbn', 'aaaaajqo', 'aaaaajud', 'aaaaajus', 'aaaaakbt', 'aaaaakfe', 'aaaaakgl', 'aaaaakgy', 'aaaaaklt', 'aaaaaklv', 'aaaaakoq', 'aaaaakqa', 'aaaaakqq', 'aaaaaktn', 'aaaaakvr', 'aaaaakyc', 'aaaaalfg', 'aaaaalgi', 'aaaaalhr', 'aaaaalib', 'aaaaalim', 'aaaaaliw', 'aaaaalof', 'aaaaalon', 'aaaaalqm', 'aaaaalug', 'aaaaaluq', 'aaaaalzg', 'aaaaamde', 'aaaaamgc', 'aaaaamhb', 'aaaaamic', 'aaaaamkx', 'aaaaamlf', 'aaaaamlm', 'aaaaamnj', 'aaaaamoa', 'aaaaamor', 'aaaaamrq', 'aaaaamsw', 'aaaaamtl', 'aaaaamuy', 'aaaaamys', 'aaaaanag', 'aaaaandx', 'aaaaanfk', 'aaaaanfl', 'aaaaanhr', 'aaaaanjh', 'aaaaankz', 'aaaaanmf', 'aaaaannz', 'aaaaanog', 'aaaaanwn', 'aaaaanzp', 'aaaaaock', 'aaaaaocl', 'aaaaaodl', 'aaaaaogc', 'aaaaaokm', 'aaaaaoom', 'aaaaaooo', 'aaaaaoou', 'aaaaaotl', 'aaaaaoui', 'aaaaaovm', 'aaaaaowq', 'aaaaaozy', 'aaaaapfa', 'aaaaapge', 'aaaaaphz', 'aaaaapjn', 'aaaaapkm', 'aaaaapri', 'aaaaapvx', 'aaaaapwd']
    no_epilepsy_id = ['aaaaaebo', 'aaaaafiy', 'aaaaaibz', 'aaaaaigj', 'aaaaajgj', 'aaaaajgn', 'aaaaajoz', 'aaaaajpt', 'aaaaajrh', 'aaaaajrm', 'aaaaajsp', 'aaaaajxf', 'aaaaajxm', 'aaaaakbp', 'aaaaakcd', 'aaaaakfq', 'aaaaakgb', 'aaaaakim', 'aaaaakkb', 'aaaaakkg', 'aaaaakmb', 'aaaaakpl', 'aaaaakvz', 'aaaaaljb', 'aaaaallk', 'aaaaalpi', 'aaaaalpw', 'aaaaalqk', 'aaaaalsc', 'aaaaalxd', 'aaaaalyg', 'aaaaalze', 'aaaaamas', 'aaaaamaw', 'aaaaamba', 'aaaaamey', 'aaaaamfh', 'aaaaamhx', 'aaaaamlk', 'aaaaammn', 'aaaaammt', 'aaaaammw', 'aaaaamnf', 'aaaaamrw', 'aaaaamum', 'aaaaamwa', 'aaaaamwl', 'aaaaanca', 'aaaaandb', 'aaaaangg', 'aaaaanig', 'aaaaankd', 'aaaaanky', 'aaaaanla', 'aaaaannv', 'aaaaanon', 'aaaaanuz', 'aaaaanvy', 'aaaaanwp', 'aaaaaoay', 'aaaaaocx', 'aaaaaods', 'aaaaaoep', 'aaaaaofk', 'aaaaaohu', 'aaaaaoie', 'aaaaaojm', 'aaaaaokb', 'aaaaaooz', 'aaaaaore', 'aaaaaosn', 'aaaaaoti', 'aaaaaott', 'aaaaaovh', 'aaaaaoxn', 'aaaaaoxx', 'aaaaaoyh', 'aaaaaoyn', 'aaaaaoyt', 'aaaaapap', 'aaaaapfw', 'aaaaapgp', 'aaaaapkt', 'aaaaapkw', 'aaaaapld', 'aaaaaplq', 'aaaaapls', 'aaaaapmi', 'aaaaapmu', 'aaaaapnx', 'aaaaappj', 'aaaaappo', 'aaaaapre', 'aaaaapsj', 'aaaaapsm', 'aaaaapsq', 'aaaaapuc', 'aaaaapuo', 'aaaaapwy', 'aaaaaqaw']
    # fmt: on

    data = []
    labels = []
    montage_type = []
    for subject in tqdm(subjects):
        if subject % 2 == 0:
            subject_id = epilepsy_id[subject // 2 - 1]
            sub_folder = "00_epilepsy"
            label = "epilepsy"
        else:
            subject_id = no_epilepsy_id[subject // 2]
            sub_folder = "01_no_epilepsy"
            label = "no_epilepsy"

        files = glob.glob(os.path.join(DATA_PATH, sub_folder, subject_id, "**", "*.edf"), recursive=True)
        for file in files:
            raw = read_raw_edf(file, verbose="error", preload=preload)
            if raw.times[-1] < 60:
                raw.close()
                continue
            if len(raw.info["ch_names"]) < 19:
                raw.close()
                continue
            data.append(raw)
            labels.append(label)
            montage_type.append(os.path.basename(os.path.dirname(file)))
        
    return data, labels, montage_type


class TUEGEpilepsyDataset(BaseClinicalDataset):
    """
    The dataset contains EEG recordings of subjects with and without epilepsy.
        Subjects are mapped to the corresponding files in the dataset using the 
        following rule: even-numbered subjects have epilepsy, while odd-numbered 
        subjects do not.
    - Epilepsy: 100 Subjects with a total of 1785 Recordings
    - Control: 100 Subjects with a total of 513 Recordings
    """
    def __init__(
        self,
        target_class: ClinicalClasses,
        subjects: Sequence[int],
        target_channels: Optional[Sequence[str]] = None,
        target_frequency: Optional[int] = None,
        preload: bool = False,
    ):
        # fmt: off
        super().__init__(
            name="TUEG Epilepsy",
            target_classes=[target_class],
            available_classes=[ClinicalClasses.EPILEPSY],
            subjects=subjects,
            target_channels=target_channels,
            target_frequency=target_frequency,
            sampling_frequency=0, # individual for each recording
            channel_names=[], # individual for each recording
            preload=preload,
        )
        # fmt: on
        logging.info("in TUEGEpilepsyDataset.__init__")
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
        if len(data_files) == 2298:
            # If .edf files are found, we assume the dataset is already downloaded
            return

        # prompt the user to get the password and username from the website
        print("You need to get a password to download TUEG Epilepsy from https://isip.piconepress.com/projects/nedc/html/tuh_eeg/")

        print("\nPlease enter your password:")
        
        os.system(f"rsync -auxvL nedc-tuh-eeg@www.isip.piconepress.com:data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/ {str(DATA_PATH)}")
        

    def load_data(self) -> None:
        self._download()
        
        self.data, self.labels, montage_type = self.cache.cache(_load_data_tueg_epilepsy)(self.subjects, self._preload) # type: ignore
        self.meta["montage_type"] = montage_type

    def get_data(self) -> Tuple[List[List[Raw]], List[str], Dict]:
        """Get the data of the TUEG Epilepsy dataset.
        
        The dataset contains EEG recordings of subjects with and without epilepsy.
        Subjects are mapped to the corresponding files in the dataset using the 
        following rule: even-numbered subjects have epilepsy, while odd-numbered 
        subjects do not.
    
        Returns:
            Tuple:
                - List[Raws]: `Raw` EEG recordings.
                - List[str]: Labels (`"epilepsy"` or `"no_epilepsy"`) for each EEG recording.
                - Dict: Metadata containing montage types for each EEG signal.
        """

        if not hasattr(self, "data") or self.data is None or not hasattr(self, "labels") or self.labels is None or self.meta is None:
            self.load_data()
        return self.data, self.labels, self.meta