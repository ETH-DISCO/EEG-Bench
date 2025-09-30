import pandas as pd
import numpy as np
import re
from glob import glob
from brainfeatures.data_set.abstract_data_set import DataSet
from scipy.io import loadmat
import mne
from resampy import resample
import fnmatch as fm
import warnings
import mat73 # type: ignore

class FeaturesDataSet(DataSet):
    """
    Dataset class for all CLI_UNM EEG datasets.
    """
    def __init__(self, data_path, extension=".h5", subset=None,
                 key="natural", n_recordings=None, datasets=None,
                 target="label", max_recording_mins=None, tfreq=100):
        self.max_recording_mins = max_recording_mins
        self.extension = extension
        self.data_path = data_path
        assert target in ["pd", "sc", "dep", "ocd", "mtbi", "bdi", "oci", "age", "sex"], "target was " + target # TODO
        assert datasets is not None, "datasets must be specified"
        self.datasets = datasets
        self.target = target
        assert subset in ["train", "eval"], "subset must be 'train' or 'eval'"
        self.subset = subset
        self.key = key
        self.tfreq = tfreq
        
        self.subject_ids = []
        self.file_paths = []
        self.targets = []

        if not extension == ".h5":
            raise ValueError("Only extracted and aggregated features allowed.")

        assert data_path.endswith("/"), "data path must end with '/'"
        assert extension.startswith("."), "file extension must start with '.'"

    def load(self):
        """
        Load metadata and file paths for the dataset.
        """
        print("\nLoading data...")
        for dataset in self.datasets:
            self.file_paths.extend(glob(self.data_path + dataset + "/" + self.subset +"/*" + self.extension, recursive=False))
            print(f"Found {len(self.file_paths)} files in {self.data_path + dataset + '/' + self.subset}")
            if len(self.file_paths) == 0:
                raise ValueError(f"No files with extension {self.extension} found in {self.data_path}.")
            # file_names contains all file paths to either a cli unm dataset or some train/eval data

        for file_path in self.file_paths:            
            targets_df = pd.read_hdf(file_path, key="targets")
            assert len(targets_df) == 1, "too many rows in targets df"
            targets = targets_df.iloc[-1].to_dict()
            self.targets.append(targets)

            info_df = pd.read_hdf(file_path, key="info")
            assert len(info_df) == 1, "too many rows in info df"
            info = info_df.iloc[-1].to_dict()
            sfreq = info["sfreq"]
            self.subject_ids.append(info["subject_id"])
            assert sfreq == self.tfreq, f"sfreq {sfreq} does not match target frequency {self.tfreq}"
        
        assert len(self.file_paths) == len(self.targets), "lengths differ"

    def __getitem__(self, index):
        """
        Returns a single data sample.
        """
        file_name = self.file_paths[index]
        if self.target not in self.targets[index]:
            print(f"target {self.target} not found in {self.targets[index]}")
            print(f"file_name: {file_name}")
            target = 0
        else:
            target = self.targets[index][self.target]
        signals = pd.read_hdf(file_name, key="data")

        return signals, self.tfreq, target

    def __len__(self):
        """
        Returns the total number of examples in the dataset.
        """
        return len(self.file_paths)
