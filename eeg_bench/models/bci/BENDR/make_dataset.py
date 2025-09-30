from .bendr_datasets import BENDRBCIDataset
from ..LaBraM.utils_2 import map_label, n_unique_labels
import numpy as np
from typing import List, Tuple, Optional, cast
from resampy import resample
from mne.filter import filter_data, notch_filter
from mne.io import BaseRaw
from tqdm import tqdm
import gc
import os
import pickle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

channel_mapping = { "FP1": ["FP1", "FZ"],
                    "FP2": ["FP2", "FZ"],
                    "F7": ["F7", "FC3"],
                    "F3": ["F3", "FC1"],
                    "FZ": ["FZ", "FCZ"],
                    "F4": ["F4", "FC2"],
                    "F8": ["F8", "FC4"],
                    "T7": ["T7", "FT7", "TP7", "T3", "C5"],
                    "C3": ["C3"],
                    "CZ": ["CZ"],
                    "C4": ["C4"],
                    "T8": ["T8", "FT8", "TP8", "T4", "C6"],
                    "P7": ["P7", "CP3", "T5"],
                    "P3": ["P3", "CP1"],
                    "PZ": ["PZ", "CPZ"],
                    "P4": ["P4", "CP2"],
                    "P8": ["P8", "CP4", "T6"],
                    "O1": ["O1", "P1", "POZ"],
                    "O2": ["O2", "P2", "POZ"]}

def make_dataset(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, l_freq: float = 0.1, 
                 h_freq: float = 75.0, train: bool = True) -> BENDRBCIDataset:

    print("\ndata shape: ", data.shape)
    if len(data) == 0:
        if train:
            return BENDRBCIDataset(data, labels, sampling_rate, ch_names), BENDRBCIDataset(data, labels, sampling_rate, ch_names)
        else:
            return BENDRBCIDataset(data, labels, sampling_rate, ch_names)
    
    reorder_channels = []
    new_ch_names = []
    ch_names = [ch.upper() for ch in ch_names]
    print("ch_names: ", ch_names)
    for key, value in channel_mapping.items():
        if key in ch_names:
            reorder_channels.append(ch_names.index(key))
            new_ch_names.append(key)
        else:
            found = False
            for v in value:
                if v in ch_names:
                    reorder_channels.append(ch_names.index(v))
                    new_ch_names.append(v)
                    found = True
                    break
            if not found:
                reorder_channels.append(len(ch_names))
                new_ch_names.append("0")
                data = np.insert(data, len(ch_names), 0, axis=1)
                print(f"Channel {key} not found")
    target_channels = new_ch_names
            
    data = data[:, reorder_channels, :]
    print("reorder_channels: ", new_ch_names)

    # bandpass filter
    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    # notch filter
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    # resample data
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    
    # Extend data to have a whole number of seconds by padding with zeros
    n_samples = data.shape[2]
    n_seconds = 4 # np.ceil(n_samples / target_rate).astype(int)
    new_n_samples = n_seconds * target_rate
    if new_n_samples > n_samples:
        padding = new_n_samples - n_samples
        data = np.pad(data, ((0, 0), (0, 0), (0, padding)), mode='constant', constant_values=0)
    elif new_n_samples < n_samples:
        data = data[:, :, :new_n_samples]

    # One hot encode labels if they are not None
    if labels is not None:
        labels = np.array([map_label(label, task_name) for label in labels])
        labels = np.eye(n_unique_labels(task_name))[labels]
        print("labels shape: ", labels.shape)  
    if train:
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.1, random_state=42)
        return BENDRBCIDataset(data_train, labels_train, target_rate, target_channels), BENDRBCIDataset(data_val, labels_val, target_rate, target_channels)
    else:
        return BENDRBCIDataset(data, labels, target_rate, target_channels)
        