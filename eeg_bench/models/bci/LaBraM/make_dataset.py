from .labram_datasets import LaBraMBCIDataset
from .utils_2 import map_label, n_unique_labels
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
import logging

standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9H', 'TTP7H', 'TPP9H', 'FTT10H', 'TPP8H', 'TPP10H', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def make_dataset(data: np.ndarray, labels: np.ndarray|None, task_name: str, sampling_rate: int, 
                 ch_names: List[str], target_rate: int = 200, target_channels: Optional[List[str]] = None,
                 l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, split_size=0.1) -> LaBraMBCIDataset:
    """
    data: np.ndarray, shape=(n_trials, n_channels, n_samples)
    labels: np.ndarray, shape=(n_trials,)
    ch_names: List[str], list of channel names
    target_channels: List[str], list of target channel names
    sampling_rate: int, sampling rate of the data
    target_rate: int, target sampling rate
    l_freq: int, low cut-off frequency
    h_freq: int, high cut-off frequency
    """
    print("\ndata shape: ", data.shape)
    logging.info(f"data shape: {data.shape}")
    if len(data) == 0:
        if train:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names), LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
        else:
            return LaBraMBCIDataset(data, labels, sampling_rate, ch_names)
    # filter out the channels that are not in the target_channels
    if target_channels is not None:
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = [ch.upper() for ch in target_channels]
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]
    else:
        # target_channels = ch_names
        ch_names = [ch.upper() for ch in ch_names]
        target_channels = list(set([ch.upper() for ch in standard_1020]).intersection(set(ch_names)))
        data = data[:, [ch_names.index(ch) for ch in target_channels], :]

    # bandpass filter
    data = filter_data(data, sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False)
    # notch filter
    data = notch_filter(data, Fs=sampling_rate, freqs=50, verbose=False)
    # resample data
    data = resample(data, sampling_rate, target_rate, axis=2, filter='kaiser_best')
    
    logging.info(f"data shape after resampling: {data.shape}")
    # Extend data to have a whole number of seconds by padding with zeros or trimming
    n_samples = data.shape[2]
    n_seconds = np.floor(n_samples / target_rate).astype(int)
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
        data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=split_size, random_state=42)
        return LaBraMBCIDataset(data_train, labels_train, target_rate, target_channels), LaBraMBCIDataset(data_val, labels_val, target_rate, target_channels)
    else:
        return LaBraMBCIDataset(data, labels, target_rate, target_channels)
        
