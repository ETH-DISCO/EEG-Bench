from .bendr_datasets import BENDRBCIDataset
import numpy as np
from typing import List, Tuple, Optional, cast, Dict
from resampy import resample
from mne.filter import filter_data, notch_filter
from mne.io import BaseRaw
from tqdm import tqdm
import gc
import os
import pickle
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from joblib import Memory
from ...config import get_config_value


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

    print("\ndata shape: ", [d.shape for d in data])
    if len(data) == 0:
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
            
    data = [d[reorder_channels, :] for d in data]

    # bandpass filter
    data = [filter_data(d.astype("float64"), sfreq=sampling_rate, l_freq=l_freq, h_freq=h_freq, method='fir', verbose=False) for d in data]
    # notch filter
    data = [notch_filter(d, Fs=sampling_rate, freqs=50, verbose=False) for d in data]
    # resample data
    data = [resample(d.astype("float32"), sampling_rate, target_rate, axis=2, filter='kaiser_best') for d in data]

    """
    target_duration = 301
    prepared_data = []
    for d in data:
        duration = d.shape[1] / target_rate
        if duration < target_duration:
            pad = np.zeros((d.shape[0], target_rate*target_duration - d.shape[1]))
            d = np.concatenate((d, pad), axis=1)
        elif duration > target_duration:
            d = d[:, :target_rate*target_duration]
        prepared_data.append(d[:, target_rate:])
    data = np.stack(prepared_data, axis=0)
    print("data shape: ", data.shape)
    """

    # One hot encode labels if they are not None
    if labels is not None:
        if task_name == "parkinsons_clinical":
            label_mapping = {'parkinsons': 0, 'no_parkinsons': 1}
        elif task_name == "schizophrenia_clinical":
            label_mapping = {'schizophrenia': 0, 'no_schizophrenia': 1}
        elif task_name == "depression_clinical":
            label_mapping = {'depression': 0, 'no_depression': 1}
        elif task_name == "mtbi_clinical":
            label_mapping = {True: 0, False: 1}
        elif task_name == "ocd_clinical":
            label_mapping = {'ocd': 0, 'no_ocd': 1}
        else:
            raise ValueError("Invalid task name")
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(label_mapping))[labels]
        print("labels shape: ", labels.shape)  
    
    return BENDRBCIDataset(data, labels, target_rate, target_channels)
        
def process_one_file(args):
    raw, target_rate, l_freq, h_freq = args

    # get the signals
    raw.load_data()
    raw.set_eeg_reference("average")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.notch_filter(50.0)
    raw.resample(target_rate, n_jobs=1)

    signals = raw.get_data(units="uV")
    ch_names = raw.info["ch_names"]

    reorder_channels = []
    new_ch_names = []
    ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]
    #print("ch_names: ", ch_names)
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
                signals = np.insert(signals, len(ch_names), 0, axis=1)
                print(f"Channel {key} not found")
            
    signals = signals[reorder_channels, :]

    return signals

def process_one_file_epilepsy(args):
    raw, m, target_rate, l_freq, h_freq = args

    # get the signals
    raw.load_data()
    raw.set_eeg_reference("average")
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    raw.notch_filter(50.0)
    raw.resample(target_rate, n_jobs=1)

    signals = raw.get_data(units="uV")
    ch_names = raw.info["ch_names"]

    reorder_channels = []
    new_ch_names = []
    ch_names = [ch.upper()[4:].split('-')[0] for ch in ch_names]
    #print("ch_names: ", ch_names)
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
                signals = np.insert(signals, len(ch_names), 0, axis=1)
                print(f"Channel {key} not found")
            
    signals = signals[reorder_channels, :]

    return signals

def prepare_cached_data(X, target_rate, l_freq, h_freq, n_jobs):
    parameters = [(X[i], target_rate, l_freq, h_freq) for i in range(len(X))]
    with Pool(n_jobs) as pool:
        processed_signals = list(tqdm(pool.imap(process_one_file, parameters), total=len(parameters)))
    return processed_signals

def prepare_cached_data_epilepsy(X, meta, target_rate, l_freq, h_freq, n_jobs):
    montage_types = meta["montage_type"]
    parameters = [(X[i], montage_types[i], target_rate, l_freq, h_freq) for i in range(len(X))]
    with Pool(n_jobs) as pool:
        processed_signals = list(tqdm(pool.imap(process_one_file_epilepsy, parameters), total=len(parameters)))
    return processed_signals

def make_dataset_abnormal(data: List[BaseRaw], labels: List[str]|None, task_name: str,
                      target_rate: int = 200, target_channels: Optional[List[str]] = None, 
                      l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, val_per: float = 0.2) -> BENDRBCIDataset:

    cache = Memory(location=get_config_value("cache"), verbose=0)
    data = cache.cache(prepare_cached_data)(data, target_rate, l_freq, h_freq, n_jobs=16)
    gc.collect()

    # One hot encode labels if they are not None
    if labels is not None:
        label_mapping = {'abnormal': 0, 'normal': 1}
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(label_mapping))[labels]
        print("labels shape: ", labels.shape)  
    
    return BENDRBCIDataset(data, labels, target_rate, target_channels)

def make_dataset_epilepsy(data: List[BaseRaw], labels: List[str]|None, meta: Dict, task_name: str,
                      target_rate: int = 200, target_channels: Optional[List[str]] = None, 
                      l_freq: float = 0.1, h_freq: float = 75.0, train: bool = True, val_per: float = 0.2) -> BENDRBCIDataset:

    cache = Memory(location=get_config_value("cache"), verbose=0)
    data = cache.cache(prepare_cached_data_epilepsy)(data, meta, target_rate, l_freq, h_freq, n_jobs=16)
    gc.collect()

    # One hot encode labels if they are not None
    if labels is not None:
        label_mapping = {'epilepsy': 0, 'no_epilepsy': 1}
        labels = np.vectorize(label_mapping.get)(labels)
        labels = np.eye(len(label_mapping))[labels]
        print("labels shape: ", labels.shape)  
    
    return BENDRBCIDataset(data, labels, target_rate, target_channels)
        