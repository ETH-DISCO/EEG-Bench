from .batcher.base import EEGDataset
from ...LaBraM.utils_2 import map_label
import numpy as np
from tqdm import tqdm
from resampy import resample
from mne.filter import filter_data, notch_filter
from sklearn.model_selection import train_test_split
from joblib import Memory
from .....config import get_config_value
import os


def preprocess_trials(trial, meta, required_channels):
    # reorder channels and select subset
    #print(trial.shape) # (256, 22, 1001)
    resampling_rate = 250
    chs = meta["channel_names"]
    channel_indices = []
    for ch in required_channels:
        if ch in chs:
            channel_indices.append(chs.index(ch))
        else:
            channel_indices.append(None)

    trial_data = []
    #print(channel_indices.count(None)) # 0
    for idx, ch in zip(channel_indices, required_channels):
        if idx is not None:
            trial_data.append(trial[:, idx, :])  # Select the data for that channel
        else:
            trial_data.append(np.zeros((trial.shape[0], trial.shape[2])))  # Shape (n_samples, n_timepoints)

    trial = np.array(trial_data).transpose(1, 0, 2)  # Shape (n_samples, n_channels, n_timepoints)

    # resample to 250Hz
    sampling_rate = meta["sampling_frequency"]
    trial = resample(trial, sampling_rate, resampling_rate, axis=-1, filter='kaiser_best')
    trial = filter_data(trial, sfreq=sampling_rate, l_freq=0.5, h_freq=75.0, method='fir', verbose=False)
    trial = notch_filter(trial, Fs=sampling_rate, freqs=50, verbose=False)

    original_length = trial.shape[-1]

    current_seconds = int(np.ceil(original_length / resampling_rate))
    if current_seconds % 2 == 1:
        target_seconds = current_seconds + 1
    else:
        target_seconds = current_seconds

    target_length = target_seconds * resampling_rate
    pad_width = target_length - original_length

    if pad_width > 0:
        trial = np.pad(trial, pad_width=((0, 0), (0, 0), (0, pad_width)), mode="constant")
    
    trial = map2pret(trial)
    
    return trial

def map2pret(data):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(current_dir, "..", "inputs", "tMatrix_value.npy")
    P = np.load(relative_path)
    return np.matmul(P, data)  # 22x22, 22xTime

class NeuroGPTDataset2(EEGDataset):
    def __init__(self, trials, labels, meta, task_name, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True, index_map=None):
        super().__init__([], sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.target_channels = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
        
        if index_map is not None:
            self.trials = trials
            self.labels = labels
            self.index_map = index_map
        else:
        
            if labels is None:
                self.labels = None
            else:
                self.labels = [[map_label(l, task_name) for l in ls] for ls in labels if len(ls) > 0]
                    
            self.cache = Memory(location=get_config_value("cache"), verbose=0)
            self.trials = [self.cache.cache(preprocess_trials)(trial, m, self.target_channels) 
                        for trial, m in tqdm(zip(trials, meta), total=len(trials)) 
                        if len(trial) > 0]
            self.index_map = [
                (trial_idx, sample_idx)
                for trial_idx, trial in enumerate(self.trials)
                for sample_idx in range(len(trial))
            ]

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        trial_idx, sample_idx = self.index_map[idx]
        data = self.trials[trial_idx][sample_idx]
        if self.labels is None:
            label = None
        else:
            label = self.labels[trial_idx][sample_idx]
        return self.preprocess_sample(data, self.num_chunks, label)            
            
    def split_train_val(self, val_split=0.1, seed=42):
        """
        Split the dataset into training and validation sets.
        Args:
            val_split (float): Fraction of the dataset to use for validation.
        Returns:
            Tuple[NeuroGPTDataset2, NeuroGPTDataset2]: Training and validation dataset instances.
        """

        idx_train, idx_val = train_test_split(
            np.arange(len(self.index_map)),
            test_size=val_split,
            random_state=seed,
            shuffle=True,
        )
        
        train_dataset = NeuroGPTDataset2(self.trials, self.labels, None, "", self.sample_keys, chunk_len=self.chunk_len, num_chunks=self.num_chunks, ovlp=self.ovlp, root_path="", gpt_only=self.gpt_only, index_map=[self.index_map[i] for i in idx_train])
        val_dataset = NeuroGPTDataset2(self.trials, self.labels, None, "", self.sample_keys, chunk_len=self.chunk_len, num_chunks=self.num_chunks, ovlp=self.ovlp, root_path="", gpt_only=self.gpt_only, index_map=[self.index_map[i] for i in idx_val])
        
        return train_dataset, val_dataset