import os
import pdb
import numpy as np
from .base import EEGDataset
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from resampy import resample


class BCIDataset(EEGDataset):
    def __init__(self, trials, labels, meta, task_name, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__([], sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.Fs = 250  # 250Hz from original paper
        self.P = np.load("./models/NeuroGPT/inputs/tMatrix_value.npy")

        self.channels = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
        
        if task_name == "Left Hand vs Right Hand MI":
            self.labels_string2int = {'left_hand': 0, 'right_hand': 1}
            self.labels_int2string = {0: 'left_hand', 1: 'right_hand'}
        elif task_name == "Right Hand vs Feet MI":
            self.labels_string2int = {'right_hand': 0, 'feet': 1}
            self.labels_int2string = {0: 'right_hand', 1: 'feet'}
        elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
            self.labels_string2int = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue':3 }
            self.labels_int2string = {0: 'left_hand', 1: 'right_hand', 2: 'feet', 3: 'tongue'}
        else:
            raise ValueError("Invalid task name")

        self.data_all = []
        if labels is None:
            self.labels = None
        else:
            self.labels = np.concatenate([self.encode_labels(l) for l in labels], axis=0)
                
        # print(len(trials)) # 1
        # print(trials[0].shape) (256, 22, 1001)
        #self.trials = trials[0][:, :, :1000]
        self.trials = np.concatenate([self.preprocess_trials(trial, m) for trial, m in zip(trials, meta)], axis=0)
        self.num_trials_per_sub = [len(trial) for trial in trials]
        print(self.trials.shape) # (x, 22, 100)

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.preprocess_sample(self.trials[idx], self.num_chunks, None)
        else:
            return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def preprocess_trials(self, trial, meta):
        # reorder channels and select subset
        print(trial.shape) # (256, 22, 1001)
        chs = meta["channel_names"]
        required_channels = self.channels
        channel_indices = []
        for ch in required_channels:
            if ch in chs:
                channel_indices.append(chs.index(ch))
            else:
                channel_indices.append(None)

        trial_data = []
        print(channel_indices.count(None)) # 0
        for idx, ch in zip(channel_indices, required_channels):
            if idx is not None:
                trial_data.append(trial[:, idx, :])  # Select the data for that channel
            else:
                trial_data.append(np.zeros((trial.shape[0], trial.shape[2])))  # Shape (n_samples, n_timepoints)

        trial = np.array(trial_data).transpose(1, 0, 2)  # Shape (n_samples, n_channels, n_timepoints)
        print(trial.shape) # (22, 256, 1001)    
        
        #trial = trial[:, [chs.index(ch) for ch in self.channels], :]

        # resample to 250Hz
        sampling_rate = meta["sampling_frequency"]
        trial = resample(trial, sampling_rate, self.Fs, axis=-1, filter='kaiser_best')
        # crop to 1000ms
        # trial has shape (256, 22, 750)
        if trial.shape[2] < 1000:
            trial = np.pad(trial, ((0, 0), (0, 0), (0, 1000-trial.shape[2])), 'constant')
        trial = trial[:, :, :1000]
        return trial

    def decode_predictions(self, labels):
        return np.array([self.labels_int2string[l] for l in labels])

    def encode_labels(self, labels):
        return np.array([self.labels_string2int[l] for l in labels])

class MotorImageryDataset(EEGDataset):
    def __init__(self, task_name, filenames, sample_keys, chunk_len=500, num_chunks=10, ovlp=50, root_path="", gpt_only=True):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path, gpt_only=gpt_only)

        self.data_all = []
        for fn in self.filenames:
            print(fn)
            self.data_all.append(np.load(fn))
        
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 1023: 'rejected'} # , 783: 'unknown', 1023: 'rejected'
        # Types of motor imagery
        if task_name == "Left Hand vs Right Hand MI":
            self.labels_string2int = {'left_hand': 0, 'right_hand': 1}
            self.labels_int2string = {0: 'left_hand', 1: 'right_hand'}
        elif task_name == "Right Hand vs Feet MI":
            self.labels_string2int = {'right_hand': 0, 'feet': 1}
            self.labels_int2string = {0: 'right_hand', 1: 'feet'}
        elif task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
            self.labels_string2int = {'left_hand': 0, 'right_hand': 1, 'feet': 2, 'tongue':3 }
            self.labels_int2string = {0: 'left_hand', 1: 'right_hand', 2: 'feet', 3: 'tongue'}
        else:
            raise ValueError("Invalid task name")
        self.task_name = task_name

        self.Fs = 250  # 250Hz from original paper
        self.P = np.load("./models/NeuroGPT/inputs/tMatrix_value.npy")

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()

        print(self.trials[0].shape) # (22, 1000)
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])

    def map2pret(self, data):
        return np.matmul(self.P, data) # 22x22, 22xTime
    
    def decode_predictions(self, labels):
        return np.array([self.labels_int2string[l] for l in labels])

    def get_trials_from_single_subj(self, sub_id):
        raw = self.data_all[sub_id]['s'].T
        events_type = self.data_all[sub_id]['etyp'].T
        events_position = self.data_all[sub_id]['epos'].T
        events_duration = self.data_all[sub_id]['edur'].T
        artifacts = self.data_all[sub_id]['artifacts'].T
        # Channel default is C3
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trial_labels = self.get_labels(sub_id)
        
        trials = []
        classes = []
        for j, index in enumerate(idxs):
            try:
                # print(index)
                # type_e = events_type[0, index+1]
                # class_e = self.mi_types[type_e]
                # if type_e == 1023:
                #     continue
                # classes.append(self.labels_string2int[class_e])
                label = trial_labels[j]
                if self.task_name == "Left Hand vs Right Hand MI":
                    if label == 2 or label == 3:
                        continue
                elif self.task_name == "Right Hand vs Feet MI":
                    if label == 0 or label == 3:
                        continue
                    elif label == 2:
                        label = 1
                    elif label == 1:
                        label = 0
                elif self.task_name == "Left Hand vs Right Hand vs Feet vs Tongue MI":
                    pass

                classes.append(label)

                start = events_position[0, index]
                stop = start + events_duration[0, index]
                trial = raw[:22, start+500 : stop-375]
                #add band-pass filter
                # self.bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=5)
                trials.append(trial)
            except:
                # print("Cannot load trial")
                continue
        return trials, classes

    def get_labels(self, sub_id):
        label_path = self.root_path + "true_labels/"
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        labels = loadmat(label_path + sub_name +".mat")["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
        # map to same channel configuration as pretraining
        #trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num
    
    # def normalize(self, data):
    #     return (data - np.mean(data)) / np.std(data)
    
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.
        
        Parameters:
        - data: The EEG signal
        - lowcut: Low cut-off frequency
        - highcut: High cut-off frequency
        - fs: Sampling rate (frequency)
        - order: Order of the filter
        
        Returns:
        - Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data