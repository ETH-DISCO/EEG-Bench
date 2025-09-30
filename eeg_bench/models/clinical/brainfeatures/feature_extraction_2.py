from .feature_generation.generate_features import generate_features_of_one_file
from .preprocessing.preprocess_raw import preprocess_one_file
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import logging
import time

def extract_features(signals, label, t_channels, sfreq, feature_generation_params):
    epoch_duration_s = feature_generation_params["epoch_duration_s"]
    max_abs_val = feature_generation_params["max_abs_val"]
    window_name = feature_generation_params["window_name"]
    band_limits = feature_generation_params["band_limits"]
    agg_mode = feature_generation_params["agg_mode"]
    discrete_wavelet = feature_generation_params["discrete_wavelet"]
    continuous_wavelet = feature_generation_params["continuous_wavelet"]
    band_overlap = feature_generation_params["band_overlap"]
    domains = feature_generation_params["domains"]
    chunk_len_s = feature_generation_params["chunk_len_s"]
    resample_freq = feature_generation_params["resample_freq"]
    cutoff_start = feature_generation_params["cutoff_start_s"]
    cutoff_end = feature_generation_params["cutoff_end_s"]
    max_recording_len_min = feature_generation_params["max_recording_len_min"]
    
    signals, new_freq, _ = preprocess_one_file(
        signals=signals,
        fs=sfreq,
        target=None,
        sec_to_cut_start=cutoff_start,
        sec_to_cut_end=cutoff_end,
        duration_recording_mins=max_recording_len_min,
        resample_freq=resample_freq,
        max_abs_val=max_abs_val-1,
        clip_before_resample=False)
        
    if chunk_len_s is None:
        # generate the features
        signals = pd.DataFrame(signals, index=t_channels)
        features_df = generate_features_of_one_file(
            signals, new_freq, epoch_duration_s, max_abs_val, window_name,
            band_limits, agg_mode, discrete_wavelet,
            continuous_wavelet, band_overlap, domains)
        # features has shape (1, n_features)
        if features_df is None:
            logging.error("feature generation failed")
            return
        return features_df.to_numpy(), label
    else:
        chunk_len = int(chunk_len_s * new_freq)
        n_chunks = int(np.floor(signals.shape[1] / chunk_len))
        features = []
        for i in range(n_chunks):
            chunk = signals[:, i*chunk_len:(i+1)*chunk_len]
            chunk = pd.DataFrame(chunk, index=t_channels)
            features_df = generate_features_of_one_file(
                chunk, new_freq, epoch_duration_s, max_abs_val, window_name,
                band_limits, agg_mode, discrete_wavelet,
                continuous_wavelet, band_overlap, domains)
            if features_df is None:
                logging.error("feature generation failed")
                return
            features.append(features_df.to_numpy())
        if label is not None:
            if "is_multilabel_task" in feature_generation_params and feature_generation_params["is_multilabel_task"]:
                num_labels_per_chunk = label.shape[-1] // n_chunks
                labels = np.array(label)[:num_labels_per_chunk*n_chunks].reshape((n_chunks, -1))
            else:
                labels = np.repeat(label, n_chunks)
        else:
            labels = np.array(n_chunks)
        return np.vstack(features), labels

def process_one_abnormal(parameters):
    raw, label, t_channels, feature_generation_params = parameters

    ch_name_pattern="EEG {}-REF"
    chs = [ch_name_pattern.format(ch) for ch in t_channels]
    raw = raw.reorder_channels(chs)
    sfreq = raw.info["sfreq"]
    signals = raw.get_data(units="uV")
    
    features, labels = extract_features(signals, label, t_channels, sfreq, feature_generation_params)
    return features, labels


def process_one_epilepsy(parameters):
    raw, label, montage, t_channels, feature_generation_params = parameters
    if "le" in montage:
        ch_name_pattern="EEG {}-LE"
    else:
        ch_name_pattern="EEG {}-REF"
    chs = [ch_name_pattern.format(ch) for ch in t_channels]
    raw = raw.reorder_channels(chs)
    sfreq = raw.info["sfreq"]
    signals = raw.get_data(units="uV")

    features, labels = extract_features(signals, label, t_channels, sfreq, feature_generation_params)
    return features, labels

def process_one_multilabel(parameters):
    raw, label, t_channels, feature_generation_params = parameters
    raw = raw.reorder_channels(t_channels)
    sfreq = raw.info["sfreq"]
    signals = raw.get_data(units="uV")
    features, labels = extract_features(signals, label, t_channels, sfreq, feature_generation_params)
    return features, labels

def process_one_cli_unm(parameters):
    data, label, o_channels, t_channels, sfreq, feature_generation_params = parameters
    channel_indices = [o_channels.index(ch.upper()) for ch in t_channels]
    signals = data[channel_indices, :]

    features, labels = extract_features(signals, label, t_channels, sfreq, feature_generation_params)
    return features, labels



def _prepare_data_cached(X, y, meta, dataset_name, feature_generation_params):
    
    # have three different cases: normal, abnormal, epilepsy
    # with each of them their own process function
    task_name = meta[0]["task_name"]

    if "Abnormal" in dataset_name:
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
    elif "Epilepsy" in dataset_name:
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
    elif task_name == "parkinsons_clinical":
        t_channels = ['FP2', 'CP2', 'P2', 'CP4', 'AF8', 'P7', 'F7', 'F4', 'P3', 'O1', 'AF4', 'POZ', 'P6', 'OZ', 'C1', 'P1', 'AFZ', 'P5', 'PO7', 'F8', 'FZ', 'F1', 'TP8', 'FT8', 'PO8', 'C2', 'P4', 'F5', 'CP3', 'FC2', 'C4', 'CZ', 'F3', 'FC3', 'CP5', 'FP1', 'P8', 'T8', 'FT7', 'C3', 'T7', 'C5', 'FC5', 'O2', 'FCZ', 'FC6', 'FC1', 'FC4', 'AF7', 'AF3', 'CP6', 'C6', 'F2', 'F6', 'TP7', 'CP1']
    elif task_name == "schizophrenia_clinical":
        t_channels = ['Fp1', 'Fz', 'F3', 'F7', 'AFp9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'AFp10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'F6', 'F2', 'AF4', 'AF8']
    elif task_name == "mtbi_clinical":
        t_channels = ['Fp1', 'Fz', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8', 'CPz']
    elif task_name == "ocd_clinical":
        t_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    elif task_name == "seizure_clinical":
        t_channels = ["F3-C3", "F8-T8", "F4-C4", "FP1-F7", "P4-O2", "P8-O2", "T8-P8", "FZ-CZ", "P7-O1", "T7-P7", "FP2-F8", "CZ-PZ", "FP1-F3", "C3-P3", "C4-P4", "P3-O1", "F7-T7",  "FP2-F4"]
    elif task_name == "sleep_stages_clinical":
        t_channels = ['FPZ-CZ', 'PZ-OZ', 'EOGh']
    elif task_name in ["binary_artifact_clinical", "multiclass_artifact_clinical"]:
        t_channels = ['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6']
    else:
        t_channels = ['C4', 'FC3', 'P6', 'O1', 'CP4', 'C5', 'PO7', 'TP7', 'F4', 'P3', 'CP6', 'C3', 'FC4', 'F5', 'FC5', 'CP2', 'F2', 'P2', 'P5', 'F8', 'CP1', 'FC1', 'C6', 'F7', 'C2', 'T7', 'FCZ', 'CZ', 'AF3', 'FC6', 'F6', 'TP8', 'CP5', 'P7', 'O2', 'F1', 'FC2', 'FZ', 'F3', 'P8', 'C1', 'P4', 'POZ', 'T8', 'PO8', 'AF4', 'P1', 'OZ', 'CP3']

    n_jobs = os.cpu_count()
    if n_jobs < 1:
        n_jobs = 1
    
    if "Abnormal" in dataset_name:
        X = X[0]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        parameters = [(raw, label, t_channels, feature_generation_params) for raw, label in zip(X, y)]
        with Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap(process_one_abnormal, parameters), total=len(parameters), desc="Processing abnormal data"))
        all_data, all_labels = zip(*results)
        all_data = np.vstack(all_data)
        all_labels = np.hstack(all_labels)
    elif "Epilepsy" in dataset_name:
        X, montage_types = X[0], meta[0]["montage_type"]
        if y is None:
            y = [None] * len(X)
        else:
            y = y[0]
        parameters = [(raw, label, montage, t_channels, feature_generation_params) for raw, label, montage in zip(X, y, montage_types)]
        with Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap(process_one_epilepsy, parameters), total=len(parameters), desc="Processing epilepsy data"))
        all_data, all_labels = zip(*results)
        all_data = np.vstack(all_data)
        all_labels = np.hstack(all_labels)
    elif dataset_name in ["CHB-MIT", "TUAR", "Sleep-Telemetry"]:
        all_data = []
        all_labels = []
        if y is None:
            y = [None] * len(X)
        for data, labels, m in zip(X, y, meta):
            if labels is None:
                labels = [None] * len(data)
            parameters = [(raw, label, t_channels, feature_generation_params) for raw, label in zip(data, labels)]
            with Pool(n_jobs-1) as pool:
                results = list(tqdm(pool.imap(process_one_multilabel, parameters), total=len(parameters), desc=f"Processing {dataset_name}"))
            if len(results) == 0:
                continue
            tmp_data, tmp_labels = zip(*results)
            all_data.extend(tmp_data)
            all_labels.extend(tmp_labels)
        all_data = np.vstack(all_data)
        all_labels = np.vstack(all_labels)
    else:
        all_data = []
        all_labels = []
        if y is None:
            y = [None] * len(X)
        for data, labels, m in zip(X, y, meta):
            if labels is None:
                labels = [None] * len(data)
            sfreq = m['sampling_frequency']
            dataset_name = m['name']
            o_channels = m['channel_names']
            o_channels = [ch.upper() for ch in o_channels]
            parameters = [(signals, label, o_channels, t_channels, sfreq, feature_generation_params) for signals, label in zip(data, labels)]
            with Pool(3) as pool:
                results = list(tqdm(pool.imap(process_one_cli_unm, parameters), total=len(parameters), desc=f"Processing {dataset_name}"))
            if len(results) == 0:
                continue
            tmp_data, tmp_labels = zip(*results)
            all_data.extend(tmp_data)
            all_labels.extend(tmp_labels)
        all_data = np.vstack(all_data)
        all_labels = np.hstack(all_labels)

    return all_data, all_labels
    