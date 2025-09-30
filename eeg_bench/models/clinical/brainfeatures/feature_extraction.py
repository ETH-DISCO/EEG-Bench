from .feature_generation.generate_features import generate_features_of_one_file
from .preprocessing.preprocess_raw import preprocess_one_file
from multiprocessing import Pool
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import logging
import time

def process_one_epilepsy_file_wrapper(args):
    raw, montage, domains, feature_generation_params = args
    return process_one_epilepsy_file(raw, montage, domains, **feature_generation_params)

def process_one_epilepsy_file(raw, montage, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap):
    
    # select the channels
    wanted_chs = sorted(['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'])
    if "le" in montage:
        ch_name_pattern="EEG {}-LE"
    else:
        ch_name_pattern="EEG {}-REF"
    if ch_name_pattern is not None:
        chs = [ch_name_pattern.format(ch) for ch in wanted_chs]
    else:
        chs = wanted_chs
    raw = raw.reorder_channels(chs)
    # get the sfreq
    sfreq = raw.info["sfreq"]
    # get the signals
    signals = raw.get_data(units="uV")
    if np.min(signals) < -800 or np.max(signals) > 800:
        signals = signals * 1e-1

    # do the preprocessing
    signals, sfreq, _ = preprocess_one_file(
        signals=signals,
        fs=sfreq,
        target=None,
        sec_to_cut_start=60,
        sec_to_cut_end=0,
        duration_recording_mins=20,
        resample_freq=100,
        max_abs_val=max_abs_val,
        clip_before_resample=False)
    
    # generate the features
    signals = pd.DataFrame(signals, index=chs)

    features_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)
    # features has shape (1, n_features)
    if features_df is None:
        logging.error("feature generation failed")
        return
    return features_df.to_numpy()

def process_one_abnormal_file_wrapper(args):
    raw, domains, feature_generation_params = args
    return process_one_abnormal_file(raw, domains, **feature_generation_params)

def process_one_abnormal_file(raw, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap):
    
    # select the channels
    wanted_chs = sorted(['C3', 'C4', 'CZ', 'F3', 'F4', 'F7', 'F8', 'FP1', 'FP2', 'FZ', 'O1', 'O2', 'P3', 'P4', 'PZ', 'T3', 'T4', 'T5', 'T6'])
    ch_name_pattern="EEG {}-REF"
    if ch_name_pattern is not None:
        chs = [ch_name_pattern.format(ch) for ch in wanted_chs]
    else:
        chs = wanted_chs
    raw = raw.reorder_channels(chs)
    # get the sfreq
    sfreq = raw.info["sfreq"]
    # get the signals
    signals = raw.get_data(units="uV")
    # do the preprocessing
    signals, sfreq, _ = preprocess_one_file(
        signals=signals,
        fs=sfreq,
        target=None,
        sec_to_cut_start=60,
        sec_to_cut_end=0,
        duration_recording_mins=20,
        resample_freq=100,
        max_abs_val=max_abs_val,
        clip_before_resample=False)
    
    # generate the features
    signals = pd.DataFrame(signals, index=chs)

    features_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)
    # features has shape (1, n_features)
    if features_df is None:
        logging.error("feature generation failed")
        return
    return features_df.to_numpy()

def process_one_file_wrapper(args):
    signals, sfreq, domains, feature_generation_params = args
    return process_one_file(signals, sfreq, domains, **feature_generation_params)

def process_one_file(signals, sfreq, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap):
    
    features_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)
    # features has shape (1, n_features)
    if features_df is None:
        logging.error("feature generation failed")
        return
    return features_df.to_numpy()
    

def _prepare_data_cached(X, meta, channels, domains, feature_generation_params, n_jobs):
    all_data = []
    for data, m in zip(X, meta):
        dataset_name = m['name']
        ch_names = m['channel_names']
        ch_names = [ch.upper() for ch in ch_names]
        channel_indices = [ch_names.index(ch.upper()) for ch in channels]
        new_channels = [ch_names[i] for i in channel_indices]
        data = [pd.DataFrame(d[channel_indices, :], index=new_channels) for d in data]
        with Pool(n_jobs) as pool:
            sfreq = m['sampling_frequency']
            parameters = [(signals, sfreq, domains, feature_generation_params) for signals in data]

            processed_signals = list(tqdm(pool.imap(process_one_file_wrapper, parameters), total=len(parameters), desc=f"Processing signals of {dataset_name}"))
        processed_signals = np.vstack(processed_signals)
        all_data.append(processed_signals)
    all_data = np.vstack(all_data)
    return all_data

def _prepare_abnormal_data_cached(X, meta, channels, domains, feature_generation_params, n_jobs):
    # in tueg we only have one dataset
    data = X[0] # data is now a list of BaseRaw objects (for every file one object)
    dataset_name = meta[0]["name"]

    with Pool(n_jobs) as pool:
        parameters = [(raw, domains, feature_generation_params) for raw in data]
        processed_signals = list(tqdm(pool.imap(process_one_abnormal_file_wrapper, parameters), total=len(parameters), desc=f"Processing signals of {dataset_name}"))
    
    features = np.vstack(processed_signals)
    return features

def _prepare_epilepsy_data_cached(X, meta, channels, domains, feature_generation_params, n_jobs):
    # in tueg we only have one dataset
    data = X[0] # data is now a list of BaseRaw objects (for every file one object)
    dataset_name = meta[0]["name"]
    montage_types = meta[0]["montage_type"]

    with Pool(n_jobs) as pool:
        parameters = [(raw, m ,domains, feature_generation_params) for raw, m in zip(data, montage_types)]
        processed_signals = list(tqdm(pool.imap(process_one_epilepsy_file_wrapper, parameters), total=len(parameters), desc=f"Processing signals of {dataset_name}"))
    
    features = np.vstack(processed_signals)
    return features