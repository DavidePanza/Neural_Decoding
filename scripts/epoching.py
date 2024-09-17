#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import mne

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.epoch_utils import get_csv_and_parse_rows, sort_data_im, dump_data

def process_eeg_data(subject_id):
    """
    Processes EEG data for a given subject by performing the following steps:
    1. Loading raw EEG data and extracting events.
    2. Filtering out unwanted events.
    3. Converting and processing behavior log files.
    4. Creating unique identifiers for non-distractor trials.
    5. Epoching the EEG data and applying filters.
    6. Saving the processed data to a file.

    Args:
        subject_id (int or str): The ID of the subject to process.
    
    Returns:
        None
    """

    # Define path to EEG data
    data_path = '/Users/davide/Documents/Work/github/EEG/data/raw/' 

    # Construct the filename for the EEG data
    eeg_filename = f'sub_{subject_id}.vhdr'
    eeg_file_path = os.path.join(data_path, eeg_filename)

    # Load the EEG data
    raw_data = mne.io.read_raw_brainvision(eeg_file_path, preload=True)

    # Extract events from the EEG data
    events, event_ids = mne.events_from_annotations(raw_data)

    # Filter out unwanted events
    excluded_event_codes = [253, 254, 255, 99999, 243, 244, 245, 246]
    events_to_keep = events[~np.isin(events[:, 2], excluded_event_codes)]

    # Set up for log file processing
    logfile_columns = [
        'block', 'seq', 'i', 'ix', 'fullset_idx', 'trig_ID', 'image_name', 'category', 
        'img_number', 'duration', 'actual_target_dur', 'actual_mask_dur', 'label', 
        'image_path', 'mask_path'
    ]

    # Define a mapping for image categories to numeric IDs
    category_mapping = {
        'bear': 0, 'elephant': 1, 'person': 2, 'car': 3, 'dog': 4, 'apple': 5, 
        'chair': 6, 'plane': 7, 'bird': 8, 'zebra': 9, 'distractor': np.nan
    }

    # Paths for behavior log files
    logfile_dir = '/Users/davide/Documents/Work/github/EEG/data/raw/trials_logs' 
    behavior_file_path = os.path.join(logfile_dir, f'img_{subject_id}.txt')
    parsed_logfile_path = os.path.join(logfile_dir, f'sub_{subject_id}_parsed.csv')

    # Convert .txt file to .csv 
    get_csv_and_parse_rows(behavior_file_path, logfile_columns, parsed_logfile_path)

    # Load and process the parsed log file
    log_data = pd.read_csv(parsed_logfile_path, na_values='NaN')
    log_data['category_n'] = log_data['category'].map(category_mapping)

    # Create a numeric label for categories
    category_codes, _ = pd.factorize(log_data['label'], sort=True)
    log_data['ncat'] = category_codes + 1

    # Define distractor trigger IDs
    distractor_trigger_ids = [243, 244, 245, 246]

    # Get indices of trials that are not distractors
    non_distractor_indices = [
        i for i in range(len(log_data))
        if log_data.loc[i, 'trig_ID'] not in distractor_trigger_ids
    ]

    # Extract relevant information for non-distractor trials
    sequence_indices = log_data.iloc[non_distractor_indices]['i']
    sequence_numbers = log_data.iloc[non_distractor_indices]['seq']

    # Create unique identifiers for each trial based on its attributes
    identifiers = []

    for idx in non_distractor_indices:
        label = log_data.loc[idx, 'ncat']
        category = log_data.loc[idx, 'category_n'].astype(int)
        duration = log_data.loc[idx, 'duration']
        trigger_id = log_data.loc[idx, 'img_number']
        unique_id = int(f"{label}{category}{duration}{trigger_id}")
        identifiers.append(unique_id)

    # Update event information with new identifiers
    updated_events = events_to_keep.copy()
    updated_events[:, 2] = identifiers

    # Set parameters for epoching
    highpass = 0.01
    lowpass = 40
    trial_window = [-0.05, 0.5]

    # Define file paths for epoched data
    output_path = '/Users/davide/Documents/Work/github/EEG/data/epoched/'
    epoch_label = 'eeg_epochs'
    noise_normalization = 'raw'
    epoch_filename = os.path.join(output_path, f'{epoch_label}_{noise_normalization}_s{subject_id}.pkl')

    # Get EEG channels
    eeg_channels = mne.pick_types(raw_data.info, eeg=True)

    # Apply filters
    raw_filtered = raw_data.copy().filter(l_freq=highpass, h_freq=lowpass, picks=eeg_channels)

    # Create epochs
    epochs = mne.Epochs(raw_filtered, updated_events, tmin=trial_window[0], tmax=trial_window[1], 
                        picks='eeg', baseline=(None, 0), preload=True, reject=None)

    # Extract data
    eeg_data = epochs.get_data()
    time_points = epochs.times
    event_ids = updated_events[:, 2]

    # Select and sort trials
    channels = epochs.ch_names
    sorted_data = sort_data_im(eeg_data, event_ids, channels, time_points, sequence_indices, sequence_numbers)

    # Save results
    dump_data(sorted_data, epoch_filename)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python epoching.py <subject_id>")
        sys.exit(1)

    subject_id = sys.argv[1]
    process_eeg_data(subject_id)