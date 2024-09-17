import os
import pickle
import numpy as np


def get_data(sub, perm):
    """
    Loads and processes decoding data for different durations and stores mean values for each train/test pair.

    Args:
        sub (str): Subject identifier.
        perm (int): Permutation number.

    Returns:
        dict: A dictionary containing mean-decoding values for each duration (keys: '2', '6').
    """
    durations = ['2', '6']
    diags = {dur: [] for dur in durations}

    for dur in durations:
        path = '/Users/davide/Documents/Work/github/EEG/results/decoding/'
        file = f'dec_obj_subj{sub}_dur{dur}.pkl'
        load_path = os.path.join(path, file)

        with open(load_path, 'rb') as f:
            decoding_data = pickle.load(f)

        for i in range(2):
            for ii in range(2):
                data_array = np.array(list(decoding_data['DA'][i][ii]))
                mean_data = np.nanmean(data_array, axis=(0, 1))
                diags[dur].append(mean_data)

    return diags


def reshape_data(diag_list):
    """
    Reshapes the decoded data into subject-by-datapoint arrays for control and challenge conditions.

    Args:
        diag_list (list): List of dictionaries containing decoded data for multiple subjects.

    Returns:
        tuple: Two dictionaries with reshaped data for control and challenge conditions.
               Keys: '2' for 2s duration, '6' for 6s duration.
               Each dictionary contains a 2D array (n_subjects x n_datapoints).
    """
    n_datapoints = diag_list[0]['2'][0].shape[0]
    all_subj_con = {'2': [], '6': []}
    all_subj_chal = {'2': [], '6': []}
    durations = ['2', '6']

    for dur in durations:
        for datapoints in range(n_datapoints):
            act_dtpoint_cont = []
            act_dtpoint_chal = []
            for sub in range(len(diag_list)):
                act_dtpoint_cont.append(np.round(diag_list[sub][dur][3][datapoints], 3))
                act_dtpoint_chal.append(np.round(diag_list[sub][dur][0][datapoints], 3))
            all_subj_con[dur].append(act_dtpoint_cont)
            all_subj_chal[dur].append(act_dtpoint_chal)

        all_subj_con[dur] = np.array(all_subj_con[dur]).T
        all_subj_chal[dur] = np.array(all_subj_chal[dur]).T

    return all_subj_con, all_subj_chal
