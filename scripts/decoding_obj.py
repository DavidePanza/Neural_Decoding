#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import scipy.linalg
import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import _cov

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.decoding_utils import (
    load_data,
    dump_data,
    downsample_data,
    get_pseudotrials_obj,
    my_ceil
)

def main():
    # =============================================================================
    # Constants and Parameters
    # =============================================================================

    # Define paths and parameters
    PATH = '/Users/davide/Documents/Work/github/EEG/data/epoched/'
    OUT_PATH = '/Users/davide/Documents/Work/github/EEG/results/decoding/'
    SUBJ = int(sys.argv[1])
    EPOCH_LABEL = 'eeg_epochs'
    PREPROCESSING_TYPE = 'raw'
    EPOCH_FILE = os.path.join(PATH, f'{EPOCH_LABEL}_{PREPROCESSING_TYPE}_s{SUBJ}.pkl')

    # Parameters for subsampling and conditions
    OBJ_SUBSAMPLE_FACTOR = 5
    CONDITIONS = {'challenge': '1', 'control': '2'}

    # Iteration parameters
    DURATION = str(sys.argv[2])
    N_PERMS = int(sys.argv[3])
    OUTPUT_LABEL = 'dec_obj'
    OUTPUT_NAME = os.path.join(OUT_PATH, f'{OUTPUT_LABEL}_subj{SUBJ}_dur{DURATION}.pkl')

    print(f'Object decoding for subject {SUBJ}')
    print(f'Saved as: {OUTPUT_NAME}')

    # =============================================================================
    # Load and Prepare Data
    # =============================================================================

    # Load and downsample data
    data = load_data(EPOCH_FILE)
    print('Data loaded')
    data = downsample_data(data, OBJ_SUBSAMPLE_FACTOR)

    # Subset Duration
    mask_dur = np.array(data['duration']) == DURATION

    # Apply masks
    data['eeg'] = data['eeg'][mask_dur, :, :, :]
    data['category'] = np.array(data['category'])[mask_dur]
    data['label'] = np.array(data['label'])[mask_dur]

    # Process data
    objs, objs_counts = np.unique(data['category'], return_counts=True)
    n_objects = len(objs)
    labels = list(CONDITIONS.keys())
    lab_masks = np.array([np.array(data['label']) == c for c in np.unique(data['label'])])
    eeg_cat = np.array((data['eeg'][lab_masks[0]], data['eeg'][lab_masks[1]]))
    obj_cat = np.array((np.array(data['category'])[lab_masks[0]], np.array(data['category'])[lab_masks[1]]))

    n_labels, n_imgs_cat, n_trials, n_sensors, n_time = eeg_cat.shape
    decoding_results = np.full((n_labels, n_labels, n_objects, n_objects, n_time), np.nan)

    # =============================================================================
    # Perform Decoding
    # =============================================================================

    TEST_RATIO = .1

    for p in tqdm.tqdm(range(N_PERMS)):
        pstrials, n_pstrials, binsize = get_pseudotrials_obj(eeg_cat, obj_cat)
        n_test = int(my_ceil(n_pstrials * TEST_RATIO))
        folds = int(n_pstrials / n_test)
        ps_ixs = np.arange(n_pstrials)

        if p == 0:
            print(f'Binsize: {binsize}, Number of pseudotrials: {n_pstrials}')

        for ci in range(n_labels):
            for cci in range(n_labels):
                for cv in range(folds):
                    print(f'Training on: {labels[ci]}, Testing on: {labels[cci]}, CV round: {cv}')

                    test_ix = np.arange(n_test) + (cv * n_test)
                    train_ix = np.delete(ps_ixs.copy(), test_ix)

                    ps_train = pstrials[ci][:, :, train_ix, :, :]
                    ps_test = pstrials[cci][:, :, test_ix, :, :]

                    # Shuffle training trials across pseudotrials
                    n_objects_, binsize_, n_pseudotrials_, n_channels_, n_timepoints_ = ps_train.shape
                    ps_train = np.reshape(ps_train, (n_objects_, binsize_ * n_pseudotrials_, n_channels_, n_timepoints_))
                    ps_train = ps_train[:, np.random.permutation(ps_train.shape[1]), :, :]
                    ps_train = np.reshape(ps_train, (n_objects_, binsize_, n_pseudotrials_, n_channels_, n_timepoints_))

                    # Compute average covariance matrix (sigma) for each object, whitening the data
                    sigma_ = np.empty((n_objects, n_sensors, n_sensors))
                    for c in range(n_objects):
                        sigma_[c] = np.mean([
                            _cov(np.reshape(ps_train[c, :, :, :, t], (len(train_ix) * binsize_, n_sensors)), shrinkage='auto')
                            for t in range(n_time)
                        ], axis=0)

                    sigma = sigma_.mean(axis=0)
                    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)

                    ps_train = (ps_train.swapaxes(3, 4) @ sigma_inv).swapaxes(3, 4)
                    ps_test = (ps_test.swapaxes(3, 4) @ sigma_inv).swapaxes(3, 4)

                    # Average pseudotrials
                    ps_train = np.nanmean(ps_train, axis=1)
                    ps_test = np.nanmean(ps_test, axis=1)

                    # Decoding
                    for cA in range(n_objects):
                        for cB in range(cA + 1, n_objects):
                            for t in range(n_time):
                                train_x = np.array((ps_train[cA, :, :, t], ps_train[cB, :, :, t]))
                                train_x = np.reshape(train_x, (len(train_ix) * 2, n_sensors))

                                train_y = np.array([1] * len(train_ix) + [2] * len(train_ix))
                                test_y = np.array([1] * len(test_ix) + [2] * len(test_ix))

                                classifier = LinearSVC(
                                    penalty='l2',
                                    loss='hinge',
                                    C=0.5,
                                    multi_class='ovr',
                                    fit_intercept=True,
                                    max_iter=10000
                                )
                                classifier.fit(train_x, train_y)

                                test_x = np.array((ps_test[cA, :, :, t], ps_test[cB, :, :, t]))
                                test_x = np.reshape(test_x, (len(test_ix) * 2, n_sensors))

                                pred_y = classifier.predict(test_x)
                                acc_score = accuracy_score(test_y, pred_y)
                                decoding_results[ci, cci, cA, cB, t] = np.nansum(
                                    np.array((decoding_results[ci, cci, cA, cB, t], acc_score))
                                )

    # Average results
    decoding_results /= (N_PERMS * folds)

    # Save results
    out_dict = {
        'DA': decoding_results,
        'label': np.unique(data['label']),
        'category': np.unique(data['category']),
        'time': data['time'],
        'subject': SUBJ,
        'n_perms': N_PERMS
    }
    dump_data(out_dict, OUTPUT_NAME)

if __name__ == "__main__":
    main()