import os
import pickle
import numpy as np


def get_boot_stats(all_subj_con, all_subj_chal, num_samples=100, verbose=False):
    """
    Performs bootstrapping on the decoded data to compute mean values and confidence intervals (CIs).

    Args:
        all_subj_con (dict): Decoded data for control condition.
        all_subj_chal (dict): Decoded data for challenge condition.
        num_samples (int, optional): Number of bootstrap samples to generate. Default is 100.

    Returns:
        tuple: Bootstrapped mean and confidence intervals for control, challenge, and their difference.
               Each element in the tuple contains three dictionaries for '2' and '6' durations.
    """
    datapoints = all_subj_con['2'].shape[1]
    durations = ['2', '6']
    boot_mean_con = {dur: [] for dur in durations}
    boot_mean_chal = {dur: [] for dur in durations}
    boot_mean_diff = {dur: [] for dur in durations}
    boot_CI_con = {dur: [] for dur in durations}
    boot_CI_chal = {dur: [] for dur in durations}
    boot_CI_diff = {dur: [] for dur in durations}

    for dur in durations:
        if verbose:
            print(f'duration: {dur}')
        for d_point in range(datapoints):
            if d_point % 10 == 0:
                if verbose:
                    print(d_point)

            pop_means_con, pop_means_chal, pop_means_diff = [], [], []

            for _ in range(num_samples):
                boot_sample_con = np.random.choice(all_subj_con[dur][:, d_point], size=len(all_subj_con[dur][:, d_point]), replace=True)
                boot_sample_chal = np.random.choice(all_subj_chal[dur][:, d_point], size=len(all_subj_chal[dur][:, d_point]), replace=True)
                boot_sample_diff = boot_sample_con - boot_sample_chal

                pop_means_con.append(np.mean(boot_sample_con))
                pop_means_chal.append(np.mean(boot_sample_chal))
                pop_means_diff.append(np.mean(boot_sample_diff))

            boot_mean_con[dur].append(np.mean(pop_means_con))
            boot_mean_chal[dur].append(np.mean(pop_means_chal))
            boot_mean_diff[dur].append(np.mean(pop_means_diff))

            boot_CI_con[dur].append([np.percentile(pop_means_con, 2.5), np.percentile(pop_means_con, 97.5)])
            boot_CI_chal[dur].append([np.percentile(pop_means_chal, 2.5), np.percentile(pop_means_chal, 97.5)])
            boot_CI_diff[dur].append([np.percentile(pop_means_diff, 2.5), np.percentile(pop_means_diff, 97.5)])

        boot_CI_con[dur] = np.array(boot_CI_con[dur])
        boot_CI_chal[dur] = np.array(boot_CI_chal[dur])
        boot_CI_diff[dur] = np.array(boot_CI_diff[dur])

    return boot_mean_con, boot_mean_chal, boot_mean_diff, boot_CI_con, boot_CI_chal, boot_CI_diff


def get_peaks(boot_mean_con, boot_mean_chal):
    """
    Extracts the peak values (height and latency) for control and challenge conditions.

    Args:
        boot_mean_con (dict): Bootstrapped mean data for control condition.
        boot_mean_chal (dict): Bootstrapped mean data for challenge condition.

    Returns:
        dict: A dictionary containing peak height and latency for both control and challenge conditions.
              Keys: 'con', 'chal', with values being lists of [peak height, peak latency].
    """
    durations = ['2', '6']
    avg_height_lat = {'con': [], 'chal': []}

    for dur in durations:
        height_con = np.max(boot_mean_con[dur])
        height_chal = np.max(boot_mean_chal[dur])
        lat_con = np.where(boot_mean_con[dur] == height_con)[0][0]
        lat_chal = np.where(boot_mean_chal[dur] == height_chal)[0][0]

        avg_height_lat['con'].append([height_con, lat_con])
        avg_height_lat['chal'].append([height_chal, lat_chal])

    return avg_height_lat