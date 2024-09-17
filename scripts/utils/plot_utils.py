import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_between_cond(boot_mean_con, boot_mean_chal, avg_height_lat, boot_CI_con, boot_CI_chal, boot_mean_diff, boot_CI_diff):
    """
    Plot decoding accuracy between different conditions for each duration.

    Args:
        boot_mean_con (dict): Bootstrapped mean values for control condition.
        boot_mean_chal (dict): Bootstrapped mean values for challenge condition.
        avg_height_lat (dict): Peak height and latency for control and challenge conditions.
        boot_CI_con (dict): Confidence intervals for control condition.
        boot_CI_chal (dict): Confidence intervals for challenge condition.
        boot_mean_diff (dict): Bootstrapped mean differences between conditions.
        boot_CI_diff (dict): Confidence intervals for the difference.
    """
    durations = ['2', '6']
    durations_labels = ['34ms', '100ms']
    train_test_conditions = ['Challenge', 'Cont-Chal', 'Chal-Cont', 'Control']
    x_fill = np.arange(len(boot_mean_con['2']))
    marker_size = 11
    
    for idx, duration in enumerate(durations):
        plt.figure(figsize=(6, 5))
        
        # Plot control condition
        plt.plot(boot_mean_con[duration], '-b', label=train_test_conditions[3])
        plt.plot(avg_height_lat['con'][idx][1], avg_height_lat['con'][idx][0], marker='*', markersize=marker_size, color='b', markeredgecolor='black')
        plt.fill_between(x_fill, boot_CI_con[duration][:, 0], boot_CI_con[duration][:, 1], color='blue', alpha=0.1)
        
        # Plot challenge condition
        plt.plot(boot_mean_chal[duration], '-r', label=train_test_conditions[0])
        plt.plot(avg_height_lat['chal'][idx][1], avg_height_lat['chal'][idx][0], marker='*', markersize=marker_size, color='r', markeredgecolor='black')
        plt.fill_between(x_fill, boot_CI_chal[duration][:, 0], boot_CI_chal[duration][:, 1], color='red', alpha=0.1)
        
        # Plot difference
        plt.plot(np.array(boot_mean_diff[duration]) + 0.5, 'g-', linewidth=0.5, label='Difference')
        plt.fill_between(x_fill, boot_CI_diff[duration][:, 0] + 0.5, boot_CI_diff[duration][:, 1] + 0.5, color='green', alpha=0.1)
        
        plt.ylim(0.4, 0.75)
        plt.axvline(x=10, color='gray', linestyle='--')
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Decoding Accuracy')
        plt.xticks(np.linspace(0, 110, num=12), ['-50', '0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500'], rotation=45)
        plt.title(f'Object Decoding ({durations_labels[idx]})')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


def plot_across_dur(boot_mean_con, boot_mean_chal, avg_height_lat, boot_CI_con, boot_CI_chal, boot_mean_diff, boot_CI_diff):
    """
    Plots the decoding accuracy across different durations for control and challenge conditions. It includes confidence intervals
    and plots differences between durations for both conditions.

    Args:
        boot_mean_con (dict): Dictionary of bootstrapped mean decoding accuracies for control condition across different durations.
        boot_mean_chal (dict): Dictionary of bootstrapped mean decoding accuracies for challenge condition across different durations.
        avg_height_lat (dict): Dictionary of average decoding accuracies and latencies for control and challenge conditions.
        boot_CI_con (dict): Dictionary of bootstrapped confidence intervals for control condition.
        boot_CI_chal (dict): Dictionary of bootstrapped confidence intervals for challenge condition.
        boot_mean_diff (dict): Dictionary of bootstrapped mean differences between conditions across durations.
        boot_CI_diff (dict): Dictionary of bootstrapped confidence intervals for differences between conditions.

    Returns:
        None
    """
    
    # Get difference
    con_diff = (np.array(boot_mean_con['6']) - np.array(boot_mean_con['2'])) + 0.5
    cha_diff = (np.array(boot_mean_chal['6']) - np.array(boot_mean_chal['2'])) + 0.5
    
    # Initialize variables
    durations = ['2', '6']
    durations_lab = ['34ms', '100ms']
    x_fill = np.arange(len(boot_mean_con['2']))
    
    palette_con = [[8/255,48/255,107/255], [107/255,174/255,214/255]]
    palette_chal = [[103/255,0/255,13/255], [251/255,106/255,74/255]]
    marker_size = 11
        
    # Plot control condition
    plt.figure()
    plt.subplots(1, figsize=(6, 5))
    for dur_idx, dur in enumerate(durations):
        plt.plot(boot_mean_con[dur], '-', label=durations_lab[dur_idx], color=palette_con[dur_idx], zorder=1)
        plt.plot(avg_height_lat['con'][dur_idx][1], avg_height_lat['con'][dur_idx][0], marker='*', markersize=marker_size, color=palette_con[dur_idx], markeredgecolor='black', zorder=2)
        plt.fill_between(x_fill, y1=boot_CI_con[dur][:, 0], y2=boot_CI_con[dur][:, 1], alpha=0.08, color=palette_con[dur_idx])

        # Plot difference for control
        if dur_idx == 0:
            plt.plot(con_diff, 'g-', linewidth=0.5)
        else:
            plt.plot(con_diff, 'g-', linewidth=0.5, label='Difference')

        # Add reference lines and formatting
        plt.axvline(x=10, color='gray', linestyle='--')
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.xticks(np.linspace(0, 110, num=12), ['-50', '0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500'], rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Decoding Accuracy')
        plt.ylim(0.4, 0.9)
        plt.legend(loc='upper right', frameon=False, labelspacing=0.3)

    plt.tight_layout()  
    plt.show()

    # Plot challenge condition
    plt.figure()
    plt.subplots(1, figsize=(6, 5))
    for dur_idx, dur in enumerate(durations):
        plt.plot(boot_mean_chal[dur], '-', label=durations_lab[dur_idx], color=palette_chal[dur_idx], zorder=1)
        plt.plot(avg_height_lat['chal'][dur_idx][1], avg_height_lat['chal'][dur_idx][0], marker='*', markersize=marker_size, color=palette_chal[dur_idx], markeredgecolor='black', zorder=2)
        plt.fill_between(x_fill, y1=boot_CI_chal[dur][:, 0], y2=boot_CI_chal[dur][:, 1], alpha=0.08, color=palette_chal[dur_idx])

        # Plot difference for challenge
        if dur_idx == 0:
            plt.plot(cha_diff, 'g-', linewidth=0.5)
        else:
            plt.plot(cha_diff, 'g-', linewidth=0.5, label='Difference')

        # Add reference lines and formatting
        plt.axvline(x=10, color='gray', linestyle='--')
        plt.axhline(y=0.5, color='gray', linestyle='--')
        plt.xticks(np.linspace(0, 110, num=12), ['-50', '0', '50', '100', '150', '200', '250', '300', '350', '400', '450', '500'], rotation=45)
        plt.xlabel('Time')
        plt.ylabel('Decoding Accuracy')
        plt.ylim(0.4, 0.9)
        plt.legend(loc='upper right', frameon=False, labelspacing=0.3)

    plt.tight_layout()  
    plt.show()




