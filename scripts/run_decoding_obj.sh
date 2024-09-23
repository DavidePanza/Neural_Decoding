#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --qos=standard
#SBATCH --time=4-6:00:00
#SBATCH --job-name=object_decoding


# activate envs
source activate eeg

# script to run
participants=(4 5)
presentation_times=('2' '6')
test_ratios=(0.1)   # if pseudotrials == random use only (0.1) otherwise (0.1 0.2)
pseudotrials=('fixed')    # 'random_separated_img' 'fixed' 'random'
n_perm=50

# Loop through participants
for participant in "${participants[@]}"; do
    # Loop through presentation times
    for presentation_time in "${presentation_times[@]}"; do
        # Loop through test ratios
        for test_ratio in "${test_ratios[@]}"; do
            # Loop through pseudotrials
            for pseudotrial in "${pseudotrials[@]}"; do
                # Run your Python script with the current participant, presentation time, test ratio, and pseudotrial
                python decoding_obj.py ${participant} ${presentation_time} ${test_ratio} ${pseudotrial} ${n_perm}
            done
        done
    done
done
