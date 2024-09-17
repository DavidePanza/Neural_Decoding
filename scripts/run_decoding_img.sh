#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --qos=standard
#SBATCH --time=6-6:00:00
#SBATCH --job-name=object_decoding


# activate envs
source activate eeg


participants=(5)  # This is an array with one element, the number 5. Add more participants separated by spaces if needed.
presentation_times=('6')  # This is an array with two elements: '2' and '6'.

# Loop through participants
for participant in "${participants[@]}"; do
    # Loop through presentation times
    for presentation_time in "${presentation_times[@]}"; do
        # Run your Python script with the current participant and presentation time
        python decoding_img.py "${participant}" "${presentation_time}"
        # python your_script.py --participant "$participant" --presentation-time "$presentation_time"
    done
done

