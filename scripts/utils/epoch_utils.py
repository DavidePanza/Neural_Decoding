# epoch_utils.py
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd


def load_data(file):
    """Loads and returns data from a pickle file.

    Args:
        file (str): The path to the pickle file to be loaded.

    Returns:
        object: The data loaded from the pickle file.
    """
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    return data


def get_csv_and_parse_rows(filename, header, outfile):
    """
    Reads a CSV file, parses rows to ensure each has the correct number of columns,
    and writes the parsed data to an output CSV file.

    Args:
        filename (str): The input CSV file to read from.
        header (list): A list of column names for the CSV file.
        outfile (str): The output CSV file to write the processed data to.

    Returns:
        None
    """
    ncol = len(header)
    res = []

    with open(filename) as infile:  # Read CSV
        for line in infile:
            val = line.strip().split(",")
            # Check if ncol elements are in each line
            if len(val) == ncol:
                res.append(val)
            # Else, split into smaller chunks and append
            else:
                res.extend([val[i:i+ncol] for i in range(0, len(val), ncol)])

    df = pd.DataFrame(res, columns=header)
    df.to_csv(outfile, index=False)
    
    
def dump_data(data, filename):
    print('writing file: ' + filename)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def get_cat(x):
	return str(x)[1]

def get_im(x):
	return str(int(str(x)[3:]))
 
def get_dur(x):
	return str(int(str(x)[2]))

def get_lab(x):
	return str(x)[0]


def sort_data_im(eeg, dat_id, chans, time, im_ix, sq_ix):
    """
    Sorts EEG data based on unique identifiers, shuffling trials for each unique ID 
    and returning the sorted data along with corresponding metadata.

    Args:
        eeg (np.ndarray): The EEG data with shape (trials, channels, timepoints).
        dat_id (np.ndarray): Array of identifiers corresponding to each trial in the EEG data.
        chans (list): List of EEG channels.
        time (np.ndarray): Time points corresponding to the EEG data.
        im_ix (np.ndarray): Image index array corresponding to each trial.
        sq_ix (np.ndarray): Sequence index array corresponding to each trial.

    Returns:
        dict: A dictionary containing sorted EEG data and associated metadata:
            - 'eeg' (np.ndarray): The sorted EEG data.
            - 'img_ix' (np.ndarray): The sorted image indices.
            - 'seq_ix' (np.ndarray): The sorted sequence indices.
            - 'id' (np.ndarray): The unique trial IDs.
            - 'label' (np.ndarray): Labels associated with each trial ID.
            - 'category' (np.ndarray): Categories associated with each trial ID.
            - 'duration' (np.ndarray): Durations associated with each trial ID.
            - 'img_n' (np.ndarray): Image numbers associated with each trial ID.
            - 'trial' (np.ndarray): The trial indices.
            - 'chans' (list): The EEG channels.
            - 'time' (np.ndarray): The time points for the EEG data.
    """
    dshape = eeg.shape
    ids, id_counts = np.unique(dat_id, return_counts=True)

    sdat = np.empty((len(ids), id_counts.min(), dshape[1], dshape[2]))
    nids = np.zeros((len(ids)))
    nimix = np.empty((len(ids), id_counts.min()))
    nsqix = np.empty((len(ids), id_counts.min()))
    
    for c in tqdm(range(len(ids))):
        d = eeg[dat_id == ids[c]]
        im_ix_ = np.array(im_ix)
        im_ = im_ix_[dat_id == ids[c]]
        seq_ix_ = np.array(sq_ix)
        sq_ = seq_ix_[dat_id == ids[c]]

        shuffle_ix = np.arange(id_counts[c])
        np.random.shuffle(shuffle_ix)
        selected_trials = shuffle_ix[:id_counts.min()]
        sdat[c] = d[selected_trials, :, :]
        nids[c] = int(ids[c])
        nimix[c] = im_[selected_trials]
        nsqix[c] = sq_[selected_trials]

    ndat = {
            'eeg': sdat,
            'img_ix': nimix,
            'seq_ix': nsqix,
            'id': nids.astype(int),
            'label': np.array([get_lab(int(x)) for x in nids]),
            'category': np.array([get_cat(int(x)) for x in nids]),
            'duration': np.array([get_dur(int(x)) for x in nids]),
            'img_n': np.array([get_im(int(x)) for x in nids]),
            'trial': np.arange(id_counts.min()),
            'chans': chans,
            'time': time
            }

    return ndat