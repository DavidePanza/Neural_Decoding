
import pickle
import numpy as np
from typing import Dict, Tuple, List, Any

def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        Dict[str, Any]: The data loaded from the file.
    """
    print(f'Loading file: {file_path}')
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def dump_data(data: Dict[str, Any], filename: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data (Dict[str, Any]): Data to be saved.
        filename (str): Path to the pickle file.
    """
    print(f'Writing file: {filename}')
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def subsample_data(data: Dict[str, Any], sub_factor: int) -> Dict[str, Any]:
    """
    Subsample data by a given factor.

    Args:
        data (Dict[str, Any]): Data dictionary containing 'eeg', 'time', and 'id'.
        sub_factor (int): Factor by which to subsample the data.

    Returns:
        Dict[str, Any]: The subsampled data.
    """
    shape = data['eeg'].shape
    sub_indices = list(range(0, shape[3], sub_factor))
    data['eeg'] = data['eeg'][:, :, :, sub_indices]
    data['time'] = data['time'][sub_indices]
    data['id'] = [int(x) for x in data['id']]
    return data


def downsample_data(data: Dict[str, Any], sub_factor: int) -> Dict[str, Any]:
    """
    Downsample data by a given factor.

    Args:
        data (Dict[str, Any]): Data dictionary containing 'eeg' and 'time'.
        sub_factor (int): Factor by which to downsample the data.

    Returns:
        Dict[str, Any]: The downsampled data.
    """
    shape = data['eeg'].shape
    if len(shape) == 4:
        sub_indices = list(range(0, shape[3], sub_factor))
        data['eeg'] = data['eeg'][:, :, :, sub_indices]
        data['time'] = data['time'][sub_indices]
    elif len(shape) == 3:
        sub_indices = list(range(0, shape[2], sub_factor))
        data['eeg'] = data['eeg'][:, :, sub_indices]
        data['time'] = data['time'][sub_indices]
    return data


def get_pseudotrials_img(data: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """
    Generate pseudotrials by random permutation of trials for image data.

    Args:
        data (Dict[str, Any]): Data dictionary containing 'eeg'.

    Returns:
        Tuple[np.ndarray, int]: The pseudotrials and the number of trials per pseudotrial.
    """
    shape = data['eeg'].shape
    k = shape[1]
    l = int(shape[1] / k)

    while l < 5:
        k -= 1
        l = int(shape[1] / k)

    data['eeg'] = data['eeg'][:, np.random.permutation(shape[1]), :, :]
    data['eeg'] = data['eeg'][:, :l*k, :, :]

    pseudotrials = np.reshape(data['eeg'], (shape[0], k, l, shape[2], shape[3]))
    pseudotrials = pseudotrials.mean(axis=1)

    return pseudotrials, k



def get_pseudotrials_obj(data: Dict[str, Any], object_indices: List[int]) -> Tuple[np.ndarray, int, int]:
    """
    Generate pseudotrials with a fixed number of trials for each object.

    Args:
        data (Dict[str, Any]): Data dictionary containing EEG data.
        object_indices (List[int]): List of object indices.

    Returns:
        Tuple[np.ndarray, int, int]: The pseudotrials, number of pseudotrials, and number of trials.
    """
    unique_objects, object_counts = np.unique(object_indices, return_counts=True)
    n_conds, n_images, n_trials, n_channels, n_time = data.shape

    min_count = int(np.min(object_counts) / 2)
    k = min_count  # Number of pseudotrials equal to the smallest count
    l = n_trials

    pseudotrials = np.full((n_conds, len(unique_objects), l, k, n_channels, n_time), np.nan)
    for i, obj in enumerate(unique_objects):
        mask = np.array(object_indices) == obj
        d = np.array((data[0, mask[0]], data[1, mask[1]]))
        d = d[:, np.random.permutation(d.shape[1])]
        d = d[:, :min_count]
        d = np.swapaxes(d, 1, 2)
        pseudotrials[:, i] = d

    return pseudotrials, k, l


def subset_data(data: Dict[str, Any], dim: int, mask: List[int]) -> Dict[str, Any]:
    """
    Subset data along a specified dimension using a mask.

    Args:
        data (Dict[str, Any]): Data dictionary to be subsetted.
        dim (int): Dimension along which to subset the data.
        mask (List[int]): List of indices to retain in the subset.

    Returns:
        Dict[str, Any]: The subsetted data.
    """
    subsetted_data = {}
    for key in data.keys():
        if key not in ['trial', 'chans', 'time']:
            idxs = [slice(None)] * len(np.array(data[key]).shape)
            idxs[dim] = mask
            subsetted_data[key] = np.copy(data[key])[tuple(idxs)]
    return subsetted_data


def get_sets(objects: List[int], test_size: float, binsize: int, n_perms: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training and testing sets for cross-validation.

    Args:
        objects (List[int]): List of object labels.
        test_size (float): Proportion of data to be used for testing.
        binsize (int): Number of bins for the pseudotrials.
        n_perms (int): Number of permutations to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of training and testing indices.
    """
    unique_objects, object_counts = np.unique(objects, return_counts=True)

    set_len = np.min(object_counts)
    train_len = int(np.round(set_len * (1 - test_size)))
    test_len = set_len - train_len

    train_indices = np.full((n_perms, len(unique_objects), train_len), np.nan)
    test_indices = np.full((n_perms, len(unique_objects), test_len), np.nan)

    for i in range(n_perms):
        for o in range(len(unique_objects)):
            obj_indices = np.where(np.array(objects) == unique_objects[o])[0]
            np.random.seed(i)
            shuffled_indices = np.random.permutation(obj_indices)

            train_indices[i, o, :] = shuffled_indices[:train_len]
            test_indices[i, o, :] = shuffled_indices[train_len:set_len]

    return train_indices, test_indices


def my_ceil(a: float, precision: int = 0) -> float:
    """
    Custom ceiling function with specified precision.

    Args:
        a (float): Number to round up.
        precision (int): Number of decimal places to round to.

    Returns:
        float: The rounded-up number.
    """
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)