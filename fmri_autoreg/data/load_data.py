from typing import Dict, List, Tuple, Union

from pathlib import Path
import os
import re
import numpy as np
import h5py
import json
import pickle as pk
import torch
from nilearn.connectome import ConnectivityMeasure
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler


def load_params(params):
    """Load parameters from json file or json string."""
    if isinstance(params, Path):
        params = str(params)
    if os.path.splitext(params)[1] == ".json":
        with open(params) as json_file:
            param_dict = json.load(json_file)
    else:
        param_dict = json.loads(params)
    return param_dict


def load_data(
    path: Union[Path, str],
    h5dset_path: Union[List[str], str],
    standardize: bool = False,
    dtype: str = "data",
) -> List[Union[np.ndarray, str, int, float]]:
    """Load time series or phenotype data from the hdf5 files.

    Args:
        path (Union[Path, str]): Path to the hdf5 file.
        h5dset_path (Union[List[str], str]): Path to data inside the
            h5 file.
        standardize (bool, optional): Whether to standardize the data.
            Defaults to False. Only applicable to dtype='data'.
        dtype (str, optional): Attribute label for each subject or
            "data" to load the time series. Defaults to "data".

    Returns:
        List[Union[np.ndarray, str, int, float]]: loaded data.
    """
    if isinstance(h5dset_path, str):
        h5dset_path = [h5dset_path]
    data_list = []
    if dtype == "data":
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                data_list.append(h5file[p][:])
        if standardize and data_list:
            means = np.concatenate(data_list, axis=0).mean(axis=0)
            stds = np.concatenate(data_list, axis=0).std(axis=0)
            data_list = [(data - means) / stds for data in data_list]
        return data_list
    else:
        with h5py.File(path, "r") as h5file:
            for p in h5dset_path:
                subject_node = "/".join(p.split("/")[:-1])
                data_list.append(h5file[subject_node].attrs[dtype])
        return data_list


def load_h5_data_path(
    path: Union[Path, str],
    data_filter: Union[str, None] = None,
    shuffle: bool = False,
    random_state: int = 42,
) -> List[str]:
    """Load dataset path data from HDF5 file.

    Args:
      path (str): path to the HDF5 file
      data_filter (str): regular expression to apply on run names
        (default=None)
      shuffle (bool): whether to shuffle the data (default=False)

    Returns:
      (list of str): HDF5 path to data
    """
    data_list = []
    with h5py.File(path, "r") as h5file:
        for dset in _traverse_datasets(h5file):
            if data_filter is None or re.search(data_filter, dset):
                data_list.append(dset)
    if shuffle and data_list:
        rng = np.random.default_rng(seed=random_state)
        rng.shuffle(data_list)
    return data_list


def _traverse_datasets(hdf_file):
    """Load nested hdf5 files.
    https://stackoverflow.com/questions/51548551/reading-nested-h5-group-into-numpy-array
    """  # ruff: noqa: W505
    def h5py_dataset_iterator(g, prefix=""):
        for key in g.keys():
            item = g[key]
            path = f"{prefix}/{key}"
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


def load_darts_timeseries(path, task_filter=None, standardize=False, shuffle=False):
    ts_list = []
    scaler = Scaler(scaler=StandardScaler())
    rng = np.random.default_rng()
    with h5py.File(path, "r") as h5file:
        for key in list(h5file.keys()):
            if task_filter is None or re.search(task_filter, key):
                values = h5file[key][:].astype(np.float32)
                if shuffle:
                    values = rng.shuffle(values)
                run_data = TimeSeries.from_values(values)
                if standardize:
                    run_data = scaler.fit_transform(run_data)
                ts_list.append(run_data)
    return ts_list


def make_input_labels(
    tng_data, val_data, seq_length, time_stride, lag, compute_edge_index=False, thres=0.9
):
    """Generate pairs of inputs and labels from time series.

    Args:
      tng_data (list of numpy arrays): training data
      val_data (list of numpy arrays): validation data
      seq_length (int): length of input sequences
      time_stride (int): stride of the sliding window
      lag (int): time points difference between the end of the input sequence and the time point used for label
      compute_edge_index (bool): wether to compute a connectivity graph (for graph convolutions)
      thres (float): threshold used for the connectivity graph, e.g. 0.9 means that only the 10%
        strongest edges are kept (default=0.9)

    Returns:
      (tuple): tuple containing:
        X_tng (numpy array): training input
        Y_tng (numpy array): training labels
        X_val (numpy array): validation input
        Y_val (numpy array): validation labels, validation input, validation labels,
        edge_index (tuple of numpy arrays): edges of the connectivity matrix (None if compute_edge_index is False)
    """
    X_tng, Y_tng = make_seq(tng_data, seq_length, time_stride, lag)
    X_val, Y_val = make_seq(val_data, seq_length, time_stride, lag)

    if compute_edge_index:
        tng_data_concat = np.concatenate(tng_data, axis=0)
        edge_index = get_edge_index(tng_data_concat, thres)
    else:
        edge_index = None

    return X_tng, Y_tng, X_val, Y_val, edge_index


def make_seq(data_list, length, stride=1, lag=1):
    """Slice a list of timeseries with sliding windows and get corresponding labels.

    For each data in data list, pairs genreated will correspond to :
    `data[k:k+length]` for the sliding window and `data[k+length+lag-1]` for the label, with k
    iterating with the stride value.

    Args:
      data_list (list of numy arrays): list of data, data must be of shape (time_steps, features)
      length (int): length of the sliding window
      stride (int): stride of the sliding window (default=1)
      lag (int): time step difference between last time step of sliding window and label time step (default=1)

    Returns:
      (tuple): a tuple containing:
        X_tot (numpy array): sliding windows array of shape (nb of sequences, features, length)
        Y_tot (numpy array): labels array of shape (nb of sequences, features)
    """
    X_tot = []
    Y_tot = []
    delta = lag - 1
    for data in data_list:
        X = []
        Y = []
        for i in range(0, data.shape[0] - length - delta, stride):
            X.append(np.moveaxis(data[i : i + length], 0, 1))
            Y.append(data[i + length + delta])
        X_tot.append(np.array(X))
        Y_tot.append(np.array(Y))
    if len(X_tot) > 0:
        X_tot = np.concatenate(X_tot)
        Y_tot = np.concatenate(Y_tot)
    return X_tot, Y_tot


def get_edge_index(data_file, dset_paths, threshold=0.9):
    """Create connectivity matrix with more memory efficient way.

    Args:
      data_file: path to datafile
      dset_path (list of str): path to time series data
      threshold (float): threshold used for the connectivity graph, e.g. 0.9 means that only the 10%
        strongest edges are kept (default=0.9)

    Returns:
      (tuple of numpy array): edges of the connectivity matrix
    """
    connectome_measure = ConnectivityMeasure(kind="correlation", discard_diagonal=True)
    avg_corr_mats = None
    for dset in dset_paths:
        data = load_data(
            path=data_file,
            h5dset_path=dset,
            standardize=False,
            dtype="data"
        )
        corr_mat = connectome_measure.fit_transform(data)[0]
        if avg_corr_mats is None:
            avg_corr_mats = corr_mat
        else:
            avg_corr_mats += corr_mat
    avg_corr_mats /= len(dset_paths)

    thres_index = int(avg_corr_mats.shape[0] * avg_corr_mats.shape[1] * threshold)
    thres_value = np.sort(avg_corr_mats.flatten())[thres_index]
    adj_mat = avg_corr_mats * (avg_corr_mats >= thres_value)
    edge_index = np.nonzero(adj_mat)
    return edge_index


class Dataset:
    """Simple dataset for pytorch training loop"""

    def __init__(self, data_file, dset_paths, seq_length, time_stride, lag):
        self.data_file = data_file
        self.dset_paths = dset_paths
        self.param = {"length": seq_length, "stride": time_stride, "lag": lag}

    def __len__(self):
        return len(self.dset_paths)

    def __getitem__(self, index):
        sample_dset = self.dset_paths[index]
        # read the data
        input = load_data(
            path=self.data_file,
            h5dset_path=sample_dset,
            standardize=False,
            dtype="data"
        )
        # generate lables
        X, Y = make_seq(input, **self.param)
        sample = {
            "input": [torch.tensor(x, dtype=torch.float32) for x in X],
            "label": [torch.tensor(y, dtype=torch.float32) for y in Y]
        }
        return sample
