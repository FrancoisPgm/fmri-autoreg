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
    if os.path.splitext(params)[1] == ".json":
        with open(params) as json_file:
            param_dict = json.load(json_file)
    else:
        param_dict = json.loads(params)
    return param_dict


def load_data(path, task_filter=None, standardize=False, shuffle=False):
    """Load pre-processed data from HDF5 file.

    Args:
      path (str): path to the HDF5 file
      task_filter (str): regular expression to apply on run names (default=None)
      standardize (bool): bool (default=False)
      shuffle (bool): wether to shuffle the data (default=False)

    Returns:
      (list of numpy arrays): loaded data
    """
    data_list = []
    with h5py.File(path, "r") as h5file:
        for key in list(h5file.keys()):
            if task_filter is None or re.search(task_filter, key):
                data_list.append(h5file[key][:])
    if standardize and data_list:
        means = np.concatenate(data_list, axis=0).mean(axis=0)
        stds = np.concatenate(data_list, axis=0).std(axis=0)
        data_list = [(data - means) / stds for data in data_list]
    if shuffle and data_list:
        rng = np.random.default_rng()
        data_list = [rng.shuffle(d) for d in data_list]
    return data_list


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


def get_edge_index(data, threshold=0.9):
    """Create connectivity matrix.

    Args:
      data (numpy array): time series data, of shape (time, nodes)
      threshold (float): threshold used for the connectivity graph, e.g. 0.9 means that only the 10%
        strongest edges are kept (default=0.9)

    Returns:
      (tuple of numpy array): edges of the connectivity matrix
    """
    connectome_measure = ConnectivityMeasure(kind="correlation", discard_diagonal=True)
    corr_mat = connectome_measure.fit_transform([data])[0]
    thres_index = int(corr_mat.shape[0] * corr_mat.shape[1] * threshold)
    thres_value = np.sort(corr_mat.flatten())[thres_index]
    adj_mat = corr_mat * (corr_mat >= thres_value)
    edge_index = np.nonzero(adj_mat)
    return edge_index


class Dataset:
    """Simple dataset for pytorch training loop"""

    def __init__(self, X, Y):
        self.inputs = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.labels = [torch.tensor(y, dtype=torch.float32) for y in Y]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = {"input": self.inputs[index], "label": self.labels[index]}
        return sample
