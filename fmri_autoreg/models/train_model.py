import numpy as np
import os
import argparse
import json
import pickle as pk
import csv
from math import ceil
from sklearn.metrics import r2_score
from fmri_autoreg.data.load_data import load_data, Dataset
from fmri_autoreg.models.make_model import make_model
from fmri_autoreg.models.predict_model import predict_model
from fmri_autoreg.tools import check_path
from torch.utils.data import DataLoader
from torch.cuda import is_available as cuda_is_available


def train(params, data, verbose=1):
    """Train a model according to params dict.

    Args:
      params (dict): paramter dictionary
      data (tuple): tuple containing the training and validation h5 dataset path and edge index
        tuple
      base_dir (str): path to a directory to prepend to data file paths in parameters dict (default=None)
      verbose (int): level of verbosity (default=1)

    Returns:
      (tuple): tuple containing:
        model: trained model
        r2_tng (numpy array): training r2 score
        r2_val (numpy array): validation r2 score
        Z_tng (numpy array): training prediction
        Y_tng (numpy array): training label
        Z_val (numpy array): validation prediction
        Y_val (numpy array): validation label
        losses (numpy array): losses
        checkpoints (dict): scores and mean losses at checkpoint epochs
    """
    X_tng_dsets, X_val_dsets, edge_index = data  # unpack data
    tmp = load_data(
            path=params["data_file"],
            h5dset_path=X_tng_dsets[0],
            standardize=False,
            dtype="data"
        )
    n_emb = tmp.shape[1]

    # make model
    model, train_model = make_model(params, n_emb, edge_index)
    tng_dataset = Dataset(
        params["data_file"],
        X_tng_dsets,
        params["seq_length"],
        params["time_stride"],
        params["lag"]
    )
    val_dataset = Dataset(
        params["data_file"],
        X_val_dsets,
        params["seq_length"],
        params["time_stride"],
        params["lag"]
    )
    tng_dataloader = DataLoader(
        tng_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params["num_workers"],
        pin_memory=cuda_is_available()
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=params["num_workers"],
        pin_memory=cuda_is_available()
    )

    # train model
    model, losses, checkpoints = train_model(
        model=model,
        params=params,
        tng_dataloader=tng_dataloader,
        val_dataloader=val_dataloader,
        verbose=verbose,
    )

    # compute r2 score
    r2_mean = {}
    for name, dset in zip(["tng", "val"], [X_tng_dsets, X_val_dsets]):
        r2 = predict_model(
            model=model,
            params=params,
            data_file=params["data_file"],
            dset=dset,
        )
        r2_mean[name] = np.mean(r2)
    return model, r2_mean['tng'], r2_mean['val'], losses, checkpoints
