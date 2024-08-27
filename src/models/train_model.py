import numpy as np
import os
import argparse
import json
import pickle as pk
import csv
from math import ceil
from sklearn.metrics import r2_score

from src.data.load_data import load_params, load_data, make_input_labels
from src.models.make_model import make_model
from src.tools import check_path


def train(params, data, verbose=1):
    """Train a model according to params dict.

    Args:
      params (dict): paramter dictionary
      data (tuple): tuple containing the training and validation data and edge index
        tuple; if None, the data is loaded from the parameters in the params dict
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
    X_tng, Y_tng, X_val, Y_val, edge_index = data
    n_emb = X_tng.shape[1]

    # make model
    model, train_model = make_model(params, n_emb, edge_index)
    if verbose:
        print("model made")

    # train model
    model, losses, checkpoints = train_model(
        model=model,
        params=params,
        X_tng=X_tng,
        Y_tng=Y_tng,
        X_val=X_val,
        Y_val=Y_val,
        verbose=verbose,
    )

    # compute score
    batch_size = len(X_tng) if not "batch_size" in params else params["batch_size"]
    Z_tng = np.concatenate(
        [model.predict(x) for x in np.array_split(X_tng, ceil(X_tng.shape[0] / batch_size))]
    )
    Z_val = np.concatenate(
        [model.predict(x) for x in np.array_split(X_val, ceil(X_val.shape[0] / batch_size))]
    )
    r2_tng = r2_score(Y_tng, Z_tng, multioutput="raw_values")
    r2_val = r2_score(Y_val, Z_val, multioutput="raw_values")
    if verbose:
        print("tng mean r2 : {}, val mean r2 : {}".format(r2_tng.mean(), r2_val.mean()))

    return model, r2_tng, r2_val, Z_tng, Y_tng, Z_val, Y_val, losses, checkpoints


def main(args):
    """Train a model and save the results."""
    params = load_params(args.param)
    if args.base_dir:
        params["tng_data_file"] = os.path.join(args.base_dir, params["tng_data_file"])
        params["val_data_file"] = os.path.join(args.base_dir, params["val_data_file"])
    shuffle = params["shuffle"] if "shuffle" in params else False
    standardize = params["standardize"] if "standardize" in params else False
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None

    output_dir = check_path(args.output_dir)
    os.makedirs(output_dir)
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    tng_data = load_data(
        params["tng_data_file"],
        params["tng_task_filter"],
        standardize,
        shuffle,
    )
    val_data = load_data(
        params["val_data_file"],
        params["val_task_filter"],
        standardize,
        shuffle,
    )

    if not tng_data or not val_data:
        if args.verbose:
            print("No tng_data or val_data found.")
        return None

    data = make_input_labels(
        tng_data,
        val_data,
        params["seq_length"],
        params["time_stride"],
        params["lag"],
        compute_edge_index,
        thres,
    )
    del tng_data
    del val_data
    if args.verbose:
        print("data loaded")

    model, r2_tng, r2_val, Z_tng, Y_tng, Z_val, Y_val, losses, checkpoints = train(
        params, data, verbose=args.verbose
    )
    model = model.to("cpu")
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)

    np.save(os.path.join(output_dir, "r2_tng.npy"), r2_tng)
    np.save(os.path.join(output_dir, "r2_val.npy"), r2_val)
    np.save(os.path.join(output_dir, "pred_tng.npy"), Z_tng)
    np.save(os.path.join(output_dir, "labels_tng.npy"), Y_tng)
    np.save(os.path.join(output_dir, "pred_val.npy"), Z_val)
    np.save(os.path.join(output_dir, "labels_val.npy"), Y_val)
    if losses is not None:
        np.save(
            os.path.join(output_dir, "losses.npy"),
            np.array([losses["tng"], losses["val"]]),
        )
    if checkpoints:
        with open(os.path.join(output_dir, "checkpoints.csv"), "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(checkpoints[0].keys()))
            writer.writeheader()
            writer.writerows(checkpoints)

    if args.verbose:
        print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a model and save the results.")
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    parser.add_argument("--param", "-p", type=str, help="Parameters : path to json file or dict")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        help="Verbosity level, 0 to 2. Default is 1.",
    )
    parser.add_argument(
        "--base_dir",
        "-b",
        type=str,
        help="Base directory for data files.",
        default=None,
    )
    args = parser.parse_args()
    main(args)
