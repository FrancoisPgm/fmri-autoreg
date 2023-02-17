import numpy as np
import warnings
import os
import argparse
import pickle as pk
from math import ceil
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from src.data.load_data import load_params, load_data, make_input_labels


def predict_model(model, params, data_file, task_filter):
    """Use trained model to predict on data and compute R2 score."""
    data_list = load_data(
        data_file,
        task_filter=task_filter,
        standardize=params["standardize"] if "standardize" in params else False,
    )

    X, Y, _, _, edge_index = make_input_labels(
        data_list,
        [],
        params["seq_length"],
        params["time_stride"],
        params["lag"],
    )
    del data_list
    batch_size = len(X) if not "batch_size" in params else params["batch_size"]
    Z = np.concatenate([model.predict(x) for x in np.array_split(X, ceil(X.shape[0] / batch_size))])
    r2 = r2_score(Y, Z, multioutput="raw_values")
    return r2, Z, Y


def predict_horizon(
    model, seq_length, horizon, data_file, task_filter, batch_size, stride=1, standardize=False
):
    """For models trained to predict t+1, reuse predictions to iteratively predict to t+horizon."""
    data_list = load_data(
        data_file,
        task_filter=task_filter,
        standardize=standardize,
    )
    X = make_input_labels(data_list, [], seq_length + horizon, stride, 0)[0]
    del data_list
    if not len(X):
        warnings.warn(f"No data found in {data_file} for task {task_filter}", RuntimeWarning)
        return (None,) * 3
    Z = []
    Y = X[:, :, seq_length:]

    for i_batch in range((X.shape[0] - 1) // batch_size + 1):
        x_batch = X[i_batch * batch_size : (i_batch + 1) * batch_size, :, :seq_length]
        z_batch = []
        for lag in range(1, horizon + 1):
            z_batch.append(np.expand_dims(model.predict(x_batch), axis=-1))
            x_batch = np.concatenate((x_batch[:, :, 1:], z_batch[-1]), axis=-1)
        Z.append(np.concatenate(z_batch, axis=-1))

    Z = np.concatenate(Z, axis=0)

    r2 = []
    for i in range(horizon):
        r2.append(r2_score(Y[:, :, i], Z[:, :, i], multioutput="raw_values"))

    return np.array(r2), Z, Y


def main():
    """Use trained model to predict on data and save R2 score, prediction and labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="Path to model file or dir.")
    parser.add_argument("--data_file", "-f", type=str, help="Path to data HDF5 file.")
    parser.add_argument(
        "--task_filter", "-t", type=str, default="", help="Regex to filter run names."
    )
    parser.add_argument("--out_dir", "-o", type=str, help="Path to output directory.")
    parser.add_argument("--tag", type=str, default="", help="Tag to append to output file names.")
    parser.add_argument("--verbose", "-v", type=int, default=1, help="Verbosity level.")
    args = parser.parse_args()

    model_path = (
        args.model if os.path.splitext(args.model)[1] else os.path.join(args.model, "model.pkl")
    )
    out_dir = args.out_dir if args.out_dir else os.path.dirname(model_path)

    model = pk.load(open(model_path, "rb"))
    params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))

    r2, Z, Y = predict_model(model, params, args.data_file, task_filter)
    if args.verbosity:
        print("mean r2 score :", r2.mean())

    tag = "_" + args.tag if args.tag else ""

    np.save(os.path.join(out_dir, "r2_prediction" + tag + ".npy"), r2)
    np.save(os.path.join(out_dir, "Y_prediction" + tag + ".npy"), Y)
    np.save(os.path.join(out_dir, "Z_prediction" + tag + ".npy"), Z)


if __name__ == "__main__":
    main()
