import numpy as np
import os
import re
import argparse
import json
import pickle as pk
import csv
from math import ceil
from sklearn.metrics import r2_score
from src.data.load_data import (
    load_params,
    load_data,
    make_input_labels,
    make_seq,
    load_darts_timeseries,
)
from src.tools import check_path
from src.models.models import models_needing_edge_index
from src.models.train_model import train
from src.models.train_darts_model import train as train_darts
from src.models.train_darts_model import compute_R2


SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
ATLASES = ["mist_197"]
LAGS = list(range(1, 7))


def main():
    """Train model using parameters dict and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, help="output directory")
    parser.add_argument("--param", "-p", type=str, help="Parameters : path to json file or dict")
    parser.add_argument("--is_darts", action="store_true")
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

    params = load_params(args.param)
    if args.base_dir:
        params["tng_data_file"] = os.path.join(args.base_dir, params["tng_data_file"])
        params["val_data_file"] = os.path.join(args.base_dir, params["val_data_file"])
        params["test_data_file"] = os.path.join(args.base_dir, params["test_data_file"])
    standardize = params["standardize"] if "standardize" in params else False
    compute_edge_index = params["model"] in models_needing_edge_index
    thres = params["edge_index_thres"] if compute_edge_index else None
    output_dir = check_path(args.output_dir)
    os.makedirs(output_dir)

    for sub in SUBJECTS:
        params["tng_data_file"] = re.sub("sub-0[1-6]", sub, params["tng_data_file"])
        params["val_data_file"] = re.sub("sub-0[1-6]", sub, params["val_data_file"])
        params["test_data_file"] = re.sub("sub-0[1-6]", sub, params["test_data_file"])
        for atlas in ATLASES:
            params["tng_data_file"] = re.sub("mist_[0-9]+", atlas, params["tng_data_file"])
            params["val_data_file"] = re.sub("mist_[0-9]+", atlas, params["val_data_file"])
            params["test_data_file"] = re.sub("mist_[0-9]+", atlas, params["test_data_file"])

            if args.is_darts:
                if args.verbose:
                    print(sub)
                lag = max(LAGS)
                params["horizon"] = lag

                out_subdir = os.path.join(output_dir, f"{sub}_{atlas}")
                os.makedirs(out_subdir)
                with open(os.path.join(out_subdir, "params.json"), "w") as f:
                    json.dump(params, f, indent=2)

                tng_series = load_darts_timeseries(
                    params["tng_data_file"], params["tng_task_filter"], standardize
                )
                val_series = load_darts_timeseries(
                    params["val_data_file"], params["val_task_filter"], standardize
                )
                test_series = load_darts_timeseries(
                    params["test_data_file"], params["test_task_filter"], standardize
                )
                data = tng_series, val_series
                if not tng_series or not val_series or not test_series:
                    print("No tng_data or val_data or test_data found.")
                    return None

                model, tng_R2, val_R2 = train_darts(params, data, verbose=args.verbose)
                start = (
                    model.input_chunk_length
                    if hasattr(model, "input_chunk_length")
                    else -min(model.lags["target"])
                )
                if "horizon" in params:
                    horizon = params["horizon"]
                else:
                    horizon = (
                        model.output_chunk_length if hasattr(model, "output_chunk_length") else 1
                    )
                test_R2 = compute_R2(test_series, model, horizon, start, args.verbose > 2)

                np.save(os.path.join(out_subdir, "tng_R2.npy"), val_R2)
                np.save(os.path.join(out_subdir, "val_R2.npy"), val_R2)
                np.save(os.path.join(out_subdir, "test_R2.npy"), test_R2)
                with open(os.path.join(out_subdir, "model.pkl"), "wb") as f:
                    pk.dump(model, f)
                if args.verbose:
                    print("Results saved at {}".format(out_subdir))

            else:  # not darts model
                for lag in LAGS:
                    if args.verbose:
                        print(f"{sub} lag-{lag}")
                    params["lag"] = lag
                    out_subdir = os.path.join(output_dir, f"{sub}_{atlas}_lag-{lag}")
                    os.makedirs(out_subdir)
                    with open(os.path.join(out_subdir, "params.json"), "w") as f:
                        json.dump(params, f, indent=2)

                    tng_data = load_data(
                        params["tng_data_file"], params["tng_task_filter"], standardize
                    )
                    val_data = load_data(
                        params["val_data_file"], params["val_task_filter"], standardize
                    )
                    test_data = load_data(
                        params["test_data_file"], params["test_task_filter"], standardize
                    )

                    if not tng_data or not val_data or not test_data:
                        print("No tng_data or val_data or test_data found.")
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

                    model, r2_tng, r2_val, Z_tng, Y_tng, Z_val, Y_val, _, _ = train(
                        params, data, verbose=args.verbose
                    )

                    X_test, Y_test = make_seq(
                        test_data, params["seq_length"], params["time_stride"], params["lag"]
                    )
                    del test_data
                    batch_size = len(X_tng) if not "batch_size" in params else params["batch_size"]
                    Z_test = np.concatenate(
                        [
                            model.predict(x)
                            for x in np.array_split(X_test, ceil(X_test.shape[0] / batch_size))
                        ]
                    )
                    r2_test = r2_score(Y_test, Z_test, multioutput="raw_values")

                    model = model.to("cpu")
                    with open(os.path.join(out_subdir, "model.pkl"), "wb") as f:
                        pk.dump(model, f)

                    np.save(os.path.join(out_subdir, "r2_tng.npy"), r2_tng)
                    np.save(os.path.join(out_subdir, "r2_val.npy"), r2_val)
                    np.save(os.path.join(out_subdir, "r2_test.npy"), r2_test)
                    np.save(os.path.join(out_subdir, "pred_tng.npy"), Z_tng)
                    np.save(os.path.join(out_subdir, "labels_tng.npy"), Y_tng)
                    np.save(os.path.join(out_subdir, "pred_val.npy"), Z_val)
                    np.save(os.path.join(out_subdir, "labels_val.npy"), Y_val)
                    np.save(os.path.join(out_subdir, "pred_val.npy"), Z_test)
                    np.save(os.path.join(out_subdir, "labels_val.npy"), Y_test)

                    if args.verbose:
                        print("Results saved at {}".format(out_subdir))


if __name__ == "__main__":
    main()
