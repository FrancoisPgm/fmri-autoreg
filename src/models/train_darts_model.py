import os
import json
import numpy as np
import darts.models
import torch.optim.lr_scheduler
import torch.nn
from copy import deepcopy
import argparse
import pickle as pk
from sklearn.metrics import r2_score
from src.data.load_data import load_params, load_darts_timeseries
from src.tools import check_path


def compute_R2(series_list, model, horizon, start, verbose=False):
    r2_list = [[] for _ in range(horizon)]
    for series in series_list:
        pred = model.historical_forecasts(
            series,
            start=start,
            forecast_horizon=horizon,
            stride=1,
            retrain=False,
            last_points_only=False,  # =horizon>1,
            verbose=verbose,
        )
        # pred_per_lag = []
        # time_indices = []
        for i in range(horizon):
            pred_i = []
            time_i = []
            for pred_chunk in pred:
                pred_chunk_values = np.nan_to_num(pred_chunk.values(), copy=False)
                pred_i.append(pred_chunk_values[i])
                time_i.append(pred_chunk.time_index[i])
            # pred_per_lag.append(np.array(pred_i))
            # time_indices.append(np.array(time_i))
            r2_list[i].append(r2_score(series.values()[time_i], pred_i, multioutput="raw_values"))

    return np.array(r2_list)


def train(params, data, verbose=1):
    tng_series, val_series = data
    params = deepcopy(params)

    if "lr_scheduler_cls" in params["model_params"]:
        params["model_params"]["lr_scheduler_cls"] = getattr(
            torch.optim.lr_scheduler, params["model_params"]["lr_scheduler_cls"]
        )
    if "loss_fn" in params["model_params"]:
        params["model_params"]["loss_fn"] = getattr(torch.nn, params["model_params"]["loss_fn"])()

    model = getattr(darts.models, params["model"])(**params["model_params"])
    start = (
        model.input_chunk_length
        if hasattr(model, "input_chunk_length")
        else -min(model.lags["target"])
    )
    horizon = 6
    if "horizon" in params:
        horizon = params["horizon"]
    #else:
    #    horizon = model.output_chunk_length if hasattr(model, "output_chunk_length") else 1

    if verbose:
        print("Fitting model.")
    model.fit(tng_series)
    if verbose:
        print("Model fitted.")

    if verbose > 1:
        print("Computing training R2.")
    tng_R2 = None # compute_R2(tng_series, model, horizon, start, verbose > 2)
    if verbose > 1:
        print("Computing validation R2.")
    val_R2 = compute_R2(val_series, model, horizon, start, verbose > 2)

    if verbose:
        print(f"mean R2 validation: {val_R2.mean()}")

    return model, tng_R2, val_R2


def main():
    """Train model using parameters dict and save results."""
    parser = argparse.ArgumentParser()
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

    params_raw = load_params(args.param)
    params = deepcopy(params_raw)
    if args.base_dir:
        params["tng_data_file"] = os.path.join(args.base_dir, params["tng_data_file"])
        params["val_data_file"] = os.path.join(args.base_dir, params["val_data_file"])

    standardize = params["standardize"] if "standardize" in params else False
    tng_series = load_darts_timeseries(
        params["tng_data_file"],
        params["tng_task_filter"],
        standardize,
    )
    val_series = load_darts_timeseries(
        params["val_data_file"],
        params["val_task_filter"],
        standardize,
    )
    data = tng_series, val_series
    if not tng_series or not val_series:
        print("No tng_data or val_data found.")
        return None
    if args.verbose:
        print("Time series loaded.")

    model, tng_R2, val_R2 = train(params, data, verbose=args.verbose)

    out_dir = check_path(args.output_dir)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params_raw, f, indent=4)
    np.save(os.path.join(out_dir, "tng_R2.npy"), val_R2)
    np.save(os.path.join(out_dir, "val_R2.npy"), val_R2)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pk.dump(model, f)
    if args.verbose:
        print(f"Results and model saved in {out_dir}")


if __name__ == "__main__":
    main()
