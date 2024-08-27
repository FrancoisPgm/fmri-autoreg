import os
import csv
import re
import numpy as np
import argparse

from src.models.train_model import train
from src.models.predict_model import predict_horizon
from src.data.load_data import load_data, load_params, make_input_labels


HORIZON = 6
rng = np.random.default_rng(2022)


def main(args):
    """Train a model with different amount of training data and save scores."""
    params = load_params(args.param)
    batch_size = params["batch_size"] if "batch_size" in params else 100

    fieldnames = list(params.keys()) + [
        "r2_mean_tng",
        "r2_std_tng",
        "r2_mean_val",
        "r2_std_val",
        "sub",
        "n_vol",
    ]

    if not os.path.exists(args.output):
        with open(args.output, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

    sub = args.sub
    params["sub"] = sub
    params["tng_data_file"] = re.sub("sub-0[1-6]", sub, params["tng_data_file"])
    params["val_data_file"] = re.sub("sub-0[1-6]", sub, params["val_data_file"])
    tng_data = load_data(params["tng_data_file"], params["tng_task_filter"])
    val_data = load_data(params["val_data_file"], params["val_task_filter"])
    tng_data_trial = tng_data[: args.n]
    n_vol = sum([d.shape[0] for d in tng_data_trial])

    X_tng, Y_tng, X_val, Y_val, edge_index = make_input_labels(
        tng_data_trial,
        val_data,
        params["seq_length"],
        params["time_stride"],
        params["lag"],
        compute_edge_index=True,
    )
    print(sub, f"{args.n} runs, {n_vol} volumes, training shape: {X_tng.shape}")
    trial_data = (X_tng, Y_tng, X_val, Y_val, edge_index)
    model, r2_tng = train(params, data=trial_data, verbose=args.verbose)[:2]
    params["r2_mean_tng"] = r2_tng.mean()
    params["r2_std_tng"] = r2_tng.std()
    params["n_vol"] = n_vol

    r2_val = predict_horizon(
        model,
        params["seq_length"],
        HORIZON,
        params["val_data_file"],
        params["val_task_filter"],
        batch_size,
    )[0]

    for i in range(HORIZON):
        params["lag"] = i + 1
        params["r2_mean_val"] = r2_val[i].mean()
        params["r2_std_val"] = r2_val[i].std()

        with open(args.output, "a") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writerow(params)
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, help="Output path.")
    parser.add_argument(
        "--param",
        "-p",
        type=str,
        help="Path to the json file containing the hyper-parameters.",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, help="Level of verbosity."
    )
    parser.add_argument("-n", type=int, help="Number of runs to use.")
    parser.add_argument("--sub", type=str, help="Subject.")
    args = parser.parse_args()
    main(args)
