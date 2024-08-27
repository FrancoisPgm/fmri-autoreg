import os
import csv
from itertools import product as itertools_product
import argparse
from tqdm import tqdm
from copy import deepcopy

from src.models.train_model import train
from src.models.train_darts_model import train as train_darts
from src.data.load_data import (
    load_data,
    load_params,
    make_input_labels,
    load_darts_timeseries,
)
from src.tools import string_to_list


def main(args):
    print("Starting gridsearch")
    params = load_params(args.param)
    is_darts = "model_params" in params

    if args.subject:
        sub_to_replace = "sub-" + params["tng_data_file"].split("sub-")[1][:2]
        params["tng_data_file"] = params["tng_data_file"].replace(
            sub_to_replace, args.subject
        )
        params["val_data_file"] = params["val_data_file"].replace(
            sub_to_replace, args.subject
        )
    if args.atlas:
        atlas_to_replace = "mist_" + params["tng_data_file"].split("mist_")[1][:3]
        params["tng_data_file"] = params["tng_data_file"].replace(
            atlas_to_replace, args.atlas
        )
        params["val_data_file"] = params["val_data_file"].replace(
            atlas_to_replace, args.atlas
        )

    # Set default values of params if necessary
    standardize = params["standardize"] if "standardize" in params else False
    shuffle = params["shuffle"] if "shuffle" in params else False
    compute_edge_index = params["model"] == "Chebnet"
    thres = params["edge_index_thres"] if compute_edge_index else None

    # Load data
    load_data_func = load_darts_timeseries if is_darts else load_data
    print("Loading data...")
    tng_data = load_data_func(
        params["tng_data_file"],
        params["tng_task_filter"],
        standardize,
        shuffle,
    )
    print(f"Training data loaded from {params['tng_data_file']}.")
    val_data = load_data_func(
        params["val_data_file"],
        params["val_task_filter"],
        standardize,
        shuffle,
    )
    print(f"Validation data loaded from {params['val_data_file']}.")

    if not tng_data or not val_data:
        if args.verbose:
            print("No tng_data or val_data found.")
        return None

    if is_darts:
        fieldnames = (
            list(params)
            + list(params["model_params"])
            + ["r2_mean_tng", "r2_std_tng", "r2_mean_val", "r2_std_val", "lag"]
        )
        fieldnames.remove("model_params")

    else:
        fieldnames = list(params.keys()) + [
            "r2_mean_tng",
            "r2_std_tng",
            "loss_tng",
            "epoch",
            "r2_mean_val",
            "r2_std_val",
            "loss_val",
        ]

    # Check if output path already exists, if so count number of trials already
    # done to skip them and only compute the remaining ones
    n_trials_already_done = 0
    if os.path.exists(args.output):
        n_rows_per_trial = 1
        if "checkpoints" in params:
            checkpoints = string_to_list(params["checkpoints"])
            n_rows_per_trial += sum(c < params["nb_epochs"] for c in checkpoints)
        elif "horizon" in params:
            n_rows_per_trial = params["horizon"]
        elif "output_chunk_length" in params["model_params"]:
            n_rows_per_trial = params["model_params"]["output_chunk_length"]

        with open(args.output, "r") as f:
            n_trials_already_done += sum(l for l in f) // n_rows_per_trial - 1
            # -1 not to count header

    # Write header to output file if it doesn't exist already
    else:
        with open(args.output, "w") as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()

    # Make param combinations list and iterate over it
    param_dict = params["model_params"] if is_darts else params
    grid_keys = [key for key in param_dict if isinstance(param_dict[key], list)]
    grid_values = [param_dict[key] for key in grid_keys]

    param_list = tuple(itertools_product(*grid_values))
    trial_params = deepcopy(params)
    print("param list created")
    start = args.start + n_trials_already_done
    end = args.start + args.amount if args.amount else len(param_list)

    if is_darts:  # darts model
        data = tng_data, val_data
        for trials_values in tqdm(param_list[start:end]):
            for i, val in enumerate(trials_values):
                trial_params["model_params"][grid_keys[i]] = val

            model, r2_tng, r2_val = train_darts(trial_params, data, args.verbose)
            r2_tng_means, r2_tng_stds = r2_tng.mean((1, 2)), r2_tng.std((1, 2))
            r2_val_means, r2_val_stds = r2_val.mean((1, 2)), r2_val.std((1, 2))
            res_dict = deepcopy(trial_params)
            res_dict.pop("model_params")
            res_dict = {**res_dict, **trial_params["model_params"]}

            for i in range(r2_tng.shape[0]):
                res_dict["lag"] = i + 1
                res_dict["r2_mean_tng"] = r2_tng_means[i]
                res_dict["r2_std_tng"] = r2_tng_stds[i]
                res_dict["r2_mean_val"] = r2_val_means[i]
                res_dict["r2_std_val"] = r2_val_stds[i]

                with open(args.output, "a") as out_file:
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                    writer.writerow(res_dict)

    else:  # not darts model
        for trials_values in tqdm(param_list[start:end]):
            for i, val in enumerate(trials_values):
                trial_params[grid_keys[i]] = val

            data = make_input_labels(
                tng_data,
                val_data,
                trial_params["seq_length"],
                trial_params["time_stride"],
                trial_params["lag"],
                compute_edge_index,
                thres,
            )
            model, r2_tng, r2_val, _, _, _, _, losses, checkpoints = train(
                trial_params, data, verbose=args.verbose
            )

            if "nb_epochs" in trial_params:
                trial_params["epoch"] = trial_params["nb_epochs"] - 1
            trial_params["r2_mean_tng"] = r2_tng.mean()
            trial_params["r2_std_tng"] = r2_tng.std()
            trial_params["r2_mean_val"] = r2_val.mean()
            trial_params["r2_std_val"] = r2_val.std()
            if losses is not None:
                trial_params["loss_tng"] = losses["tng"][-1]
                trial_params["loss_val"] = losses["val"][-1]

            with open(args.output, "a") as out_file:
                writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                writer.writerow(trial_params)
                for checkpoint in checkpoints:
                    trial_params.update(checkpoint)
                    writer.writerow(trial_params)

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run a gridsearch.")
    parser.add_argument("--output", "-o", type=str, help="Output path.")
    parser.add_argument(
        "--param",
        "-p",
        type=str,
        help="Path to the json file containing the hyper-parameters value to try.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device for pytorch (cuda or cpu).",
    )
    parser.add_argument("--verbose", "-v", type=int, default=0)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Number of param combinations to skip.",
    )
    parser.add_argument(
        "--amount",
        type=int,
        help="Amount of param combinations to do.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        help="Subject to use if not defined, the one in the params json file will be used.",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        help="Atlas to use, if not defined, the one in the params json file will be used.",
    )
    args = parser.parse_args()
    main(args)
