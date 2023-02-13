import re
import json
import h5py
import torch
import argparse
import numpy as np
import os
import pickle as pk
from tqdm import tqdm
from src.data.load_data import load_params
from src.tools import load_model
from src.predict_model import predict_horizon

SUBS = [f"sub-0{i}" for i in range(1, 7)]
HORIZON = 6


def main():
    "Load model and compute score on several tasks data."
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, help="Output directory path.")
    parser.add_argument("--model", "-m", type=str, help="Path to model file or dir.")
    parser.add_argument("--task_filter", "-t", type=str, help="Regex string to filter runs.")
    parser.add_argument("--data_file", "-f", type=str, help="Path to data HDF5 file.")
    parser.add_argument(
        "--base_dir", "-b", type=str, default=None, help="Base directory for data files."
    )
    parser.add_argument("--verbose", "-v", type=int, default=0, help="Level of verbosity.")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    model_sub = "sub-" + args.model.split("sub-")[1][:2]
    sub_list = [sub for sub in SUBS if sub != model_sub]

    model_path = args.model if os.path.isfile(args.model) else os.path.join(args.model, "model.pkl")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
    params["model_path"] = model_path
    batch_size = params["batch_size"] if "batch_size" in params else 100

    for sub in sub_list:
        print(f"{model_sub}'s model predicting on {sub}'s data.")

        model_path = (
            args.model if os.path.isfile(args.model) else os.path.join(args.model, "model.pkl")
        )
        data_file_path = re.sub("sub-..", sub, args.data_file)
        data_file = h5py.File(data_file_path, "r")
        all_runs = list(data_file.keys())
        valid_runs = [
            run
            for run in all_runs
            if data_file[run][:].shape[0] > params["seq_length"] and re.search(args.task_filter, run)
        ]
        data_file.close()
        sub_dir = os.path.splitext(os.path.split(data_file_path)[1])[0]
        out_dir = args.output
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(model_path), "predict_horizon")
        os.makedirs(os.path.join(out_dir, sub_dir), exist_ok=True)
        with open(os.path.join(out_dir, sub_dir, f"params.json"), "w") as params_out_file:
            json.dump(params, params_out_file, indent=4)

        for run in tqdm(valid_runs, desc="runs"):
            r2, Z, Y = predict_horizon(
                model, params["seq_length"], HORIZON, data_file_path, run, batch_size
            )

            suffixe = run.split("_space")[0]
            np.save(os.path.join(out_dir, sub_dir, f"r2_{suffixe}.npy"), r2)
            np.save(os.path.join(out_dir, sub_dir, f"Z_{suffixe}.npy"), Z)
            np.save(os.path.join(out_dir, sub_dir, f"Y_{suffixe}.npy"), Y)
            with open(os.path.join(out_dir, sub_dir, f"params.json"), "w") as params_out_file:
                json.dump(params, params_out_file, indent=4)


if __name__ == "__main__":
    main()
