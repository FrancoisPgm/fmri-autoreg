import json
import h5py
import argparse
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from src.data.load_data import load_params
from src.tools import load_model
from src.models.predict_model import predict_horizon


SCALE = 197
HORIZON = 6
DATASETS_TASKS = {
    "hcptrt": [
        #        "motor",
        "restingstate",
        #        "language",
        #        "social",
        #        "gambling",
        #        "relational",
        "wm",
        #        "emotion",
    ],
    "movie10": ["bourne", "figures", "wolf", "life"],
}


def main():
    "Load model and compute score on several tasks data."
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory path.")
    parser.add_argument("--model", "-m", type=str, help="Path to model file or dir.")
    parser.add_argument(
        "--base_dir", "-b", type=str, default=None, help="Base directory for data files."
    )
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    model_path = args.model if os.path.isfile(args.model) else os.path.join(args.model, "model.pkl")
    model = load_model(model_path)
    if isinstance(model, torch.nn.Module):
        model.to(torch.device(device)).eval()
    sub = "sub-" + args.model.split("sub-")[1][:2]
    out_dir = args.output
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(model_path), "compare_tasks")
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(model_path)
    params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
    params["model_path"] = model_path
    with open(os.path.join(out_dir, "params.json"), "w") as params_out_file:
        json.dump(params, params_out_file, indent=4)
    batch_size = params["batch_size"] if "batch_size" in params else 100

    for dataset in tqdm(DATASETS_TASKS, desc="Datasets"):
        data_file_path = f"data/processed/MIST/mist_{SCALE}_{dataset}_{sub}.h5"
        data_file = h5py.File(data_file_path, "r")
        all_runs = list(data_file.keys())
        valid_runs = [run for run in all_runs if data_file[run][:].shape[0] > params["seq_length"]]
        data_file.close()
        for run in tqdm(valid_runs, desc="Runs"):
            r2, Z, Y = predict_horizon(
                model, params["seq_length"], HORIZON, data_file_path, run, batch_size
            )
            suffixe = f"{dataset}_{run.split('_space')[0]}"
            np.save(os.path.join(args.output, f"r2_{suffixe}.npy"), r2)
            np.save(os.path.join(args.output, f"Z_{suffixe}.npy"), Z)
            np.save(os.path.join(args.output, f"Y_{suffixe}.npy"), Y)


if __name__ == "__main__":
    main()
