import re
import h5py
import torch
import argparse
import numpy as np
import os
from tqdm import tqdm

from src.data.load_data import load_params
from src.tools import load_model
from src.models.predict_model import predict_horizon

SUBS = [f"sub-0{i}" for i in range(1, 7)]


def main(args):
    """Load darts model and compute score on several tasks data."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Working on {device}.")

    for sub in tqdm(SUBS, desc="subjects"):
        model_path = (
            args.model
            if os.path.isfile(args.model)
            else os.path.join(args.model, "model.pkl")
        )
        model_path = re.sub("sub-0[1-6]", sub, model_path)
        params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
        data_file_path = re.sub("sub-0[1-6]", sub, args.data_file)
        data_file = h5py.File(data_file_path, "r")
        all_runs = list(data_file.keys())
        valid_runs = [
            run
            for run in all_runs
            if data_file[run][:].shape[0] > params["seq_length"]
            and re.search(args.task_filter, run)
        ]
        data_file.close()
        out_dir = args.output
        if out_dir is None:
            suffixe = (
                os.path.splitext(os.path.split(args.data_file)[1])[0]
                + "_"
                + args.task_filter
            )
            out_dir = os.path.join(
                os.path.dirname(model_path), f"predict_horizon_{suffixe}"
            )
        out_dir = (
            out_dir + "_" + sub if not re.search("sub-0[1-6]", out_dir) else out_dir
        )
        out_dir = re.sub("sub-0[1-6]", sub, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        model = load_model(model_path)
        if isinstance(model, torch.nn.Module):
            model.to(torch.device(device)).eval()

        batch_size = params["batch_size"] if "batch_size" in params else 100

        for run in tqdm(valid_runs, desc="runs"):
            r2, Z, Y = predict_horizon(
                model,
                params["seq_length"],
                6,
                data_file_path,
                run,
                batch_size,
                stride=1,
                standardize=False,
            )

            suffixe = run.split("_space")[0]
            np.save(os.path.join(out_dir, f"r2_{suffixe}.npy"), r2)
            np.save(os.path.join(out_dir, f"Z_{suffixe}.npy"), Z)
            np.save(os.path.join(out_dir, f"Y_{suffixe}.npy"), Y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory path."
    )
    parser.add_argument("--model", "-m", type=str, help="Path to model file or dir.")
    parser.add_argument(
        "--task_filter", "-t", type=str, help="Regex string to filter runs."
    )
    parser.add_argument("--data_file", "-f", type=str, help="Path to data HDF5 file.")
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, help="Level of verbosity."
    )
    args = parser.parse_args()
    main(args)
