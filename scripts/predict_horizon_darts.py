import re
import h5py
import torch
import argparse
import numpy as np
import os
import pickle as pk
from tqdm import tqdm
from darts import TimeSeries
from src.data.load_data import load_params, load_darts_timeseries
from src.tools import load_model
from src.models.predict_model import predict_horizon

SUBS = [f"sub-0{i}" for i in range(1, 7)]
SUBS = ["sub-01"]


def main():
    "Load model and compute score on several tasks data."
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory path.")
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

    model_path = args.model if os.path.isfile(args.model) else os.path.join(args.model, "model.pkl")
    for sub in tqdm(SUBS, desc="subjects"):

        model_path = re.sub("sub-0[1-6]", sub, model_path)
        params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
        data_file_path = re.sub("sub-0[1-6]", sub, args.data_file)
        model = load_model(model_path)

        in_length = params["model_params"]["input_chunk_length"]
        start = params["model_params"]["input_chunk_length"]

        out_dir = args.output
        if out_dir is None:
            suffixe = os.path.splitext(os.path.split(args.data_file)[1])[0] + "_" + args.task_filter
            out_dir = os.path.join(os.path.dirname(model_path), f"predict_horizon_{suffixe}")
        out_dir = out_dir + "_" + sub if not re.search("sub-0[1-6]", out_dir) else out_dir
        out_dir = re.sub("sub-0[1-6]", sub, out_dir)
        os.makedirs(out_dir, exist_ok=True)

        data_file = h5py.File(data_file_path, "r")
        all_runs = list(data_file.keys())
        valid_runs = [
            run
            for run in all_runs
            if data_file[run][:].shape[0] > in_length and re.search(args.task_filter, run)
        ]

        for run in valid_runs:
            series = load_darts_timeseries(data_file_path, run)[0]
            series_np = series.values()
            chunk_list = [
                series_np[k : k + in_length] for k in range(series_np.shape[0] - 6 - in_length + 1)
            ]
            chunk_list = [TimeSeries.from_values(c) for c in chunk_list]
            for _ in range(6):
                pred_list = []
                new_chunk_list = []
                for chunk in chunk_list:
                    pred = model.predict(1, chunk)
                    pred_list.append(pred.values())
                    new_chunk_list.append(np.concatenate((chunk.values()[1:], pred.values())))
                pred_per_lag.append(np.concatenate(pred_list))
                chunk_list = [TimeSeries.from_values(c) for c in new_chunk_list]

            r2 = []
            for l, pred in enumerate(pred_per_lag):
                r2.append(
                    r2_score(
                        series_np[in_length + l : len(ts_np) - 6 + l + 1],
                        pred,
                        multioutput="raw_values",
                    )
                )
            suffixe = run.split("_space")[0]
            np.save(os.path.join(out_dir, f"r2_{suffixe}.npy"), r2)


if __name__ == "__main__":
    main()
