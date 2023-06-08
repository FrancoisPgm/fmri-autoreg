from pathlib import Path
import os
import torch
import inspect
import pickle as pk
import numpy as np
from glob import glob
from src.data.load_data import load_params


def check_path(path, verbose=True):
    """Check if given path (file or dir) already exists, and if so returns a
    new path with _<n> appended (n being the number of paths with the same name
    that exist already).
    """
    if isinstance(path, Path):
        path = str(path)
    path = os.path.normpath(path)
    path_wo_ext, ext = os.path.splitext(path)

    if os.path.exists(path):
        similar_paths = [p.replace(ext, "") for p in glob(path_wo_ext + "_*" + ext)]
        existing_numbers = [
            int(p.split("_")[-1]) for p in similar_paths if p.split("_")[-1].isdigit()
        ]
        n = str(max(existing_numbers) + 1) if existing_numbers else "1"
        path = path_wo_ext + "_" + n + ext
        if verbose:
            print(f"Specified path already exists, using {path} instead.")

    return path


def string_to_list(L):
    """Turn a string of comma seperated digits to a list of int."""
    return [int(l) for l in L.split(",")]


def load_model(path, n_emb=197):
    """Load a model with pickle or with torch state_dict depending on the extension."""
    ext = os.path.splitext(path)[1]
    if ext in (".pkl", ".pk"):
        with open(path, "rb") as f:
            model = pk.load(f)

    elif ext == ".pt":
        params = load_params(path.replace("model.pt", "params.json"))
        if params["model"] == "Chebnet":
            from src.models.models import Chebnet as ModelClass

            edge_index = np.load(path.replace("model.pt", "edge_index.npy"))
            params["edge_index"] = edge_index
            params["n_emb"] = n_emb
            params["seq_len"] = params["seq_length"]

        model_arg_names = inspect.getargspec(ModelClass.__init__).args[1:]
        model_kwargs = {key: params[key] for key in model_arg_names}
        model = ModelClass(**model_kwargs)
        model.load_state_dict(torch.load(path))

    else:
        raise ValueError(f"Invalid model extension: '{ext}'. Should be '.pkl', '.pk' or '.pt'.")

    return model
