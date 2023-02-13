import os
import torch
import argparse
import pickle as pk
import numpy as np
from glob import glob


def main(args):
    paths = glob(os.path.join(args.models_dir, "*/model.pk"))
    for path in paths:
        if os.path.isfile(path.replace("model.pk", "model.pt")):
            continue
        moddel = pk.load(open(path, "rb"))
        edge_index = model.edge_index.numpy()
        torch.save(model.state_dic(), path.replace("model.pk", "model.pt"))
        np.save(path.replace("model.pk", "edge_index.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, help="Directory containing model directories.")
    args = parser.parse_args()
    main(args)
