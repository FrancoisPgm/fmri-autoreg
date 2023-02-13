import os
import re
import argparse
import numpy as np
import nibabel as nib
import pickle as pk
from glob import glob
from tqdm import tqdm
import nilearn.interfaces
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker, NiftiSpheresMasker
from src.data.load_data import load_params
from src.tools import load_model
from src.data.preprocess import LOAD_CONFOUNDS_PARAMS
from src.models.predict_model import predict_horizon

SEEDS = (
    ("visual", (-16, -74, 7)),
    ("sensorimotor", (-41, -20, 62)),
    ("dorsal_attention", (-34, -38, 44)),
    ("ventral_attention", (-5, 15, 32)),
    ("fronto-parietal", (-40, 50, 7)),
    ("default-mode", (-7, -52, 26)),
)

HORIZON = 6


def main(args):
    file_list = glob(
        f"{args.data_dir}/{args.dataset}/{args.subject}/ses-*/func/"
        "*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    file_list = [p for p in file_list if re.search(args.task_filter, p)]
    confounds, _ = nilearn.interfaces.fmriprep.load_confounds(file_list, **LOAD_CONFOUNDS_PARAMS)

    model_path = (
        args.model if os.path.splitext(args.model)[1] else os.path.join(args.model, "model.pkl")
    )
    model = load_model(model_path)
    params = load_params(os.path.join(os.path.dirname(model_path), "params.json"))
    batch_size = params["batch_size"] if "batch_size" in params else 100

    seeds_ROI = {}
    for seed in SEEDS:
        seed_masker = NiftiSpheresMasker(
            [seed[1]],
            radius=1,
            detrend=False,
            standardize=False,
            verbose=0,
        )
        ROI_id = int(seed_masker.fit_transform(args.atlas_path)) - 1
        # -1 because ROIs are numbered from 1
        seeds_ROI[seed[0]] = ROI_id

    os.makedirs(args.out_dir, exist_ok=True)

    for i, file_path in tqdm(enumerate(file_list)):
        predictions = {}
        filename = os.path.split(file_path)[1].replace(".nii.gz", "")

        _, pred, atlas_series = predict_horizon(
            model, params["seq_length"], HORIZON, args.data_file, filename, batch_size
        )

        brain_mask_path = file_path.replace("desc-preproc_bold", "desc-brain_mask")

        brain_masker = NiftiMasker(
            mask_img=brain_mask_path,
            detrend=True,
            standardize="zscore",
            smoothing_fwhm=5,
        )

        brain_time_series = brain_masker.fit_transform(file_path, confounds=confounds[i])
        brain_time_series = brain_time_series[params["seq_length"] :]
        for seed, n_ROI in seeds_ROI.items():
            seed_time_series = atlas_series[:, n_ROI, 0]
            seed_vox_corr = (
                np.dot(brain_time_series[: -HORIZON + 1].T, seed_time_series)
                / seed_time_series.shape[0]
            )
            out_path = os.path.join(args.out_dir, f"{filename}_{seed}_original_connectivity.nii.gz")
            brain_masker.inverse_transform(seed_vox_corr.T).to_filename(out_path)
            for lag in range(HORIZON):
                seed_time_series = pred[:, n_ROI, lag]
                last_vol = len(brain_time_series) - HORIZON + 1 + lag
                seed_vox_corr = (
                    np.dot(brain_time_series[lag:last_vol].T, seed_time_series)
                    / seed_time_series.shape[0]
                )
                out_path = os.path.join(
                    args.out_dir, f"{filename}_{seed}_prediction_{lag}_connectivity.nii.gz"
                )
                brain_masker.inverse_transform(seed_vox_corr.T.astype(np.float32)).to_filename(
                    out_path
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, help="Output directory.")
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Base directory for data files."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Path to the hdf5 file with data projected on atlas.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset folder to use for voxel data.",
    )
    parser.add_argument("--task_filter", type=str, help="Regular expression to select runs.")
    parser.add_argument("--subject", type=str, help="Subject to use (e.g. 'sub-01').")
    parser.add_argument(
        "--model",
        type=str,
        help="Path to the model to use (similar models with different lags will be also used).",
    )
    parser.add_argument(
        "--atlas_path",
        type=str,
        default="data/external/tpl-MNI152NLin2009bSym/tpl-MNI152NLin2009bSym_res-03_atlas-MIST_desc-197_dseg.nii.gz",
    )
    args = parser.parse_args()
    main(args)
