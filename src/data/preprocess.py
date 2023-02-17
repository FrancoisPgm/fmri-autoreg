import argparse
import h5py
import os
from glob import glob
import nilearn.input_data
import nilearn.interfaces
import pickle as pk
from tqdm import tqdm


LOAD_CONFOUNDS_PARAMS = {
    "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
    "motion": "basic",
    "wm_csf": "basic",
    "global_signal": "basic",
    "demean": True,
}


def main():
    """Mask data, project on atlas and save preprocessed data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Path to dataset.", required=True)
    parser.add_argument("-s", "--subject", type=str, help="Subject.", required=True)
    parser.add_argument("-a", "--atlas", type=str, help="Path to the atlas.", required=True)
    parser.add_argument("-o", "--out_path", type=str, help="Output path.", required=True)
    args = parser.parse_args()

    file_list = glob(
        f"{args.dataset}/{args.subject}/ses-*/func/*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    with h5py.File(args.out_path, "a") as f:
        for in_path in tqdm(file_list, desc="Applying atlas masks"):
            filename = os.path.split(in_path)[1].replace(".nii.gz", "")
            if filename not in f.keys():
                brain_mask_path = in_path.replace("preproc_bold.nii.gz", "brain_mask.nii.gz")
                masker = nilearn.input_data.NiftiLabelsMasker(
                    args.atlas, detrend=True, standardize="zscore"
                )
                masker.set_params(mask_img=brain_mask_path)
                confounds, _ = nilearn.interfaces.fmriprep.load_confounds(
                    in_path, **LOAD_CONFOUNDS_PARAMS
                )
                processed_data = masker.fit_transform(in_path, confounds=confounds)
                f.create_dataset(filename, data=processed_data)

    print(f"All done, results in {args.out_path}")


if __name__ == "__main__":
    main()
