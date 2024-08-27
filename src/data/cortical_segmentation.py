import numpy as np
import nibabel as nb
from nilearn import image
from nilearn import datasets


def _load_aal():
    aal = datasets.fetch_atlas_aal()
    img_aal = nb.load(aal.maps)
    mtx_aal = img_aal.get_fdata()
    list_val = np.unique(mtx_aal)
    return img_aal, mtx_aal, list_val


def _expand_parcels(img_aal, mtx_aal, list_val):
    all_val = np.zeros(np.append(mtx_aal.shape, list_val.shape[0]))
    for ind, val in enumerate(list_val):
        if ind > 0:
            parcel = mtx_aal == val
            img_parcel = image.new_img_like(img_aal, parcel)
            img_parcel = image.smooth_img(img_parcel, fwhm=10)
            all_val[:, :, :, ind] = img_parcel.get_fdata()
    return all_val


def _resample_segmentation(source_img, target_img):
    img_r = image.resample_to_img(source_img, target_img, interpolation="nearest")
    return img_r


def cortical_segmentation(target_img):
    """
    Generate loose binary mask of the cortex, cerebellum, and central
    structures (thalami, basal ganglia).
    """
    img_aal, mtx_aal, list_val = _load_aal()
    all_val = _expand_parcels(img_aal, mtx_aal, list_val)
    mtx_extra = list_val[np.argmax(all_val, axis=3)]
    mask_central = (mtx_extra >= 7000) & (mtx_extra < 8000)
    mask_cerebellum = mtx_extra >= 9000
    mask_cortex = (mtx_extra > 0) & ~mask_central & ~mask_cerebellum
    img = dict.fromkeys(["central", "cerebellum", "cortex"])
    img["central"] = _resample_segmentation(
        image.new_img_like(img_aal, mask_central), target_img
    )
    img["cerebellum"] = _resample_segmentation(
        image.new_img_like(img_aal, mask_cerebellum), target_img
    )
    img["cortex"] = _resample_segmentation(
        image.new_img_like(img_aal, mask_cortex), target_img
    )
    return img


def get_cortical_ROI_indices(atlas_path):
    """Get indices of the ROIs in each cortical region."""
    cort_mask = cortical_segmentation(atlas_path)
    atlas_data = nb.load(atlas_path).get_fdata()
    cort_indices_overlap = {}
    cort_indices_no_overlap = {}
    # Get ROI indices in each cortical region (with possibly same ROI in multiple cortical region)
    for region, img in cort_mask.items():
        indices, counts = np.unique(
            atlas_data[img.get_fdata().astype(bool)], return_counts=True
        )
        cort_indices_overlap[region] = dict(zip(indices, counts))
        cort_indices_no_overlap[region] = []
    # Re-assign ROI indices in the region that contain the most voxels of this ROI,
    # to have each ROI in onlny one cortical region
    for i in range(1, int(atlas_data.max()) + 1):
        region = None
        count = 0
        for reg in cort_indices_overlap:
            if i in cort_indices_overlap[reg] and cort_indices_overlap[reg][i] > count:
                region = reg
                count = cort_indices_overlap[reg][i]
        if region is not None:
            cort_indices_no_overlap[region].append(i)
    return cort_indices_no_overlap


def apply_cort_seg(data, cort_idx, mean=False):
    """Average the values of 'data' per cortical region."""
    seg_data = {}
    for region, indices in cort_idx.items():
        indices = [
            i - 1 for i in indices
        ]  # -1 because MIST regions are numbered from 1
        seg_data[region] = data[indices].mean() if mean else data[indices]
    return seg_data
