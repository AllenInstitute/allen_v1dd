# Stimulus analysis changelogs
<!-- ## `v2`
**Released:** TBD

**Available at:** TBD

-  -->

## `v1`
**Released:** Wednesday, April 19, 2023

**Available at:** `/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/stimulus_analyses/stimulus_analyses_v1.h5` (4.27 GB)

- **(!!!) Plane IDs now start at 0 instead of 1!** This it to resolve consistency with session NWB files.
- Fixes bug where DG `osi` and `gosi` metrics are copies of `dsi`.
- `running_modulation` added (1d array for running modulation for each ROI).
- DG `pref_cond` and `pref_cond_index` now populated for all valid ROIs (previously was only responsive ROIs).
    - Note that this means SSI can be computed for every valid ROI. As a caveat, you may notice a shift in distribution if your inclusion is, e.g., >= 25% responsive trials. (You won't get a distribution shift if your inclusion is >= 50% responsive trials because this was the previous inclusion for `pref_cond`, a prerequisite to SSI.)
- Added natural images and natural movie stimulus analyses. (Thanks, David!)
- Added `ssi_tuning_fit` metric for SSI computed from the fitted orientation tuning curves.

## `v0`
**Released:** Friday, April 7, 2023

**Available at:** `/allen/programs/mindscope/workgroups/surround/v1dd_in_vivo_new_segmentation/data/stimulus_analyses/stimulus_analyses_v0.h5` (3.81 GB)

- First version
- DG analyses (windowed + full-field)
- LSN analyses (using basic detection; a later version will use chi-squared tests and Gaussian fits)