# Compiling stimulus analyses
This documentation explains how to compile stimulus analyses on the V1DD calcium imaging data.

An **example workflow** is given at the bottom of this file.

## Motivation
The purpose of this is to compile all session stimulus analyses (e.g., DG responses, OSI, DSI, receptive fields, etc.) into a single HDF5 (.h5) file to be used in analyses. This helps speed up analysis by enabling users to reference a single pre-compiled analysis file to identify ROI responses. It also can be continuously updated without breaking code, so as newer receptive field mapping techniques are applied, for example, old code that leverages RF center locations will still work on the updated methods.

## Code: How to make a stimulus analysis save information to H5 file
The `StimulusAnalysis` class (`allen_v1dd/stimulus_analysis/stimulus_analysis.py`) has a `save_to_h5` method that takes in a `group` parameter, which is the H5 file group for the particular stimulus. By default, this saves stimulus information such as the stimulus name and trace type (e.g., DFF, events; whatever trace was used for analysis) to the group attributes.

In subclasses (e.g., `DriftingGratings`), this method can be overridden—make sure to call the superclass method to save these attributes, however! The method can then save relevant information to these H5 files; for examples see the `DriftingGratings` and `LocallySparseNoise` classes.

## Compiling: How to compile stimulus analyses into an H5 file
Analyzing stimuli is likely going to be an expensive process that takes time to run. Thus, running this on a beefy machine is advised. I (Chase) have been running it on Michael's machine which has 32 CPUs. Even here, however, they can 6+ hours to run in parallel.

The running code is saved in `data_processing/run_stimulus_analyses.py`. The parallel processing is written in native Python, though it may in hindsight have been easier to use public packages (David mentioned Ray was a good one?).

Scripts to execute this are stored in `scripts/`. Running the following will execute stimulus analyses in the current shell window:
```bash
# Note: This script by default saves to
# OUTPUT_FILE="/mnt/drive2/chase/v1dd/v1dd_stim_analyses"
# where it will create a unique timestamped subdirectory
sh scripts/stim_analysis_isilon.sh
```
To execute the analyses in a new screen (so as not to occupy the current shell process), run the following: `sh scripts/stim_analysis_isilon_screen.sh`.

## Example workflow
1. Changes are made to the `save_to_h5` method in a `StimulusAnalysis` subclass that require recomputing these analyses.
1. `git push` these changes.
1. Log into an Allen Institute machine. (Chase has been using Michael's machine)
1. Clone this (allen_v1dd) GitHub repository into the machine.
1. `cd` into the `allen_v1dd` directory.
1. `git pull` to make sure it is up-to-date.
1. `sh scripts/stim_analysis_isilon_screen.sh` **(Before doing this, it is advised to edit the `scripts/stim_analysis_isilon` variable `OUTPUT_FILE` to change the save path)**
1. Once the script finishes running, a single h5 file will be saved in the save directory (which the script will print out). Once this has been rigorously tested (save it locally and view file with an HDF5 viewer to make sure everything saved properly and is correct), the file can be copied into the V1DD data directory on Isilon to be shared with others.