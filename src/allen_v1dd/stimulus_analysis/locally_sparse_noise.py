from os import path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
# from statsmodels.sandbox.stats.multicomp import multipletests

from .stimulus_analysis import StimulusAnalysis

def get_imshow_extent(azimuths, altitudes):
    return [azimuths[0], azimuths[-1], altitudes[0], altitudes[-1]]

class LocallySparseNoise(StimulusAnalysis):
    """Used to analyze the locally sparse noise stimulus.

    1. Does a cell have receptive field?
    2. What is a cell's receptive field?
    """
    
    def __init__(self, session, plane, trace_type="dff"):
        super().__init__("locally_sparse_noise", "lsn", session, plane, trace_type)

        self.authors = "Chase King" # TODO Fill in with Reza's group once they share code

        if trace_type == "dff":
            self.baseline_time_window = (-1, 0) # Baseline is the 1 sec window before (note this spans 3 different stimulus frames)
            self.response_time_window = (0, 4*self.time_per_frame) # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
            # self.response_time_window = (0, 1) # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
        else:
            self.baseline_time_window = None
            # self.response_time_window = (0, 2*self.time_per_frame) # 0.33 sec stimulus duration
            self.response_time_window = (0, 4*self.time_per_frame) # Span to the next stimulus

        self.pixel_on = 255
        self.pixel_off = 0
        self.pixel_gray = 127

        self.n_shuffles = 10000
        self.frac_sig_trials_thresh = 0.25

        self._frame_images = None
        self._sweep_responses = None
        self._design_matrix = None
        self._trial_template = None
        self._receptive_fields = None
        self._rf_centers = None
        self._imshow_extent = None

    def save_to_h5(self, group):
        super().save_to_h5(group)

        group.attrs["trace_type"] = self.trace_type
        group.attrs["frac_sig_trials_thresh"] = self.frac_sig_trials_thresh
        group.attrs["image_shape"] = self.image_shape
        group.attrs["azimuths"] = self.azimuths
        group.attrs["altitudes"] = self.altitudes
        group.attrs["grid_size"] = self.grid_size

        # Is responsive
        is_responsive = np.zeros((self.n_rois, 3), dtype=bool)
        for roi in range(self.n_rois):
            if self.is_roi_valid[roi]:
                resp_on = self.has_receptive_field(roi, rf_type="on")
                resp_off = self.has_receptive_field(roi, rf_type="off")
                is_responsive[roi, :] = [resp_on, resp_off, resp_on or resp_off]
        ds = group.create_dataset("is_responsive", data=is_responsive)
        ds.attrs["columns"] = ["has_rf_on", "has_rf_off", "has_rf_on_or_off"]

        # Receptive fields
        ds = group.create_dataset("receptive_fields", data=self.receptive_fields)
        ds.attrs["dimensions"] = ["roi", "on_off", "row", "column"]
        
        # RF centers
        ds = group.create_dataset("rf_centers", data=self.rf_centers)
        ds.attrs["dimensions"] = ["roi", "on (0) and off (1)", "azimuth (0) and altitude (1) (deg)"]

        # RF centers (computed by argmax)
        rf_centers_argmax = np.full((self.n_rois, 2, 2), np.nan)
        rois, onoffs, alts, azis = np.where(self.receptive_fields)
        for roi in np.unique(rois):
            for onoff in np.unique(onoffs[rois == roi]):
                rf = self.receptive_fields[roi, onoff]
                azi, alt = np.unravel_index(rf.argmax(), rf.shape) # note the ordering
                alt, azi = self.point_to_alt_azi(alt_ctr=alt+0.5, azi_ctr=azi+0.5) # Add 0.5 to center in pixel
                rf_centers_argmax[roi, onoff, :] = (azi, alt)
        ds = group.create_dataset("rf_centers_argmax", data=rf_centers_argmax)
        ds.attrs["dimensions"] = ["roi", "on (0) and off (1)", "azimuth (0) and altitude (1) (deg)"]

        

    @property
    def duration(self):
        return self.stim_meta["duration"]
    
    @property
    def grid_size(self):
        return self.stim_meta["grid_size"]
    
    @property
    def frame_images(self):
        if self._frame_images is None:
            self._load_frames()
        return self._frame_images
    
    @property
    def azimuths(self):
        if self._azimuths is None:
            self._load_frames()
        return self._azimuths
    
    @property
    def altitudes(self):
        if self._altitudes is None:
            self._load_frames()
        return self._altitudes

    @property
    def imshow_extent(self):
        if self._imshow_extent is None:
            self._imshow_extent = get_imshow_extent(azimuths=self.azimuths, altitudes=self.altitudes)
        return self._imshow_extent

    @property
    def trial_template(self):
        if self._trial_template is None:
            self._trial_template = self.frame_images[self.stim_table.frame.values]
        return self._trial_template

    @property
    def n_sweeps(self):
        return len(self.stim_table)

    @property
    def n_pixels(self):
        return self.frame_images.shape[1] * self.frame_images.shape[2]
    
    @property
    def image_shape(self):
        return self.frame_images.shape[1], self.frame_images.shape[2]
    
    @property
    def design_matrix(self):
        """
        The design matrix is a matrix of shape (2*n_pixels, n_sweeps) that contains
        information about what pixels are ON or OFF in different stimulus showings.
        A stimulus sweep is a particular showing of a stimulus (that need not be unique).
        
        Specifically, the first half of rows are indexed by (i, j) where i is 1 iff pixel i is ON in stimulus condition j.
        The second half of rows are indexed by (n_pixels+i, j) where i is 1 iff pixel i is OFF in stimulus condition j.

        Returns:
            np.ndarray: Of shape (2*n_pixels, n_sweeps)
        """
        if self._design_matrix is None:
            stim_pixels = self.frame_images[self.stim_table.frame.values].reshape((-1, self.n_pixels))
            design_matrix_on = np.where(stim_pixels == self.pixel_on, True, False)
            design_matrix_off = np.where(stim_pixels == self.pixel_off, True, False)
            self._design_matrix = np.concatenate((design_matrix_on, design_matrix_off), axis=1)
            self._design_matrix = self._design_matrix.T # shape (2*n_pixels, n_sweeps)

        return self._design_matrix

    # def convolve(self, img, sigma=4):
    #     """
    #     2D Gaussian convolution.

    #     Copied from https://github.com/AllenInstitute/AllenSDK/blob/9ef5214dcb04a61fe4c04bf19a5cb13c9e1b03f1/allensdk/brain_observatory/receptive_field_analysis/utilities.py#L56
    #     """
    #     from scipy.interpolate import interp2d
    #     from scipy.ndimage import gaussian_filter
    #     from skimage.measure import block_reduce

    #     if img.sum() == 0:
    #         return img

    #     img_pad = np.zeros((3 * img.shape[0], 3 * img.shape[1]))
    #     img_pad[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]] = img

    #     x = np.arange(3 * img.shape[0])
    #     y = np.arange(3 * img.shape[1])
    #     g = interp2d(y, x, img_pad, kind='linear')

    #     if img.shape[0] == 16:
    #         upsample = 4
    #         offset = -(1 - .625)
    #     elif img.shape[0] == 8:
    #         upsample = 8
    #         offset = -(1 - .5625)
    #     else:
    #         raise NotImplementedError
            
    #     ZZ_on = g(offset + np.arange(0, img.shape[1] * 3, 1. / upsample), offset + np.arange(0, img.shape[0] * 3, 1. / upsample))
    #     ZZ_on_f = gaussian_filter(ZZ_on, float(sigma), mode='constant')

    #     z_on_new = block_reduce(ZZ_on_f, (upsample, upsample))
    #     z_on_new = z_on_new / z_on_new.sum() * img.sum()
    #     z_on_new = z_on_new[img.shape[0]:2 * img.shape[0], img.shape[1]:2 * img.shape[1]]

    #     return z_on_new

    @property
    def design_matrix_blur(self):
        # TODO
        # for stim_condition_index in range(design_matrix.shape[1]):
        #     design_matrix[:lsn.n_pixels, stim_condition_index] = convolve(design_matrix[:lsn.n_pixels, stim_condition_index].reshape(lsn.image_shape)).flatten()
        #     design_matrix[lsn.n_pixels:, stim_condition_index] = convolve(design_matrix[lsn.n_pixels:, stim_condition_index].reshape(lsn.image_shape)).flatten()
        return None

    @property
    def sweep_responses(self):
        """
        Sweep responses is a np.ndarray of shape (n_stim_showings, n_rois) where the value at
        position (i, j) is the jth ROI's response to the ith stimulus shown.
        """

        if self._sweep_responses is None:
            self._sweep_responses = np.zeros((len(self.stim_table), self.n_rois), dtype=float)
            for i in self.stim_table.index:
                start = self.stim_table.at[i, "start"]
                self._sweep_responses[i] = self.get_responses(start, self.baseline_time_window, self.response_time_window, self.trace_type)

        return self._sweep_responses
    
    @property
    def receptive_fields(self):
        """
        Array of shape (n_rois, 2, n_image_rows, n_image_columns) where each entry is the fraction of significant responses
        at each pixel. Dimension 1 corresponds to ON (0) and OFF (1). Values less than self.frac_sig_trials are set to zero.
        """
        if self._receptive_fields is None:
            design_matrix_int = self.design_matrix.astype(int) # shape (2*n_pixels, n_sweeps)
            n_pixel_trials = self.design_matrix.sum(axis=1) # shape (2*n_pixels,)
            roi_boot_95 = np.quantile(self.get_spont_null_dist(self.baseline_time_window, self.response_time_window, n_boot=self.n_shuffles, trace_type=self.trace_type, cache=False), 0.95, axis=1) # shape (n_rois,)
            sig_sweep_responses = self.sweep_responses > roi_boot_95 # shape (n_sweeps, n_rois)
            frac_sig_pixel_responses = design_matrix_int.dot(sig_sweep_responses).T / n_pixel_trials # shape (n_rois, 2*n_pixels)
            frac_sig_pixel_responses[frac_sig_pixel_responses < self.frac_sig_trials_thresh] = 0 # Zero out pixels below significance threshold
            frac_sig_pixel_responses[~self.is_roi_valid] = 0 # Zero out invalid ROIs
            self._receptive_fields = frac_sig_pixel_responses.reshape(self.n_rois, 2, *self.image_shape)

        return self._receptive_fields
    
    @property
    def rf_centers(self):
        """
        Array of shape (n_rois, 2, 2). Dimension 1 corresponds to ON (0) and OFF (1). Dimension 2 corresponds to
        azimuth (0) and altitude (1). Values of np.nan mean the ROI does not have a given RF.
        """
        if self._rf_centers is None:
            self._rf_centers = np.full((self.n_rois, 2, 2), np.nan)

            # Only iterate over ROIs with an RF
            rois, onoffs, alts, azis = np.where(self.receptive_fields)
            
            for roi in np.unique(rois):
                roi_mask = rois == roi
                for onoff in np.unique(onoffs[roi_mask]):
                    mask = roi_mask & (onoffs == onoff)
                    alt, azi = self.point_to_alt_azi(alt_ctr=np.mean(alts[mask]) + 0.5, azi_ctr=np.mean(azis[mask]) + 0.5) # Add 0.5 to center in pixel
                    self._rf_centers[roi, onoff, :] = (azi, alt)

        return self._rf_centers

    def has_receptive_field(self, roi, rf_type=None):
        if rf_type is None:
            rf = self.receptive_fields[roi]
        else:
            rf = self.receptive_fields[roi, self._rf_type_idx(rf_type)]
        return bool(rf.max() >= self.frac_sig_trials_thresh) # otherwise it is a numpy type


    def _rf_type_idx(self, rf_type):
        if type(rf_type) is int:
            return rf_type
        
        if rf_type == "on":
            return 0
        elif rf_type == "off":
            return 1
        else:
            raise ValueError(f"Bad rf_type: {rf_type}")
            

    def _load_frames(self):
        lsn_frames_file = path.join(self.session.v1dd_client.database_path, "stim_movies", "lsn_9deg_28degExclusion_jun_256.npy")
        all_frame_images = np.load(lsn_frames_file)
        
        # Incorrect stimulus:
        # from tifffile import tifffile
        # lsn_frames_file = path.join(self.session.v1dd_client.database_path, "stim_movies", "stim_locally_sparse_nois_16x28.tif")
        # all_frame_images = tifffile.imread(lsn_frames_file)

        self._frame_images = all_frame_images[:self.stim_table["frame"].max()+1]

        _, nrows, ncols = all_frame_images.shape
        self._azimuths = (np.arange(ncols) - ncols//2 + 0.5) * self.grid_size
        self._altitudes = (np.arange(nrows) - nrows//2 + 0.5) * self.grid_size

    def get_stim_indices_from_frames(self, frames: list):
        return self.stim_table.index[self.stim_table["frame"].isin(frames)]
    
    def point_to_alt_azi(self, alt_ctr, azi_ctr):
        # Assumes lsn.altitudes and lsn.azimuths are equally-spaced (which is true in our case; 9.3 deg spacing)
        alt = alt_ctr * (self.altitudes[-1] - self.altitudes[0]) / len(self.altitudes) + self.altitudes[0]
        azi = azi_ctr * (self.azimuths[-1] - self.azimuths[0]) / len(self.azimuths) + self.azimuths[0]
        return alt, azi

    def plot_rf(self, rf, rf_type, desc=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        is_on = self._rf_type_idx(rf_type) == 0
        ax.imshow(rf, cmap=("Reds" if is_on else "Blues"), interpolation="none", origin="lower", vmin=0, vmax=0.5, extent=self.imshow_extent)
        # ax.set_xticks(ticks=self.azimuths, labels=[f"{azi:.0f}" for azi in self.azimuths])
        # ax.set_yticks(ticks=self.altitudes, labels=[f"{alt:.0f}" for alt in self.altitudes])
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(f"{'ON' if is_on else 'OFF'} receptive field{'' if desc is None else f' ({desc})'}", color=("red" if is_on else "blue"))
        ax.axvline(x=0, color="lightgray", linewidth=0.5, zorder=0)
        ax.axhline(y=0, color="lightgray", linewidth=0.5, zorder=0)
        return ax
    