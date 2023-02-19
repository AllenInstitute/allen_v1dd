from os import path

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
# from statsmodels.sandbox.stats.multicomp import multipletests

from .stimulus_analysis import StimulusAnalysis

class LocallySparseNoise(StimulusAnalysis):
    """Used to analyze the locally sparse noise stimulus.

    1. Does a cell have receptive field?
    2. What is a cell's receptive field?
    """
    
    def __init__(self, session, plane, trace_type="dff"):
        super(LocallySparseNoise, self).__init__("locally_sparse_noise", "lsn", session, plane, trace_type)

        if trace_type == "dff":
            self.baseline_time_window = (-1, 0) # Baseline is the 1 sec window before (note this spans 3 different stimulus frames)
            self.response_time_window = (0, 4*self.TIME_PER_FRAME) # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
            # self.response_time_window = (0, 1) # Response is 0.67 sec window after (note this spans the desired stimulus along with the next one)
        else:
            self.baseline_time_window = None
            # self.response_time_window = (0, 2*self.TIME_PER_FRAME) # 0.33 sec stimulus duration
            self.response_time_window = (0, 4*self.TIME_PER_FRAME) # Span to the next stimulus

        self.pixel_on = 255
        self.pixel_off = 0
        self.pixel_gray = 127

        self.n_shuffles = 1000

        self._frame_images = None
        self._sweep_responses = None
        self._design_matrix = None
        self._trial_template = None

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
        if self._sweep_responses is None:
            self._load_responses()
        return self._sweep_responses
    
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

    def _load_responses(self):
        # lsn_dur = np.mean(self.stim_table.end - self.stim_table.start)
        # response_frame_window = get_frame_window_from_time_window(self.event_timestamps, (0, lsn_dur))
        sweep_responses = np.zeros((len(self.stim_table), self.n_rois), dtype=float)
        
        for i in self.stim_table.index:
            start = self.stim_table.at[i, "start"]
            sweep_responses[i] = self.get_responses(start, self.baseline_time_window, self.response_time_window, self.trace_type)

        self._sweep_responses = sweep_responses

    def get_stim_indices_from_frames(self, frames: list):
        return self.stim_table.index[self.stim_table["frame"].isin(frames)]
    
    def plot_rf(self, rf, is_on, desc=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        ax.imshow(rf, cmap=("Reds" if is_on else "Blues"), interpolation="none", origin="lower")
        ax.set_xticks(ticks=[x for x in range(len(self.azimuths))], labels=[f"{azi:.0f}" for azi in self.azimuths])
        ax.set_yticks(ticks=[y for y in range(len(self.altitudes))], labels=[f"{alt:.0f}" for alt in self.altitudes])
        ax.set_xlabel("Azimuth (°)", fontsize=12)
        ax.set_ylabel("Altitude (°)", fontsize=12)
        ax.set_title(f"{'ON' if is_on else 'OFF'} receptive field{'' if desc is None else f' ({desc})'}", color=("red" if is_on else "blue"))
        return ax
    