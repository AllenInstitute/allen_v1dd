import numpy as np
import pandas as pd
import scipy.stats as st
# import tifffile as tf

from .stimulus_analysis import StimulusAnalysis

class NaturalScenes(StimulusAnalysis):
    """Used to analyze natural scene stimulus.

    1. Can we differentiate responses to the 12 natural scenes (40 repeats) presented to the animal?
    2. Which cells contribute the most to decoding
    """

    def __init__(self, session, plane, trace_type="events"):
        super(NaturalScenes, self).__init__("natural_images_12", "ni", session, plane, trace_type)

        # TODO

    @property
    def duration(self):
        return self.stim_meta["duration"]