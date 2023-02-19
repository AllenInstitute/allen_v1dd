from os import path
import numpy as np
from glob import glob
import h5py

from .ophys_session import OPhysSession

class OPhysClient:
    """Represents a V1DD ophys client."""

    # Set of (mouse, column, volume) that have failed preprocessing
    # This is copied from the white paper (p. 8)
    # Actually this is from the previous data
    # FAILED_PREPROCESSING = [
    #     # Slc2
    #     (409828, 1, 5),
    #     (409828, 2, 1),

    #     # Teto1
    #     (416296, 1, 2),
    #     (416296, 2, 1),

    #     # Slc4
    #     # All volumes included
        
    #     # Slc5
    #     (427836, 2, 5),
    # ]

    def __init__(self, database_path: str=None):
        """Initialize a new V1DD client.

        Args:
            database_path (str): Path to physiology database
        """
        if database_path is None:
            database_path = path.join("allen", "programs", "mindscope", "workgroups", "surround", "v1dd_in_vivo_new_segmentation", "data")
            print("Defaulting to V1DD data in allen drive:", database_path)

        self.database_path = database_path
        self._nwb_files = None

    def get_session_id(self, nwb_file) -> str:
        """Get the corresponding session id from a file

        Args:
            nwb_file (any): nwb file

        Returns:
            str: session id from the nwb file
        """
        if type(nwb_file) is str:
            filename = path.basename(nwb_file) # structured as "X_X_date.nwb"
            end_index = filename.index("_", filename.index("_")+1) # index of second _
            return filename[:end_index]
        elif type(nwb_file) is h5py.File:
            return self.get_session_id(nwb_file.filename)
        else:
            return None

    def _update_file_cache(self):
        # Update nwb_files in cache
        self._nwb_files = glob(path.join(self.database_path, "nwbs", "**", "*.nwb"), recursive=True) # look in subfolders

    def get_all_session_ids(self) -> list:
        """Get all session IDs in the database.

        Returns:
            list: List of all session IDs in the database.
        """
        self._update_file_cache()
        # nwb_files = glob(path.join(self.database_path, "nwbs", "*.nwb")) # look in parent directory
        session_ids = [self.get_session_id(f) for f in self._nwb_files]
        session_ids.sort()
        return session_ids

    def load_ophys_session(self, session_id: str=None, mouse: int=None, column: int=None, volume: any=None, log=None) -> OPhysSession:
        """Load ophys session data

        Args:
            session_id (str): Session ID (or None if supplying mouse, column, and volume). For example, "M409828_13".
            mouse (int): Mouse ID
            column (int): Imaging column ID
            volume (any): Volume ID
            log (any): Output for session loading logs. Defaults to None (i.e., no logs).

        Returns:
            OPhysSession: Loaded ophys session data, or None if no data with the matching parameters was found.
        """
        if session_id is None:
            session_id = f"M{mouse}_{column}{volume}"

        if log is not None: log(f"Loading session {session_id}")
        if self._nwb_files is None:
            self._update_file_cache()
        
        for file_path in self._nwb_files:
            if self.get_session_id(file_path) == session_id:
                # Found a matching NWB file
                if log is not None: log(f"Found matching file {file_path}")
                ophys_session = OPhysSession(self, session_id, file_path)
                if log is not None: log(f"Session loaded")
                return ophys_session

        return None