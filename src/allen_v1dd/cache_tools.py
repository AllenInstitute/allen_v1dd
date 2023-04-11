"""
NOTE: This is currently NOT THREAD SAFE. Do not try to access the same cache key on multiple threads!
"""

from os import path
import time
from datetime import timedelta
import pickle

DEFAULT_CACHE_DIR = "../../cache"
DEFAULT_FILE_AGE_EXP = timedelta(days=7)
NEVER_OUTDATED = lambda filename, loaded_file_data: False

def is_file_too_old(filename, loaded_file_data, max_age=DEFAULT_FILE_AGE_EXP) -> bool:
    """Returns True if a file's last modified time is older than max_age

    Args:
        filename (str): Path to the file
        loaded_file_data (any): Loaded data from the pickle file.
        max_age (float or timedelta, optional): Max age for a cache object. Can be a float (representing total seconds) or timedelta object. Defaults to 7 days.

    Returns:
        bool: True if a file's last modified time is older than max_age
    """

    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    file_last_modified = path.getmtime(filename)
    now = time.time()
    return file_last_modified + max_age < now

def get_filename(data_key, cache_dir):
    return path.join(cache_dir, f"{data_key}.pickle")

def get(data_key: str, load_new_data_fn, is_outdated_fn=is_file_too_old, is_outdated_fn_kwargs=dict(), cache_dir=DEFAULT_CACHE_DIR):
    """Fetches data from a cache if it is unexpired, or loads it into the cache.

    Args:
        data_key (str): Name of the data to be stored. Must be unique, as this is used as a cache key.
        load_new_data_fn (fn): Function that returns the data object to be stored in the cache. Takes no arguments. Usually an expensive operation.
        is_outdated_fn (fn, optional): Function that takes in arguments (filename, loaded_file_data, in addition to any kwargs,
                                       and returns whether the cached object is outdated. Defaults to checking if the file hasn't been updated in 7 days.
        is_outdated_fn_kwargs (dict, optional): Keyword arguments to be passed to is_outdated_fn. Defaults to no kwargs.
        cache_dir (str, optional): Cache directory. Defaults to DEFAULT_CACHE_DIR.
    """
    filename = get_filename(data_key, cache_dir)

    if path.exists(filename):
        with open(filename, "rb") as file:
            try:
                loaded_file_data = pickle.load(file)

                if not is_outdated_fn(filename, loaded_file_data, **is_outdated_fn_kwargs):
                    # Return the loaded object if it is valid
                    return loaded_file_data
            except:
                print(f"Error while loading {data_key} from cache; reloading object.")
    else:
        # If the file doesn't exists, we need to cache it
        print(f"{data_key} not found in cache; saving.")

    # Cache and return the data
    new_data = load_new_data_fn()
    save(data_key, new_data, cache_dir=cache_dir)

    return new_data

def save(data_key: str, data, cache_dir=DEFAULT_CACHE_DIR):
    filename = get_filename(data_key, cache_dir)
    with open(filename, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)