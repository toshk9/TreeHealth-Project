import json
import datetime
from typing import List, Dict, Any

from hyperopt import Trials

import sys

sys.path.append("..")


def default_converter(o):
    """
    Converts an object to a JSON-serializable format.

    This function is intended for use as the `default` parameter in JSON encoding.
    It handles objects that are not natively serializable by the json module:
      - If the object has an `item` method, it returns the result of `o.item()`.
      - If the object is a datetime or date instance, it returns its ISO formatted string.
      - Otherwise, it raises a TypeError indicating that the object is not serializable.

    Args:
        o (Any): The object to convert.

    Returns:
        Any: A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object cannot be converted to a serializable format.

    Example:
        >>> import datetime
        >>> default_converter(datetime.datetime(2025, 2, 19))
        '2025-02-19T00:00:00'
    """
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def save_trials(trials_obj: Trials, filename: str) -> None:
    """
    Saves trial data to a JSON file.

    This function serializes the `trials` attribute of a Trials object and writes it to a JSON file.
    It uses a custom converter (default_converter) to handle non-serializable objects.
    The file is saved in the "../hyperopt/" directory with the given filename.

    Args:
        trials_obj (Trials): The Trials object containing trial data.
        filename (str): The name of the file to save the trial data.

    Returns:
        None

    Example:
        >>> save_trials(my_trials, "trials.json")
    """
    with open("../hyperopt/" + filename, "w") as f:
        json.dump(trials_obj.trials, f, indent=2, default=default_converter)


def load_all_trials(filename: str) -> List[Dict[Any, Any]]:
    """
    Loads all trial records from a JSON file.

    This function reads a JSON file containing trial records and returns them.
    The JSON file may store either a list of trial records or a dictionary representing a single trial.

    Args:
        filename (str): The path to the JSON file containing trial records.

    Returns:
        List[Dict[Any, Any]]: A list of trial records, or a single trial record if the file contains a dictionary.

    Example:
        >>> trials = load_all_trials("trials.json")
    """
    with open(filename, "r") as f:
        trials = json.load(f)
    return trials


def load_best_trial(filename: str) -> Dict[Any, Any]:
    """
    Loads the best trial from a JSON file based on the minimum loss.

    This function reads trial data from a JSON file using `load_all_trials`. If the file contains a list
    of trials, it selects and returns the trial with the smallest loss value from the 'result' field.
    If the file contains a single trial record (as a dictionary), that record is returned.
    A ValueError is raised if the file content is in an unexpected format.

    Args:
        filename (str): The path to the JSON file containing trial records.

    Returns:
        Dict[Any, Any]: The trial record with the minimum loss value, or the single trial record if only one is present.

    Raises:
        ValueError: If the loaded data is neither a list nor a dictionary.

    Example:
        >>> best_trial = load_best_trial("trials.json")
        >>> print(best_trial["result"]["loss"])
    """
    trials = load_all_trials(filename)
    if isinstance(trials, list):
        best_trial = min(trials, key=lambda t: t["result"]["loss"])
    elif isinstance(trials, dict):
        best_trial = trials
    else:
        raise ValueError("Unexpected format in file.")
    return best_trial
