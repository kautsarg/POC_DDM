import numpy as np


def time_to_index(times: list, time_vect: np.ndarray) -> list:
    """
    Returns the index of the time closest to each desired time in the input list from the time_vect array.
    This function takes a list of desired times and an array of sampling times. For each desired time,
    it finds the index of the closest time in the time_vect array.

    Arguments
    ---------
    times : list
        list of integers containing the desired times
    time_vect : np.array
        array of the sampling times

    Returns
    -------
    list
        A list of indices where each index corresponds to the position in time_vect that is closest
        to each desired time in the input list.
    """
    indices = []
    for time in times:  # for each time in the input list
        indices.append( np.argmin(np.abs(time_vect - time)) )
        # find index of the sampled time (in time_vect) closest to the desired one (time)
    return indices


def find_time_difference(time1, time2):
    """
    Find the difference in minutes between time2 and time1 (If time1>time2 the difference will be negative)

    Parameters
    ----------
    time1: int
        Time 1 in the form hhmm
    time2: int
        Time 2 in the form hhmm
    Returns
    -------
    int
        min_diff, the difference in minutes between the two times
    """
    time1, time2 = int(time1), int(time2)  # Ensure the input is integer
    hour_diff = time2 // 100 - time1 // 100 - 1  # Difference between hours
    min_diff = time2 % 100 + (60 - time1 % 100)  # Difference between minutes
    min_diff = hour_diff*60+min_diff  # Output in minutes
    return min_diff

