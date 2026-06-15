import numpy as np
import struct
from pathlib import Path
import re
import matplotlib.pyplot as plt

from titan_v4 import functions


def binary_file_read(file: Path) -> list:
    """ Unpacks .bin file and returns the data in a list

    Parameters
    ----------
    file : Path
        Path of the .bin file name
    Returns
    -------
    list
        data_list : list of all the data in the input binary file
    """
    fw = open(file, 'rb')
    data_byte = fw.read()
    data_len = len(data_byte) >> 1
    data_list = []
    for n in range(0, data_len):
        (data, ) = struct.unpack('>H', data_byte[2*n:2*(n+1)])
        data_list.append(data)
    return data_list


def list_to_numpy_readout(data_list: list, frame_start: int = 0, nrows: int = 290, ncols: int = 204, n_a_type="v04") -> (np.ndarray, np.ndarray):
    """
    Converts list of the data from the .bin readout file to a numpy array

    Parameters
    ----------
    data_list : list
        list of input data to be converted into numpy array
    frame_start : int
        first frame to be considered
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    tuple
        (time_vect, frame_3d) where
        - time_vect : np.ndarray
            1D array of the sampling times
        - frame_3d : np.ndarray
            3D array of sensor array outputs. Has shape NxMxT, where NxM is (number of rows x number of columns), T is time
    """
    # Number of parameters per frame
    P = nrows * ncols  # P=number of pixels

    if n_a_type == "v04" or n_a_type == "v03" or n_a_type == "v02":
        A = data_list[P]
        if len(data_list) % (P+A) != 0:
            print(len(data_list))
            print(P)
            print(A)
            np.savetxt("lalala.txt", data_list)
            raise "RAISED list_to_numpy_readout: Unexpected number of data points in the binary file"
    elif n_a_type == "v01":
        if len(data_list) == 0:
            raise Exception("RAISED list_to_numpy_readout: Empty readout file")
        elif len(data_list) % (P+8) == 0:
            A = 8
        elif len(data_list) % (P+7) == 0:
            A = 7
        else:
            raise "RAISED list_to_numpy_readout: Could not find the number of additional parameters"
    else:
        raise f"RAISED list_to_numpy_readout: Could not find conversion for n_a_type = {n_a_type}"
    N = P + A

    frame_end = len(data_list)//N
    n_time = frame_end - frame_start
    frame_3d = np.zeros((nrows, ncols, n_time))
    time_vect = np.zeros(n_time)
    for i, n in enumerate(range(frame_start, frame_end)):
        pack = data_list[n * N:(n + 1) * N]
        frame_1d = np.array(pack[:P])
        frame_2d = frame_1d.reshape(nrows, ncols, order='F')
        frame_3d[:, :, i] = frame_2d
        if n_a_type == "v04" or n_a_type == "v03" or n_a_type == "v02":
            time_vect[i] = pack[P+1] / 10
        else:
            time_vect[i] = pack[P] / 10

    return time_vect, frame_3d


def list_to_numpy_calib(data_list: list, nrows: int = 290, ncols: int = 204) -> np.ndarray:
    """
    Converts the input data_list into a 3d representation of the calibration data

    Parameters
    ----------
    data_list: list
        list of input data imported from the calibration binary file
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    ndarray
        frame_3d: A 3D numpy array of shape (ROWS, COLS, n_frames) of calibration data
    """
    # Number of parameters per frame
    N = nrows * ncols  # number of pixels
    # there are no additional parameters

    n_frames = len(data_list)//N
    frame_3d = np.zeros((nrows, ncols, n_frames))
    for i, n in enumerate(range(n_frames)):
        pack = data_list[n * N:(n + 1) * N]
        frame_1d = np.array(pack)
        frame_2d = frame_1d.reshape(nrows, ncols, order='F')
        frame_3d[:, :, i] = frame_2d
    return frame_3d


def list_to_numpy_gain(data_list: list, nrows: int = 290, ncols: int = 204, n_a_type="v04", exp_path=None, exp_path_readout=None, plt_show=False):
    """
    Converts the input data_list into a 3d representation of the gain data

    Parameters
    ----------
    data_list: list
        list of input data imported from the gain binary file
    nrows : int
        Optional. Number of rows in the chip; default 290 for the Titan chip. Change to 78 for Titanicks
    ncols : int
        Optional. Number of columns in the chip; default 204 for the Titan chip. Change to 56 for Titanicks

    Returns
    -------
    ndarray
        frame_3d: A 3D numpy array of shape (ROWS, COLS, n_frames) of gain data
    """
    # Number of parameters per frame
    P = nrows * ncols  # P=number of pixels
    if n_a_type == "v04":
        A = data_list[P]
        vref = data_list[P+1]
        if len(data_list) % (P+A) != 0:
            raise "RAISED list_to_numpy_gain: Unexpected number of data points in the binary file"
    elif n_a_type == "v03" or n_a_type == "v02" or n_a_type == "v01":
        if len(data_list) == 0:
            raise "RAISED list_to_numpy_gain: Empty readout file"
        elif len(data_list) % (P + 4) == 0:
            A = 4
        elif len(data_list) % (P + 6) == 0:
            A = 6
        elif len(data_list) % (P + 10) == 0:
            A = 10
        else:
            raise "RAISED list_to_numpy_gain: Could not find the number of additional parameters"

        exp_path_log = find_corresp_file(exp_path, exp_path_readout, 'log')
        vref = extract_last_vref(exp_path_log)
    else: raise f"RAISED list_to_numpy_gain: Could not find list_to_numpy_gain conversion for n_a_type = {n_a_type}"
    N = P + A

    n_time = len(data_list)//N
    frame_3d = np.zeros((nrows, ncols, n_time))
    for i in range(n_time):
        pack = data_list[i * N:(i + 1) * N]
        frame_1d = np.array(pack[:P])
        frame_2d = frame_1d.reshape(nrows, ncols, order='F')
        frame_3d[:, :, i] = frame_2d

    # Check that the number of frames in the files is as expected
    if n_a_type == "v04" or n_a_type == "v03":
        if frame_3d.shape[2] != 8:
            raise f"RAISED list_to_numpy_gain: Number of frames in the gain file: frame_3d_gain has shape {frame_3d.shape}, expected (N, M, 8)"
    else:
        if frame_3d.shape[2] != 4:
            raise f"RAISED list_to_numpy_gain: Number of frames in the gain file: frame_3d_gain has shape {frame_3d.shape}, expected (N, M, 4)"

    if plt_show:
        fig, ax = plt.subplots()
        ax.plot(frame_3d.reshape(-1, frame_3d.shape[2]).T)
        plt.show()
    return frame_3d, vref


def unwrap_time_vect(timevect: np.ndarray) -> np.ndarray:
    """
    Unwraps a time vector to correct for discontinuities.
    This function takes a numpy array of time values that may contain
    samples where the time count resents. It corrects these discontinuities to create a
    smoothly increasing time vector.

    Parameters
    ----------
    timevect: np.ndarray
        A one-dimensional numpy array of time values that may contain discontinuities.

    Returns
    -------
    np.ndarray
        The corrected time vector with discontinuities removed.
    """
    unwrap_indeces = np.argwhere(np.diff(timevect) < 0)
    tmp_unwrap_compensation = 0
    for unwrap_idx in unwrap_indeces:
        timevect[int(unwrap_idx)+1:] = timevect[int(unwrap_idx)+1:] + timevect[int(unwrap_idx)] - tmp_unwrap_compensation
        tmp_unwrap_compensation = timevect[int(unwrap_idx)]
    return timevect


def find_corresp_file(exp_path: Path, exp_path_readout: Path, type_of_file: str) -> Path:
    """
    Finds the Path of the closest .bin calibration file preceding the readout exp_path_readout in exp_path.

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data
    exp_path_readout: Path
        Path of the .bin readout file
    type_of_file: str
        "calibration" or "idx_active"

    Returns
    -------
    Path
        path_file, the Path of the closest calibration/active pixels bin file preceding the readout
    """
    exp_path_readout_str = str(exp_path_readout)
    readout_time = int(exp_path_readout_str[exp_path_readout_str.rfind('T') + 1: exp_path_readout_str.rfind('T') + 5])

    # FOR THE LOG FILE, LOOK FOR FILE AFTER READOUT
    if type_of_file == 'log':
        potential_filename_list = [i for i in exp_path.glob("*lacewing_log*.txt")]

        smallest_to_readout_time = 1000
        for potential_file in potential_filename_list:
            potential_file_str = str(potential_file)
            calib_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
            readout_to_file_time = functions.find_time_difference(readout_time, calib_time)
            if 0 <= readout_to_file_time < smallest_to_readout_time:
                path_file = potential_file
                smallest_to_readout_time = readout_to_file_time
        if smallest_to_readout_time == 1000:
            raise f'RAISED: Matching file {type_of_file} before readout not found.'
        else:
            return path_file

    # FOR OTHERS, LOOK FOR FILE BEFORE
    if type_of_file == 'calibration':
        potential_filename_list = [i for i in exp_path.glob("*alib*.bin")]
    elif type_of_file == 'idx_active':
        potential_filename_list = [i for i in exp_path.glob("*find_active*.bin")]
    elif type_of_file == 'gain':
        potential_filename_list = [i for i in exp_path.glob("*gain*.bin")]
    else:
        raise f"RAISED: Code for type_of_file {type_of_file} not yet implemented"

    smallest_to_readout_time = 1000
    for potential_file in potential_filename_list:
        potential_file_str = str(potential_file)
        calib_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
        file_to_readout_time = functions.find_time_difference(calib_time, readout_time)
        if 0 <= file_to_readout_time < smallest_to_readout_time:
            path_file = potential_file
            smallest_to_readout_time = file_to_readout_time
    if smallest_to_readout_time == 1000:
        raise f'RAISED: Matching file {type_of_file} before readout not found.'

    # print(f'DEBUG: Matched files are readout {exp_path_readout}, {path_file}')
    return path_file


def extract_last_vref(file_path):
    # List to store the extracted numerical values
    vref_values = []

    # Open the file and read lines
    with open(file_path, 'r') as file:
        for line in file:
            # Look for the pattern "vref: ~ <number>"
            match = re.search(r'vref:\s*~\s*([-+]?\d*\.?\d+)', line, re.IGNORECASE)
            if match:
                # Convert the matched value to a number (float or int)
                value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                vref_values.append(value)

    if len(vref_values) == 0:
        raise "RAISED: The Vref value could not be found in the Lacewing log file"

    return vref_values[-1]


def find_most_recent_readout(exp_path: Path):
    """
    Finds the Path of the most recent readout file in the exp_path folder

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data

    Returns
    -------
    Path
        exp_path_readout, the Path of the most recent readout tile
    """
    filename_readout_list = [i for i in exp_path.glob("*eadout*.bin")]
    if len(filename_readout_list) == 1:
        exp_path_readout = filename_readout_list[0]
    elif len(filename_readout_list) == 0:
        raise f"RAISED: No readout file found in {exp_path}"
    else:
        for i_file, potential_file in enumerate(filename_readout_list):
            potential_file_str = str(potential_file)
            readout_time = int(potential_file_str[potential_file_str.rfind('T') + 1: potential_file_str.rfind('T') + 5])
            if i_file == 0:
                max_time = readout_time
                exp_path_readout = potential_file
            else:
                if readout_time > max_time:
                    exp_path_readout = potential_file
    return exp_path_readout

