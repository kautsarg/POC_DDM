import numpy as np
from scipy.signal import convolve
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from titan_v4 import functions


def split_chem_and_temp(arr, chip='titan'):
    """ Separates the data of temperature and chemical pixels. The temperature pixels are isolated in the new array;
    In the chemical output array temperature pixels are replaced by the average of the 24 surrounding chemical pixels.
    An array of booleans idx_active_pixels is also returned, where the temperature pixels are indicated as inactive.

    Parameters
    ----------
    arr : np.array
        3D array of chemical and temperature pixels (1 temp pixel for each 24 chem pixels)
    chip : str
        Optional. String specifying which chip was used to the experiment; default is 'titan'.
        Can be changed to 'titanicks'.

    Returns
    -------
    tuple
        (arr_temp, arr_chem, idx_active_temp) where
        - arr_temp: 3D array of the temperature pixels
        - arr_chem: 3D array of the chemical pixels, where the temperature pixels are replaced by a chemical value that
         is the average of the surrounding 24 chemical pixels
         - idx_active_temp: array of booleans corresponding to the 2d chem array where all temperature pixels are False
        (inactive)
        """
    if chip == 'titanicks':
        arr_temp = arr[1::3, 1::3, :]  # obtain the temperature array by selecting one pixel every 3

        mask = np.ones((3, 3, 1)) / 8
        mask[1, 1, 0] = 0
        # mask = 1 1 1
        #        1 0 1
        #        1 1 1

        # 2D convolution of a signal witgth the mask above
        # results in an output signal where each value is the average of the surrounding ones in the input signal
        av_3d = convolve(arr, mask, mode='same')  # perform convolution
        arr_chem = arr.copy()  # copy the original to preserve the original chemical pixels
        arr_chem[1::3, 1::3, :] = av_3d[1::3, 1::3, :]  # replace the temp pixels with the average found by convolution

        #  Create array to consider the temperature pixels as inactive
        idx_active_temp = np.full((arr_chem.shape[0], arr_chem.shape[1]), True)
        idx_active_temp[1::3, 1::3] = False
        idx_active_temp = idx_active_temp.reshape(-1, order='C')

    elif chip == 'titan':
        arr_temp = arr[2::5, 2::5, :]  # obtain the temperature array by selecting one pixel every 3

        mask = np.ones((5, 5, 1)) / 24
        mask[2, 2, 0] = 0
        # mask = 1 1 1 1 1
        #        1 1 1 1 1
        #        1 1 0 1 1
        #        1 1 1 1 1
        #        1 1 1 1 1

        # 2D convolution of a signal with the mask above
        # results in an output signal where each value is the average of the surrounding ones in the input signal
        av_3d = convolve(arr, mask, mode='same')  # perform convolution
        arr_chem = arr.copy()  # copy the original to preserve the original chemical pixels
        arr_chem[2::5, 2::5, :] = av_3d[2::5, 2::5, :]  # substitute temp pixels with the average found by convolution

        #  Create array to of bool values where the chem_pixels==1 and temp_pixels==0
        idx_active_temp = np.full((arr_chem.shape[0], arr_chem.shape[1]), True)
        idx_active_temp[2::5, 2::5] = False
        idx_active_temp = idx_active_temp.reshape(-1, order='C')

    else: raise f'RAISED: The type of chip specified ({chip}) does not exist :('

    return arr_temp, arr_chem, idx_active_temp


def filter_by_vrange(x):
    """
    Identifies active pixels by checking that all the values are in v_range, where v_range = [min(X)+50, max(X)-50]

    Parameters
    ---------
    x : np.array
        Input 2D array (T x NM). T = time samples, NM = total number of pixels

    Returns
    -------
    np.array
        1D array of bool with dimension (NM). For each pixel, returns True if the output is always in v_range
    """
    v_range = [np.amin(x) + 50, np.amax(x) - 50]
    # for each pixel, check if all the values are within the given range
    # n_pixels_below_max = (x < v_range[1]).all(axis=0)  # debug
    # n_pixels_above_min = (x > v_range[0]).all(axis=0)  # debug
    n_samples_below_min_per_pixel = (x < v_range[0]).sum(axis=0)
    there_are_3orless_samples_below_min = (n_samples_below_min_per_pixel <= 3)
    # there_are_3orless_samples_below_min_totalpixels = there_are_3orless_samples_below_min.sum()  # debug
    return (x < v_range[1]).all(axis=0) & there_are_3orless_samples_below_min


def filter_by_vref(X, v_thresh=70):
    '''
    Identifies active pixels by checking if one of the first 10 derivatives d(i) is > v_thresh.
    This should be used for the experiments where the reference electrode voltage is ramped up and down
    at the start of the readout. If the pixels are responding to the reference electrode voltage variation,
    they are in contact with the solution.

    Parameters
    ---------
    X : np.array
        Input 2D array (T x NM). T = time samples, NM = total number of pixels
    v_thresh : int, optional
        Optional. Minimum value of the derivative d(i)=X(i+1)-X(i) in mV. Default is 70

    Returns
    -------
    np.array
        1D array of bool with dimension (NM). For each pixel, returns True if, during the first 10 samples,
        one of the derivatives is > v_thresh. The derivatives are calculated as d(i) = X(i+1)-X(i)
    '''
    return (np.diff(X[:10, :], axis=0) > v_thresh).any(
        axis=0)  # check if one of the first 10 derivatives is >v_thresh


def filter_by_gain(x, delta_threshold=50, range_threshold=10, plt_show=False, plt_save=False, exp_path=None):
    delta = x[:, :, 1] - x[:, :, 3]
    idx_active_vref = delta > delta_threshold

    x_max = np.max(x)
    x_min = np.min(x)
    idx_active_vrange = ~np.any(x[:, :, :4] < (x_min+range_threshold), axis=2) & ~np.any(x[:, :, :4] > (x_max-range_threshold), axis=2)

    
    fig, ax = plt.subplots(3, 2, figsize=(5, 10))
    fig.suptitle(f'{str(exp_path)}')
    ax[0, 0].imshow(idx_active_vref, cmap="viridis")
    ax[0, 0].set(title="idx_active (vref)")
    ax[0, 1].imshow(idx_active_vrange, cmap="viridis")
    ax[0, 1].set(title="idx_active (vrange)")
    ax[1, 0].plot(x.reshape(-1, x.shape[2]).T)
    ax[1, 0].set(title="gain (all)")
    ax[1, 1].plot(x.reshape(-1, x.shape[2]).T[:, idx_active_vref.reshape(-1)])
    ax[1, 1].set(title="gain (vref)")
    ax[2, 0].plot(x.reshape(-1, x.shape[2]).T[:, idx_active_vrange.reshape(-1)])
    ax[2, 0].set(title="gain (vrange)")
    ax[2, 1].plot(x.reshape(-1, x.shape[2]).T[:, idx_active_vref.reshape(-1) & idx_active_vrange.reshape(-1)])
    ax[2, 1].set(title="gain (vref & vrange)")
    plt.tight_layout()
    if plt_save:
        plt.savefig(Path(exp_path, "gain_summary.png"))
        print(f'Image gain_summary.png for experiment {exp_path} saved.')
    if plt_show:
        plt.show()

        # # TMP PLOT FOR THESIS
        # fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=200)
        # yplt=x.reshape(-1, x.shape[2]).T[:, idx_active_vref.reshape(-1) & idx_active_vrange.reshape(-1)]
        # ax.plot(np.vstack((yplt[0:4,:], yplt[6:8, :])))
        # ax.set(title="Three-point calibration for active chemical pixels", xlabel="sample", ylabel=r"$\tau$")
        # #plt.grid()
        # plt.tight_layout()
        # plt.savefig("lin_threepoint.png")
        # plt.show()

    return idx_active_vref.reshape(-1) & idx_active_vrange.reshape(-1)


def filter_by_derivative(x, vthresh_max=100, vthresh_min=0.1):
    """ Identifies active pixels by checking that the absolute value of the derivative is always below vthresh

    Parameters
    ----------
    x : ndarray
        input 2D array of shape TxNM
    vthresh : int
        threshold for active pixels. Default is 5

    Returns
    -------
    ndarray
        1D array of bool with dimension (NM). For each pixel, returns True if all the derivatives are below vthresh
    """
    x_diff = np.abs(np.diff(x, axis=0))
    return (x_diff < vthresh_max).all(axis=0) & (x_diff > vthresh_min).any(axis=0)


def find_settled_time(time_vect, X, bounds_sec=(0, 10*60)):
    """
    Finds the beginning of the experiment as the time in (bounds_sec) with the highest disturbance
    in the average signal, and allows 10 samples for settling.

    Parameters
    ----------
    time_vect: ndarray
        array of time stamps
    X: ndarray
        2d array of all pixel data, with shape TTIMESxPIXELS
    bounds_sec: tuple
        Optional. tuple indication beginning and end of the search interval in seconds;
        default is (0, 10*60) to check the first 10 minutes of the experiment.

    Returns
    -------
    tuple
        settled_index, settled_time:
        - settled_idx: index of the beginning of the experiment
        - settled_time: time of the beginning of the experiment in seconds

    """
    search_start, search_end = functions.time_to_index(bounds_sec, time_vect)  # for each time in bounds, find the index
    # of the sample (in time_vect) that is closest to the desired one (in bounds)
    X_mean = np.mean(X, axis=1)  # for each sample, calculate the mean of all pixels
    X_mean_diff = np.diff(X_mean)  # find the derivative

    loading_index = np.argmax(X_mean_diff[search_start:search_end]) + search_start + 1  # find the index
    # where the derivative is max in the specified interval
    settled_index = loading_index + 15  # add settling time
    settled_time = time_vect[settled_index]  # find the time that index corresponds to
    return loading_index, settled_index, settled_time


def find_start_end_temp(chem_1d, temp_1d, exp_path):
    # IDX_START
    n_temp = int(temp_1d.shape[0] // 2)
    mean_settled_temp = np.mean(temp_1d[n_temp:])
    start_temp = temp_1d[0]
    temp_variation = mean_settled_temp - start_temp
    in_temp = (temp_1d > (start_temp + temp_variation * 0.90))
    idx_start = np.nonzero(in_temp)[0][0]

    # IDX_END
    in_range = (temp_1d > (start_temp + temp_variation * 0.95)) & (temp_1d < (start_temp + temp_variation * 1.05))
    # check if goes out of range at the end, remove those samples
    idx_end = np.nonzero(in_range)[0][-1]  # todo: the end should be based on the max min
    
    # IDX_SETTLED
    # # find the 1st value not in range from the end
    # not_in_range = ~in_range
    # idx_settled = np.nonzero(not_in_range[:idx_end])[0][-1]+1
    # also, check that the Vref is not modified after: if there is an index with derivative > ..
    # chem_diff = np.diff(chem_1d)
    # max_in_derivative = np.max(chem_diff[idx_settled:idx_settled+30])  # todo not the max but the latest
    # if max_in_derivative > 10:
    #     idx_settled += (np.argmax(chem_diff[idx_settled:idx_settled+30]) + 5)
    chem_diff = np.abs(np.diff(chem_1d[idx_start:idx_start+50]))
    idx_vref_change = np.where(chem_diff > 20)[0]
    idx_settled = idx_start + idx_vref_change[-1] + 3 if idx_vref_change.size > 0 else idx_start

    # # DEBUG PLT
    print(f"DEBUG START_SETTLED_END: idxstart {idx_start}, idxsettled {idx_settled}, idxend {idx_end}")
    fig, ax = plt.subplots(1, 1)
    ax.plot(temp_1d)
    # ax.plot(chem_1d)
    ax.axvline(idx_start, color="g", label="start")
    ax.axvline(idx_settled, color="k", label="settled")
    ax.axvline(idx_end, color="r", label="end")
    ax.axhline((start_temp + temp_variation * 0.95), label="95%")
    ax.axhline((start_temp + temp_variation * 1.05), label="105%")
    ax.legend()
    plt.savefig(Path(exp_path, "start_end_temp.png"))
    print(f'Image start_end_temp.png for experiment {exp_path} saved.')
    # # DEBUG END

    return idx_start, idx_settled, idx_end


def find_start_end_idx(time_vect: np.ndarray, arr_chem_1d: np.ndarray, arr_temp_1d: np.ndarray, end_time_min: int, start_type: str = None,
                       excel_file: Path = None, exp_path_readout: Path = None, exp_path: Path = None):
    """
    Find the start and end of the experiment.
    - idx_start is the beginning of the experiment. For a DNA experiment, this is where the TTP is calculated from
    - idx_settled. This is where the data is settled. Any processing on the data will start from this point.
    - idx_end is the end of the relevant section of the experiment.

    Parameters
    ----------
    exp_path: Path
        Path of the folder containing the experiment data
    time_vect: nd array
        array of time samples
    arr_chem_3d:
        3d array of chemical values
    start_type: str
        specifies how the start of the array is found. Can be "auto", "temperature", "excel", "None".
        - None: the experiment starts at idx=0, and this is also the settled index.
        - "temperature": experiment starts when the temperature reaches 95% of its final value.
            idx_settled is when the temperature value reamins in range 95%-105%.
            Requires input temp_1d.
        - "auto": OUTDATED Finds highest derivative in first 10 min tp find idx_start, allows 10 samples for idx_settled
        - "excel": OUTDATED The start of the experiment is specified in an excel file. This requires optional parameters
            excel_file and exp_path_readout
    end_time_min: int/double
        maximum duration of the experiment from idx_start
    excel_file = None: Path
        Optional, default None. Required when start_type="excel". Path of the excel file that specifies the beginning of
        the experiment.
    exp_path_readout = None: Path
        Optional, default None. Required when start_type="excel". Path of the readout file.
    temp_1d = None: np.ndaray
        Optional, default None. Required when start_type="temperature". 1d array of the mean temperature in time.

    Returns
    -------
    tuple
        idx_start, idx_settled, idx_end
    """
    if start_type is None:  # None: the experiment is already loaded at the beginning of the time series
        idx_start, start_time = 0, 0
        idx_settled = 0

    elif start_type == 'temperature':
        idx_start, idx_settled, idx_end = find_start_end_temp(arr_chem_1d, arr_temp_1d, exp_path)


    elif start_type == 'excel':  # input form excel file (OUTDATED)
        excel_df = pd.read_excel(excel_file, sheet_name='Sheet1')
        excel_df['ID'] = excel_df['ID'].astype(str)
        excel_df.set_index('ID', inplace=True)
        exp_path_str = str(exp_path_readout)
        experiment_id = exp_path_str[exp_path_str.rfind('v2')+3:exp_path_str.rfind('VReadout')-1]
        start_time = excel_df.loc[experiment_id, 'injection_time']
        idx_start = functions.time_to_index([start_time], time_vect)[0]
        idx_settled = idx_start

    else:  # other inputs are not accepted
        raise ValueError(f'injection_type = {start_type} is not a valid input :( Choose auto, manual, temperature or None')

    start_time = time_vect[idx_start]
    idx_end = functions.time_to_index([end_time_min * 60 + start_time], time_vect)[0]
    
    return idx_start, idx_settled, idx_end


def split_wells(array_3d, n_wells):

    if n_wells == 2:
        top_well_3d, bot_well_3d = np.array_split(array_3d, 2)
        all_well_3d = [top_well_3d, bot_well_3d]

    elif n_wells == 4 or n_wells == 6 or n_wells == 10:
        rows = np.array_split(array_3d, n_wells//2, axis=0)
        all_well_3d = []
        for row in rows:
            row_wells = np.array_split(row, 2, axis=1)
            all_well_3d.extend(row_wells)

    else:
        raise f"n_wells can not be {n_wells}. Allowed values are 1, 2, 4, 6, 10"

    return all_well_3d


def split_wells_idxactive(idx_active, n_wells, nrows=290, ncols=204):

    if n_wells == 2:
        idx_active = idx_active.reshape(nrows, ncols, order='C')
        idx_active_top, idx_active_bot = np.array_split(idx_active, 2)
        all_idx_active = [idx_active_top.reshape(-1, order='C'), idx_active_bot.reshape(-1, order='C')]

    elif n_wells == 4 or n_wells == 6 or n_wells == 10:
        idx_active = idx_active.reshape(nrows, ncols, order='C')
        all_idx_active = []
        rows_idx_active = np.array_split(idx_active, n_wells//2, axis=0)
        for row_idx in rows_idx_active:
            row_idx_active = np.array_split(row_idx, 2, axis=1)
            all_idx_active.extend(row_idx_active)
        all_idx_active = [well_idx.reshape(-1, order='C') for well_idx in all_idx_active]

    else:
        raise f"n_wells can not be {n_wells}. Allowed values are 1, 2, 4, 6, 10"

    return all_idx_active

