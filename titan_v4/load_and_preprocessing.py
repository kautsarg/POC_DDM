import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from titan_v4 import functions
import titan_v4.load_functions as load
import titan_v4.preprocessing_functions as preprocess
import titan_v4.linearise as linearise
from titan_v4.Well import Well
from titan_v4.Experiment import Experiment
from titan_v4.plt_Experiment_summary import titan_plt_summary

def titan_load_and_preprocessing(exp_path: Path,
                                 n_wells=1, start_type=None, n_a_type="v04", end_time_min=30,
                                 print_status=False, plt_gain_calib=False, save_gain_calib=False,
                                 nrows=290, ncols=204):
    """
    Load and pre-process the data from the .bin input files exp_path_calib and exp_path_readout

    Parameters
    ----------
    exp_path: Path
        Path of the experiment folder
    n_wells: int
        Number of wells in the experiment. Optional, Default is 1; can be 1, 2, 4, 6, 10.
    start_type: str
        string specifying the type of injection time detection. Can be 'manual', 'auto', None. Default None.
        'manual' means finding the injection time in a csv file in the mother folder;
        'auto' means finding the highest derivative in the first 10 minutes of the experiment;
        None means that the experiment is already injected at the beginning of the recording.
    print_status: bool
        if True, print updates on the state of loading and preprocessing. Optional, Default False.

    Returns
    -------
    Experiment
        object of type Experiment
    """

    #  1) LOAD .BIN DATA
    #  1A) READOUT: Find most recent readout file
    exp_path_readout = load.find_most_recent_readout(exp_path)
    if print_status: print(f'EXP PATH {exp_path} \nREADOUT PATH {exp_path_readout} \nLoad data... ')

    # Load data from bin files: TIME to time_vect and CHEM DATA to frame_3d (readout)
    data_readout_list = load.binary_file_read(exp_path_readout)
    time_vect, frame_3d = load.list_to_numpy_readout(data_readout_list, nrows=nrows, ncols=ncols, n_a_type=n_a_type)

    # Time: check if the time resets during the experiment. If so, fix time axis.
    time_vect = load.unwrap_time_vect(time_vect)

    # 1B) ACTIVE PIXELS: Load data from bin files
    exp_path_idx_active = load.find_corresp_file(exp_path, exp_path_readout, 'idx_active')
    idx_active_list = load.binary_file_read(exp_path_idx_active)
    idx_active_input = np.array(idx_active_list[:nrows*ncols]).reshape(nrows, ncols, order='F').reshape(nrows*ncols, order='C')

    # 1C) GAIN FILE + LOG FILE & LINEARISATION
    exp_path_idx_active = load.find_corresp_file(exp_path, exp_path_readout, 'gain')
    gain_list = load.binary_file_read(exp_path_idx_active)
    if n_a_type == "v01" or n_a_type == "v02" or n_a_type == "v03":
        arr_gain_3d, vref = load.list_to_numpy_gain(gain_list, n_a_type=n_a_type, exp_path=exp_path, exp_path_readout=exp_path_readout, plt_show=False)
    elif n_a_type == "v04":
        arr_gain_3d, vref = load.list_to_numpy_gain(gain_list, n_a_type=n_a_type, plt_show=False)
    else:
        raise f"RAISED titan_load_and_preprocessing: Impossible to call list_to_numpy_gain with n_a_type {n_a_type}"
    idx_active_gain = preprocess.filter_by_gain(arr_gain_3d, plt_show=plt_gain_calib, plt_save=save_gain_calib, exp_path=exp_path)
    params = linearise.generate_params(frame_3d, arr_gain_3d, vref)
    frame_3d_lin, idx_active_lin = linearise.linearise(frame_3d, params, idx_active_gain)

    #  2) PREPROCESSING
    if print_status: print(f'Data loaded. \nPreprocessing start...')

    #  Split temperature and chemical data. Temperature pixels are inactive.
    arr_temp_3d, arr_chem_3d, idx_active_temp = preprocess.split_chem_and_temp(frame_3d, chip='titan')

    # The active pixels are those found active by lacewing (idx_active_input)
    # & are not temperature pixels (idx_active_temp)
    idx_active = (idx_active_input == 400) & idx_active_temp & idx_active_gain & idx_active_lin

    # Split wells and create 2d arrays for each
    if n_wells == 1:
        all_well_3d, all_well_temp_3d = [arr_chem_3d], [arr_temp_3d]
        all_well_gain_3d = [arr_gain_3d]
        all_idx_active = [idx_active]
        all_well_3d_lin = [frame_3d_lin]
    elif n_wells == 2 or n_wells == 4 or n_wells == 6 or n_wells == 10:
        all_well_3d = preprocess.split_wells(arr_chem_3d, n_wells)  # 1.CHEM DATA
        all_well_temp_3d = preprocess.split_wells(arr_temp_3d, n_wells)  # 2.TEMP DATA
        all_well_gain_3d = preprocess.split_wells(arr_gain_3d, n_wells)  # 3.GAIN DATA
        all_idx_active = preprocess.split_wells_idxactive(idx_active, n_wells)  # 4.ACTIVE PIXELS
        all_well_3d_lin = preprocess.split_wells(frame_3d_lin, n_wells)  # 5.LINEARISED 3D CHEM DATA

    else:
        raise f"RAISED: n_wells = {n_wells} is not a valid input :( Choose 1, 2, 4, 6 or 10"

    # Find beginning and end of the experiment
    temp_2dNL = arr_temp_3d.reshape(-1, arr_temp_3d.shape[2]).T  # shape: (T, Rtemp*Ctemp)

    # Calibration constants
    a = 1913.44
    b = 26.68
    c = np.min(arr_temp_3d) - 0.1
    d = 0.76

    new = -(1 / b) * np.log((arr_temp_3d - c) / a) + d # nonlinear → linear transform
    temp_2d = new.reshape(-1, new.shape[2]).T
    idx_active_temp = (np.mean(temp_2dNL, axis=0) > 10)
    temp_1d = np.mean(temp_2d[:, idx_active_temp], axis=1)

    idx_start, idx_settled, idx_end = preprocess.find_start_end_idx(time_vect,
                                                                    np.mean(arr_chem_3d.reshape(-1, arr_chem_3d.shape[2], order='C').T, axis=1),
                                                                    temp_1d, end_time_min, start_type, exp_path=exp_path)

    # # DEBUG PLOT FOR LINEARISATION
    # fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    # frame_2d_bs = frame_3d_lin.reshape(-1, frame_3d_lin.shape[2], order='C').T - frame_3d_lin.reshape(-1, frame_3d_lin.shape[2], order='C').T[0, :]
    # ax[0].plot(np.mean(frame_2d_bs, axis=1)[idx_settled:])
    # ax[0].set(title="New linearisarion", xlabel="sample", ylabel="amplitude")
    #
    # arr_tmp = arr_chem_3d[:, :, idx_settled:]
    # a = 24328.12
    # b = -0.010848
    # c = np.min(arr_tmp)-0.1
    # new = 1 / b * np.log((arr_tmp - c) / a)
    # temp_2d = new.reshape(-1, new.shape[2]).T
    # temp_1d = np.mean(temp_2d, axis=1)
    # ax[1].plot(temp_1d)
    # ax[1].set(title="Old linearisarion", xlabel="sample", ylabel="amplitude")
    # plt.tight_layout()
    # plt.show()
    # # DEBUG END

    # Save relevant data for each well
    all_well_summary = []
    for i_well in range(n_wells):
        # This is required to find active pixels in the correct section of the experiment
        n_times = all_well_3d[i_well][:, :, idx_settled:idx_end].shape[2]
        well_2d_exp = all_well_3d[i_well][:, :, idx_settled:idx_end].reshape(-1, n_times, order='C').T
        # Filter active pixels by Vrange: Each pixel needs to be in range for the entire duration of the experiment
        idx_active_range = preprocess.filter_by_vrange(well_2d_exp) & preprocess.filter_by_derivative(well_2d_exp)
        all_idx_active[i_well] = all_idx_active[i_well] & idx_active_range

        # Initialise well instance
        this_well = Well(time_vect, all_well_3d[i_well], all_well_temp_3d[i_well], all_well_gain_3d[i_well], all_well_3d_lin[i_well], idx_start, idx_settled, idx_end, all_idx_active[i_well])
        all_well_summary.append(this_well)

    # Initialise Experiment instance
    experiment = Experiment(all_well_summary, exp_path_str=str(exp_path_readout))
    experiment.temperature_3d = arr_temp_3d  # OPTIONAL: saving temperature 3d array

    if print_status: print('Preprocessing end.')
    return experiment


