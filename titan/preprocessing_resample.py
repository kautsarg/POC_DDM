import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from titan.load_and_preprocessing import titan_load_and_preprocessing


def resample1d(times, x, tsample=3):
    x_resampled = np.array([])
    t_resampled = np.array([])
    for t_current in range(int(times[0]), int(times[-1]), tsample):
        # if the sample exists use that
        idx_closest = np.argmin(np.abs(times - t_current))
        if times[idx_closest] == t_current:
            t_resampled = np.append(t_resampled, t_current)
            x_resampled = np.append(x_resampled, x[idx_closest])
        else:
            if times[idx_closest] < t_current:
                idx_before = idx_closest
                idx_after = idx_before + 1
            else:
                idx_after = idx_closest
                idx_before = idx_after - 1

            x1, x2 = x[idx_before], x[idx_after]
            t1, t2 = times[idx_before], times[idx_after]
            x_current = x1+((t_current-t1)*((x2-x1)/(t2-t1)))
            t_resampled = np.append(t_resampled, t_current)
            x_resampled = np.append(x_resampled, x_current)
    return t_resampled, x_resampled


def resample2d(times, x, tsample=3):  # x has shape Times x Pixels
    t_resampled = np.array([])
    for t_current in range(int(times[0]), int(times[-1]), tsample):
        # if the sample exists use that
        idx_closest = np.argmin(np.abs(times - t_current))
        if times[idx_closest] == t_current:
            t_resampled = np.append(t_resampled, t_current)
            if 'x_resampled' not in locals():
                x_resampled = x[idx_closest, :].reshape(1, -1)
            else:
                x_resampled = np.append(x_resampled, x[idx_closest, :].reshape(1, -1), axis=0)
        else:
            if times[idx_closest] < t_current:
                idx_before = idx_closest
                idx_after = idx_before + 1
            else:
                idx_after = idx_closest
                idx_before = idx_after - 1

            x1, x2 = x[idx_before, :], x[idx_after, :]
            t1, t2 = times[idx_before], times[idx_after]
            t_resampled = np.append(t_resampled, t_current)
            x_current = x1+((t_current-t1)*((x2-x1)/(t2-t1)))
            if 'x_resampled' not in locals():
                x_resampled = x[idx_closest, :].reshape(1, -1)
            else:
                x_resampled = np.append(x_resampled, x_current.reshape(1, -1), axis=0)
    return t_resampled, x_resampled


if __name__ == '__main__':

    # # TEST RESAMPLE1D ON ARTIFICIAL DATA
    # t = np.array([0, 1, 10, 11])
    # array = np.array([9, 6, 4, 2])
    #
    # ts, arr_sampled = resample1d(t, array)
    # print(ts, arr_sampled)
    #
    # fig, ax = plt.subplots()
    # ax.plot(t, array)
    # ax.plot(ts, arr_sampled, 'o')
    # plt.show()
    #
    # # TEST RESAMPLE 1D ON ISFET DATA
    # onedrive_path = Path("..", "..", "..", "..", "..", "Costanza", "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    # exp_folder = Path(onedrive_path, "Master Data Folder", "Calista Run Data")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    #
    # excel_path = Path(onedrive_path, "Master Data Folder", "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
    # excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
    # exp_paths = [Path(exp_folder, "D20250219_E00_C00_F4500KHz_U_LW_108_BRAF_WT_sweep_01")]  # OR THIS TO RUN ONE EXPERIMENT
    #
    # print(f"DEBUG: EXP_PATHS")
    # for i_path in range(len(exp_paths)):
    #     print(f"i_path {i_path}, path {exp_paths[i_path]}")
    #
    # for i_path, exp_path in enumerate(exp_paths):
    #     path_readout_str = str(exp_path)
    #     exp_id = path_readout_str[path_readout_str.rfind('D'):]
    #     print(f"\n-------\nDEBUG: RUN N {i_path} -- EXP_ID {exp_id}")
    #
    #     exp_row = excel_df[excel_df["File Name"] == exp_id]
    #     if exp_row["No. Of Wells"].size == 0:
    #         raise "Experiment not in excel"
    #     if exp_row["No. Of Wells"].size > 1:
    #         raise "More than one line in Excel corresponding to this experiment"
    #     n_wells = int(exp_row["No. Of Wells"])
    #     n_a_type = np.array(exp_row["Version No."])[0]
    #     print(f"DEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")
    #
    #     exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature", print_status=False,
    #                                        end_time_min=60, n_a_type=n_a_type)
    #
    #     t = exp.wells_list[0].time
    #     x = exp.wells_list[0].well_2d_bs_active_mean
    #     ts, xs = resample1d(t, x)
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(t[:15], x[:15])
    #     ax.plot(ts[:20], xs[:20], 'o')
    #     plt.show()



        # TEST RESAMPLE2D ON ARTIFICIAL DATA
        t = np.array([0, 1, 10, 11])
        array = np.array([[9, 6, 4, 2], [2, 3, 4, 7]]).T

        print(array, array.shape)

        ts, arr_sampled = resample2d(t, array)
        print(ts, arr_sampled)

        fig, ax = plt.subplots()
        ax.plot(t, array)
        ax.plot(ts, arr_sampled, 'o')
        # plt.show()

        # TEST RESAMPLE 2D ON ISFET DATA
        onedrive_path = Path("..", "..", "..", "..", "..", "Costanza",
                             "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
        exp_folder = Path(onedrive_path, "Master Data Folder",
                          "Calista Run Data")  # CHANGE THE NAME OF THE DATA FOLDER HERE

        excel_path = Path(onedrive_path, "Master Data Folder",
                          "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
        excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
        exp_paths = [
            Path(exp_folder, "D20250219_E00_C00_F4500KHz_U_LW_108_BRAF_WT_sweep_01")]  # OR THIS TO RUN ONE EXPERIMENT

        print(f"DEBUG: EXP_PATHS")
        for i_path in range(len(exp_paths)):
            print(f"i_path {i_path}, path {exp_paths[i_path]}")

        for i_path, exp_path in enumerate(exp_paths):
            path_readout_str = str(exp_path)
            exp_id = path_readout_str[path_readout_str.rfind('D'):]
            print(f"\n-------\nDEBUG: RUN N {i_path} -- EXP_ID {exp_id}")

            exp_row = excel_df[excel_df["File Name"] == exp_id]
            if exp_row["No. Of Wells"].size == 0:
                raise "Experiment not in excel"
            if exp_row["No. Of Wells"].size > 1:
                raise "More than one line in Excel corresponding to this experiment"
            n_wells = int(exp_row["No. Of Wells"])
            n_a_type = np.array(exp_row["Version No."])[0]
            print(f"DEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")

            exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature", print_status=False,
                                               end_time_min=60, n_a_type=n_a_type)

            t = exp.wells_list[0].time
            x = exp.wells_list[0].well_2d_bs_active
            ts, xs = resample2d(t, x)

            fig, ax = plt.subplots()
            ax.plot(t[:15], x[:15, :])
            ax.plot(ts[:20], xs[:20, :], 'o')
            # plt.show()

