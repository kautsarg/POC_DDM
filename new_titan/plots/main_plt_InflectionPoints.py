# D20241217_E00_C00_F4500KHz_U_LW_86_BRAF_combo_19 has a good positive and negative


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.signal import filtfilt, lfilter


from titan.plt_Experiment_summary import titan_plt_summary, titan_plt_summary_means, titan_plt_summary_temp
from titan.load_and_preprocessing import titan_load_and_preprocessing
from titan.processing_DNA import titan_plt_infl
from titan import functions

def find_infl_points(time, array_1d, tinitial=4*60, tfinal=25*60, D_RANGE=80, filter_order0=60, filter_order1=60, filter_order2=60):
    ''' Find inflection points of a signal
    and classify them in positive (indicating positive output) and negative '''
    # Calculate 1st derivative + smooth it
    array_1d_filt = lfilter(b=np.ones(filter_order0) / filter_order0, a=[1], x=array_1d)
    # 1st derivative + filter it
    array_1d_der1_filt = lfilter(b=np.ones(filter_order1) / filter_order1, a=[1], x=np.diff(array_1d_filt))
    # 2nd derivative + filter it
    array_1d_der2_filt = lfilter(b=np.ones(filter_order2) / filter_order2, a=[1], x=np.diff(array_1d_der1_filt))

    # Find inflection points
    infls = np.where(np.diff(np.sign(array_1d_der2_filt)))[0]
    # Only consider inflection points between tinitial and tfinal (that can indicate a positive sample).
    # NB. also consider the filter delay (idx_delay below)!
    idx_delay = int((filter_order0 + filter_order1 + filter_order2) / 2)

    infls = [x for x in infls if (x-idx_delay) >= 0]  # first, remove negative index
    infls = [x for x in infls if tinitial <= time[x-idx_delay] <= tfinal]  # then, check that time for adjusted infl
    # Classify inflection points
    positive_infl_idx = []
    negative_infl_idx = []
    if (array_1d_filt[-1] - array_1d_filt[0]) <= 0:
        for k, infl in enumerate(infls, 1):  #TODO IS THIS CORRECT??
            if len(array_1d_der2_filt) >= infl + D_RANGE and \
                    all(d < 0 for d in array_1d_der2_filt[infl - D_RANGE:infl - 1]) and \
                    all(d > 0 for d in array_1d_der2_filt[infl + 1:infl + D_RANGE]):  # TODO adapt the additional condition to this case

                positive_infl_idx.append(infl)
            else:
                negative_infl_idx.append(infl)
    else:
        for k, infl in enumerate(infls, 1):  # TODO ALSO ADD A CONDITION FOR INFL AT THE END
            if len(array_1d_der2_filt) >= infl + D_RANGE and \
                    all(d > 0 for d in array_1d_der2_filt[infl - D_RANGE:infl - 1]) and \
                    all(d < 0 for d in array_1d_der2_filt[infl + 1:infl + D_RANGE]):
                    #and np.abs(array_1d_filt[infl]-array_1d_filt[0]) >= 10:  # TODO adapt this parameter
                positive_infl_idx.append(infl)
            else:
                negative_infl_idx.append(infl)
    return positive_infl_idx, negative_infl_idx


def titan_exp_pr_infl(exp, plt_show=False, plt_save=False):  # todo put tinitial, tfinal as inputs & also filter orders
    if plt_show:
        fig, ax = plt.subplots(2, exp.nwells, figsize=(exp.nwells*4, 5))

    filter_order0, filter_order1, filter_order2 = 60, 60, 60  # todo make this input

    for i_well, this_well in enumerate(exp.wells_list):
        positive_infl_idx, negative_infl_idx = find_infl_points(this_well.time, this_well.well_2d_bs_active_mean,
                                                                filter_order0=filter_order0,
                                                                filter_order1=filter_order1,
                                                                filter_order2=filter_order2)

        print(f"debug: well {i_well} has inflection points {positive_infl_idx} (pos) and {negative_infl_idx} (neg)")

        if len(positive_infl_idx) >= 1:
            this_well.label = 1
        else:
            this_well.label = 0

        if plt_show:
            arr1d_filt = lfilter(b=np.ones(filter_order0) / filter_order0, a=[1], x=this_well.well_2d_bs_active_mean)
            der1 = np.diff(arr1d_filt)
            der1_filt = lfilter(b=np.ones(filter_order1) / filter_order1, a=[1], x=der1)
            der2 = np.diff(der1_filt)
            der2_filt = lfilter(b=np.ones(filter_order2) / filter_order2, a=[1], x=der2)
            idx_delay = int((filter_order0 + filter_order1 + filter_order2) / 2)

            ax[0, i_well].plot(this_well.time_min, this_well.well_2d_bs_active_mean, linewidth=0.5, color="k")
            ax[0, i_well].plot(this_well.time_min, arr1d_filt, linewidth=2, color="k")
            ax[0, i_well].set(title="BS average output", xlabel="time (s)", ylabel="Vout")
            for k, infl in enumerate(positive_infl_idx, 1):
                ax[0, i_well].axvline(x=this_well.time_min[infl], color='r', linewidth=2)
                ax[0, i_well].axvline(x=this_well.time_min[infl-idx_delay], color='r', linewidth=2, linestyle="--")
            for k, infl in enumerate(negative_infl_idx, 1):
                ax[0, i_well].axvline(x=this_well.time_min[infl], color='k', linewidth=2)
                ax[0, i_well].axvline(x=this_well.time_min[infl-idx_delay], color='k', linewidth=2, linestyle="--")

            ax[1, i_well].plot(this_well.time_min[1:], der1, linewidth=0.5, color="g")
            ax[1, i_well].plot(this_well.time_min[1:], der1_filt, linewidth=2, color="g")
            ax[1, i_well].axhline(0, color="g")
            ax_tmp = ax[1, i_well].twinx()
            ax_tmp.plot(this_well.time_min[2:], der2, linewidth=0.5, color="b")
            ax_tmp.plot(this_well.time_min[2:], der2_filt, linewidth=2, color="b")
            ax_tmp.axhline(0, color="b")
            ax[1, i_well].set(xlabel="time (s)", ylabel="1st derivative", title="1st and 2nd derivative")

    if plt_show:
        plt.tight_layout()
        plt.show()

    return


if __name__ == '__main__':

    onedrive_path = Path("..", "..", "..", "..", "..", "Costanza", "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    exp_folder = Path(onedrive_path, "Master Data Folder", "Calista Run Data")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    excel_path = Path(onedrive_path, "Master Data Folder", "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
    excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
    exp_paths = [Path(exp_folder, "D20241217_E00_C00_F4500KHz_U_LW_86_BRAF_combo_19")]  # OR THIS TO RUN ONE EXPERIMENT

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

        exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
                                           end_time_min=60, n_a_type=n_a_type,
                                           print_status=True, plt_gain_calib=False, save_gain_calib=False)

        times = exp.wells_list[0].time_min
        max_time = times[-1]
        idx30 = functions.time_to_index([30], times)[0]
        print(f"------------------------------->>>> NTIMES {times.shape} --- MAX TIME {max_time} --- IDX30MIN {idx30}")

        #titan_plt_summary(exp, exp_path, plt_save=False)
        #titan_plt_summary_means(exp, exp_path, plt_save=False)
        #titan_plt_infl(exp, exp_path, plt_show=True, plt_save=True)

        fig, ax = plt.subplots(3, 2, figsize=(10, 4.55), dpi=200, sharey='row')
        time = exp.wells_list[0].time_min

        well_negative = exp.wells_list[2]
        filteredN = filtfilt(b=np.ones(50) / 50, a=[1], x=well_negative.well_2d_bs_active_mean)
        diffN = np.diff(filteredN)
        diffNfiltered = filtfilt(b=np.ones(50) / 50, a=[1], x=diffN)
        diff2N = np.diff(diffNfiltered)
        diff2Nfiltered = filtfilt(b=np.ones(50) / 50, a=[1], x=diff2N)

        infls = np.where(np.diff(np.sign(diff2Nfiltered)))[0]
        for infl in infls:
            print(infl)

        ax[0, 0].plot(time, well_negative.well_2d_bs_active_mean, linewidth=1, color="g")
        ax[0, 0].plot(time, filteredN, linewidth=2, color="g")
        ax[0, 0].set(title="NEGATIVE", ylabel="y")
        ax[0, 0].grid()

        ax[0, 0].axvline(x=time[106], color='k', linewidth=2, linestyle="--")
        ax[0, 0].axvline(x=time[135], color='k', linewidth=2, linestyle="--")
        ax[0, 0].axvline(x=time[384], color='k', linewidth=2, linestyle="--")

        ax[1, 0].plot(time[1:], diffN, linewidth=1, color="g")
        ax[1, 0].plot(time[1:], diffNfiltered, linewidth=2, color="g")
        ax[1, 0].set(title="1st derivative", ylabel=r"$\frac{dy}{dt}$")
        ax[1, 0].grid()

        ax[2, 0].plot(time[2:], diff2N, linewidth=1, color="g")
        ax[2, 0].plot(time[2:], diff2Nfiltered, linewidth=2, color="g")
        ax[2, 0].set(title="2nd derivative", xlabel="time (min)", ylabel=r"$\frac{d^2y}{dt^2}$")
        ax[2, 0].grid()

        well_positive = exp.wells_list[1]
        filteredP = filtfilt(b=np.ones(50) / 50, a=[1], x=well_positive.well_2d_bs_active_mean)
        diffP = np.diff(filteredP)
        diffPfiltered = filtfilt(b=np.ones(50) / 50, a=[1], x=diffP)
        diff2P = np.diff(diffPfiltered)
        diff2Pfiltered = filtfilt(b=np.ones(50) / 50, a=[1], x=diff2P)

        infls = np.where(np.diff(np.sign(diff2Pfiltered)))[0]
        for infl in infls:
            print(infl)

        ax[0, 1].plot(time, well_positive.well_2d_bs_active_mean, linewidth=1, color="r")
        ax[0, 1].plot(time, filteredP, linewidth=2, color="r")
        ax[0, 1].set(title="POSITIVE", ylabel="y")
        ax[0, 1].grid()
        ax[0, 1].axvline(x=time[236], color='r', linewidth=2, linestyle="--")
        ax[0, 1].axvline(x=time[61], color='k', linewidth=2, linestyle="--")

        ax[1, 1].plot(time[1:], diffP, linewidth=1, color="r")
        ax[1, 1].plot(time[1:], diffPfiltered, linewidth=2, color="r")
        ax[1, 1].set(title="1st derivative", ylabel=r"$\frac{dy}{dt}$")
        ax[1, 1].grid()

        ax[2, 1].plot(time[2:], diff2P, linewidth=1, color="r")
        ax[2, 1].plot(time[2:], diff2Pfiltered, linewidth=2, color="r")
        ax[2, 1].set(title="2nd derivative", xlabel="time (min)", ylabel=r"$\frac{d^2y}{dt^2}$")
        ax[2, 1].grid()

        plt.tight_layout()
        plt.savefig("infl_point_plt.png", transparent=True)
        plt.show()


        # titan_exp_pr_infl(exp, plt_show=True)


