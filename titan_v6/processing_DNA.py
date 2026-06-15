import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.signal import lfilter


def find_infl_points(time, array_1d, tinitial=4*60, tfinal=35*60, d_range=60, filter_order0=60, filter_order1=60, filter_order2=60):
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
        for k, infl in enumerate(infls, 1):  # TODO check this is correct
            if len(array_1d_der2_filt) >= infl + d_range and \
                    all(d < 0 for d in array_1d_der2_filt[infl - d_range:infl - 1]) and \
                    all(d > 0 for d in array_1d_der2_filt[infl + 1:infl + d_range]):

                positive_infl_idx.append(infl)
            else:
                negative_infl_idx.append(infl)
    else:
        for k, infl in enumerate(infls, 1):
            if len(array_1d_der2_filt) >= infl + d_range and \
                    all(d > 0 for d in array_1d_der2_filt[infl - d_range:infl - 1]) and \
                    all(d < 0 for d in array_1d_der2_filt[infl + 1:infl + d_range]):
                positive_infl_idx.append(infl)
            else:
                negative_infl_idx.append(infl)
    return positive_infl_idx, negative_infl_idx

def titan_plt_infl(exp, exp_path, tinitial=4*60, tfinal=25*60, d_range=80, filter_order0=60, filter_order1=60, filter_order2=60, plt_show=False, plt_save=False):  # todo put tinitial, tfinal as inputs & also filter orders
    if plt_show:
        fig, ax = plt.subplots(2, exp.nwells, figsize=(exp.nwells*4, 5))
        fig.suptitle(f'{str(exp_path)}')

    for i_well, this_well in enumerate(exp.wells_list):
        positive_infl_idx, negative_infl_idx = find_infl_points(this_well.time, this_well.well_2d_bs_active_mean,
                                                                tinitial=tinitial, tfinal=tfinal, d_range=d_range,
                                                                filter_order0=filter_order0,
                                                                filter_order1=filter_order1,
                                                                filter_order2=filter_order2)

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
            ax[0, i_well].set(title=f"w{i_well}: BS average output", xlabel="time (s)", ylabel="Vout")
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
        if plt_save:
            plt.savefig(Path(exp_path, "infl_summary.png"))
            print(f'Image infl_summary.png for experiment {exp_path} saved.')
        plt.show()

    return