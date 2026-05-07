from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

from titan import Experiment


def titan_plt_summary(exp: Experiment, exp_path: Path, plt_save: bool = False, plt_show: bool = False):

    # PLOT
    n_wells = len(exp.wells_list)
    fig, ax = plt.subplots(8, n_wells, figsize=(4 * n_wells, 20), dpi=200, sharey='row')
    fig.suptitle(f'{str(exp_path)}')

    for i, this_well in enumerate(exp.wells_list):
        if n_wells == 1:
            ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = ax[0], ax[1], ax[2], ax[3], ax[4], ax[5], ax[6], ax[7]
        else:
            ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7 = ax[0, i], ax[1, i], ax[2, i], ax[3, i], ax[4, i], ax[5, i], ax[6, i], ax[7, i]
        pos = ax0.imshow(np.mean(this_well.well_3d_nl, axis=2), cmap='cividis')
        fig.colorbar(pos, ax=ax0)
        ax0.set(title=f'w{i}: Average chem data')

        ax1.plot(this_well.time_npr, this_well.well_2d_npr)
        ax1.plot(this_well.time_npr, np.mean(this_well.well_2d_npr, axis=1), c='k', linewidth=3)
        ax1.axvline(this_well.time_npr[this_well.idx_start], linewidth=2, color='k', linestyle='--')
        ax1.axvline(this_well.time_npr[this_well.idx_settled], linewidth=2, color='b', linestyle='--')
        ax1.axvline(this_well.time_npr[this_well.idx_end], linewidth=2, color='k', linestyle='--')
        ax1.set(title=f'w{i}: Raw exp chem data', xlabel='Time(s)', ylabel='time (pulses)')

        ax2.plot(this_well.time_npr, this_well.well_temp_mean_NEW)
        ax2.axvline(this_well.time_npr[this_well.idx_start], linewidth=2, color='k', linestyle='--')
        ax2.axvline(this_well.time_npr[this_well.idx_settled], linewidth=2, color='b', linestyle='--')
        ax2.axvline(this_well.time_npr[this_well.idx_end], linewidth=2, color='k', linestyle='--')
        ax2.set(title="temperature")

        ax3.imshow(this_well.idx_active.reshape(-1, this_well.well_ncols), cmap='cividis')
        ax3.set(title=f'w{i}: Active pixels')

        ax4.plot(this_well.time, this_well.well_2d_nl_bs_active)
        ax4.plot(this_well.time, this_well.well_2d_nl_bs_active_mean, c='k', linewidth=3)
        ax4.set(title=f'w{i}: NL BS Active', xlabel='Time(s)', ylabel='NL output')

        ax5.plot(this_well.time, this_well.well_2d_nl_bs_active_mean, c='k', linewidth=3)
        delta = (this_well.well_2d_nl_bs_active_mean[-1] - this_well.well_2d_nl_bs_active_mean[0])
        ax5.set(title=f'w{i}: NL BS Active Mean (delta = {delta:.2f})', xlabel='Time(s)', ylabel='NL output')

        ax6.plot(this_well.time, this_well.well_2d_bs_active)
        ax6.plot(this_well.time, this_well.well_2d_bs_active_mean, c='k', linewidth=3)
        ax6.set(title=f'w{i}: BS Active', xlabel='Time(s)', ylabel='linearised output')

        ax7.plot(this_well.time, this_well.well_2d_bs_active_mean, c='k', linewidth=3)
        delta = (this_well.well_2d_bs_active_mean[-1] - this_well.well_2d_bs_active_mean[0])*100
        ax7.set(title=f'w{i}: BS Active Mean (100*delta = {delta:.2f})', xlabel='Time(s)', ylabel='linearised output')

    plt.tight_layout()
    # SAVE
    if plt_save:
        plt.savefig(Path(exp_path, "exp_summary.png"))
        print(f'Image exp_summary.png  for experiment {exp_path} saved.')
    if plt_show:
        plt.show()
    return


def titan_plt_summary_means(exp: Experiment, exp_path: Path, plt_save: bool = False, plt_show: bool = False):

    # PLOT
    n_wells = len(exp.wells_list)
    fig, ax = plt.subplots(1, 1, figsize=(7,5), dpi=200, sharey='row')

    for i, this_well in enumerate(exp.wells_list):
        ax.plot(this_well.time_min, this_well.well_2d_bs_active_mean_filt(), linewidth=2, label=f"well {i+1}")

    ax.legend()
    ax.grid()
    ax.set(title="Zambia trial - Malaria - Pregnant (Pan/Pf)", xlabel="Time (min)", ylabel="Amplitude")

    plt.tight_layout()
    # SAVE
    if plt_save:
        plt.savefig(Path(exp_path, "exp_summary_easy.png"))
        print(f'Image exp_summary_easy.png  for experiment {exp_path} saved.')
    if plt_show:
        plt.show()
    return


def titan_plt_summary_temp(exp: Experiment, exp_path: Path, plt_save: bool = False, plt_show: bool = False):

    # PLOT
    n_wells = len(exp.wells_list)
    fig, ax = plt.subplots(6, n_wells, figsize=(5*n_wells, 13), dpi=200, sharey='row')

    for i, this_well in enumerate(exp.wells_list):

        ax[0, i].plot(this_well.time_npr, this_well.well_2d_temp_npr)
        ax[0, i].set(title="NON-LINEARISED TEMPERATURE")

        ax[1, i].plot(this_well.time_npr, this_well.well_temp_mean_then_lin)
        ax[1, i].set(title="GUI well_temp_mean_then_lin")

        ax[2, i].plot(this_well.time_npr, this_well.well_temp_lin2d)
        ax[2, i].set(title="GUI PARAMS well_temp_lin2d")

        ax[3, i].plot(this_well.time_npr, this_well.well_temp_mean_NEW)
        ax[3, i].set(title="NEW well_temp_mean")
        idx_active_temp = (np.mean(this_well.well_2d_temp_npr, axis=0) > 10)
        print(f"DEBUGGGG temperature active pixels {idx_active_temp.sum()} out of {idx_active_temp.shape}")

        ax[4, i].plot(this_well.time_npr, this_well.well_temp_2D_NEW[:, idx_active_temp])
        ax[4, i].set(title="NEW well_temp_mean2D")

        ax[5, i].imshow(idx_active_temp.reshape(this_well.well_temp_nrows, this_well.well_temp_ncols), cmap='cividis')
        ax[5, i].set(title="active temperature pixels")

    plt.tight_layout()
    # SAVE
    if plt_save:
        plt.savefig(Path(exp_path, "exp_summary_easy.png"))
        print(f'Image exp_summary_easy.png  for experiment {exp_path} saved.')
    if plt_show:
        plt.show()
    return

