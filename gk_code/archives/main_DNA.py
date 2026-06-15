import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.signal import lfilter


from titan.plt_Experiment_summary import titan_plt_summary, titan_plt_summary_means, titan_plt_summary_temp
from titan.load_and_preprocessing import titan_load_and_preprocessing
from titan.processing_DNA import titan_plt_infl

if __name__ == '__main__':

    # ##########  A. IF THE FILES ARE IN THE EXCEL ##########
    # onedrive_path = Path("..", "..", "..", "..", "..", "Costanza", "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    # exp_folder = Path(onedrive_path, "Master Data Folder", "Matthew Run Data")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    #
    # excel_path = Path(onedrive_path, "Master Data Folder", "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
    # excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
    #
    # # exp_paths = [f for f in exp_folder.glob('*') if f.is_dir()]  # USE THIS TO RUN ALL EXPERIMENTS IN A FOLDER
    # exp_paths = [Path(exp_folder, "D20250409_E00_C00_F4500KHz_U_demo_pipette")]  # OR THIS TO RUN ONE EXPERIMENT
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
    #     exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
    #                                        end_time_min=60, n_a_type=n_a_type,
    #                                        print_status=True, plt_gain_calib=True, save_gain_calib=True)
    #
    #     times = exp.wells_list[0].time_min
    #     max_time = times[-1]
    #     idx30 = functions.time_to_index([30], times)[0]
    #     print(f"------------------------------->>>> NTIMES {times.shape} --- MAX TIME {max_time} --- IDX30MIN {idx30}")
    #
    #     titan_plt_summary(exp, exp_path, plt_save=True)
    #     titan_plt_summary_means(exp, exp_path, plt_save=True)
    #     titan_plt_infl(exp, exp_path, plt_show=True, plt_save=True)


    ##########  B. IF THE FILES ARE NOT IN THE EXCEL  ##########
    n_wells = 10
    n_a_type = "v04"  # SPECIFY NUMBER OF WELLS AND VERSION HERE
    # onedrive_path = Path("..", "..", "..", "..", "..", "Costanza",
    #                      "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    # exp_folder = Path(onedrive_path, "Master Data Folder", "negative test")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    exp_folder = "titan"
    # exp_folder = Path(onedrive_path, "Master Data Folder", "Lacewing - 25_Zambia_Malaria")

    # exp_paths = [f for f in exp_folder.glob('*') if f.is_dir()]  # USE THIS TO RUN ALL EXPERIMENTS IN A FOLDER
    exp_paths = [Path(exp_folder, "D20250311_E00_C00_F4500KHz_U_Zam_01")]  # OR THIS TO RUN ONE EXPERIMENT

    print(f"DEBUG: EXP_PATHS")
    for i_path in range(len(exp_paths)):
        print(f"i_path {i_path}, path {exp_paths[i_path]}")

    for i_path, exp_path in enumerate(exp_paths):
        path_readout_str = str(exp_path)
        print(f"\n-------\nDEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")

        exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
                                           end_time_min=60, n_a_type=n_a_type,
                                           print_status=True, plt_gain_calib=True, save_gain_calib=True)

        titan_plt_summary(exp, exp_path, plt_save=True)
        titan_plt_summary_temp(exp, exp_path, plt_save=True)
        #titan_plt_summary_means(exp, exp_path, plt_save=True)
        titan_plt_infl(exp, exp_path, plt_show=False, plt_save=True)


    # ##########  C. TB TRIAL DATA ##########
    # onedrive_path = Path("..", "..", "..", "..", "..", "Costanza", "OneDrive - Imperial College London")  # CHANGE THE NAME OF THE PATH HERE
    # exp_folder = Path(onedrive_path, "Master Data Folder", "TB Trial")  # CHANGE THE NAME OF THE DATA FOLDER HERE
    #
    # excel_path = Path(onedrive_path, "Master Data Folder", "Run Tracker.xlsx")  # CHANGE PATH OF THE EXCEL SUMMARY HERE
    # excel_df = pd.read_excel(excel_path, sheet_name="Master Run")  # CHANGE THE NAME OF THE EXCEL SHEET NAME HERE
    #
    # exp_paths = [f for f in exp_folder.glob('*') if f.is_dir()]  # USE THIS TO RUN ALL EXPERIMENTS IN A FOLDER
    # # exp_paths = [Path(exp_folder, "D20241217_E00_C00_F4500KHz_U_LW_86_BRAF_combo_19")]  # OR THIS TO RUN ONE EXPERIMENT
    #
    # print(f"DEBUG: EXP_PATHS")
    # for i_path in range(len(exp_paths)):
    #     print(f"i_path {i_path}, path {exp_paths[i_path]}")
    #
    # for i_path, exp_path in enumerate(exp_paths):
    #     sub_exp_paths = [f for f in exp_path.glob('*') if f.is_dir()]
    #     for i_sub_exp, sub_exp_path in enumerate(sub_exp_paths):
    #         # TODO if if sub_exp_path == OLD SKIP
    #         filename_readout_list = [i for i in sub_exp_path.glob("*eadout*.bin")]
    #
    #         if len(filename_readout_list) >= 1:
    #             for path_readout in filename_readout_list:
    #                 path_readout_str = str(path_readout)
    #
    #                 # LOAD EXPERIMENT INFO FROM THE EXCEL
    #                 sample_id = path_readout_str[path_readout_str.rfind('S'):path_readout_str.rfind('S')+3]
    #                 cartridge_id = int(path_readout_str[path_readout_str.rfind('U_TB_')+5:path_readout_str.rfind('U_TB_')+6])
    #                 if cartridge_id == 1:
    #                     exp_id = str(sample_id+"_1")
    #                 elif cartridge_id == 2:
    #                     exp_id = str(sample_id+"_2")
    #                 else:
    #                     raise "unexpected"
    #
    #                 exp_row = excel_df[excel_df["File Name"] == exp_id]
    #                 if exp_row["No. Of Wells"].size == 0:
    #                     raise "Experiment not in excel"
    #                 if exp_row["No. Of Wells"].size > 1:
    #                     raise "More than one line in Excel corresponding to this experiment"
    #                 n_wells = int(exp_row["No. Of Wells"])
    #                 n_a_type = np.array(exp_row["Version No."])[0]
    #                 print(f"DEBUG: RUN N {i_path} -- NWELLS {n_wells} -- N_A_TYPE {n_a_type}")
    #
    # exp = titan_load_and_preprocessing(exp_path, n_wells=n_wells, start_type="temperature",
    #                                    end_time_min=60, n_a_type=n_a_type,
    #                                    print_status=True, plt_gain_calib=True, save_gain_calib=True)
    #
    #                 titan_plt_summary(exp, sub_exp_path, plt_save=True)
    #                 titan_plt_infl(exp, exp_path, plt_show=True, plt_save=True)