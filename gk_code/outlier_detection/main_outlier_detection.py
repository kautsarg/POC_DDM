import os
import pickle
import pandas as pd
from msc_outlier import run_msc_pipeline
from amf_outlier import run_amf_pipeline
from mean_std_outlier import run_meanstd_pipeline

exp_path = "/Users/kautsarg/Documents/Final Project/Run Data/trial test data/D20250808_E00_C00_F4500KHz_U_Sample_7"
save_path = os.path.join(exp_path, "curve_for_training_added_features.pkl")
updated_save_path = os.path.join(exp_path, "curve_for_training_latest.pkl")

# 1. Load your base data
with open(save_path, 'rb') as f:
    training_data = pickle.load(f)

dataset_name = training_data["dataset_name"]
dataset = training_data["dataset"]
kinetic_features = training_data["kinetic_features"]
Y_well = training_data["Y_well"]
ref_curves = dataset[0]

# --- USER DEFINED CONFIGURATIONS ---
linear_feature_combinations = training_data.get("linear_feature_combinations", [])
important_feature_combinations = training_data.get("important_feature_combinations", [])

# Ensure multi-dataset arrays are correctly sized (len = num datasets)
msc_configs = [
    ("msc_linear_0.05", 0.05, linear_feature_combinations),
    ("msc_linear_0.01", 0.01, linear_feature_combinations),
    ("msc_linear_0.001", 0.001, linear_feature_combinations),
    ("msc_baseline_0.05", 0.05, [["Ct", "Cy0", "log_F0"]] * len(dataset_name)),
    ("msc_baseline_0.01", 0.01, [["Ct", "Cy0", "log_F0"]] * len(dataset_name)),
    ("msc_baseline_0.001", 0.001, [["Ct", "Cy0", "log_F0"]] * len(dataset_name))
]

amf_configs = [
    ("amf_important", important_feature_combinations),
    ("amf_send_5", [["Fm", "Fb", "Sc", "Cs", "send_5"]] * len(dataset_name)),
    ("amf_send_10", [["Fm", "Fb", "Sc", "Cs", "send_10"]] * len(dataset_name)),
    ("amf_send_15", [["Fm", "Fb", "Sc", "Cs", "send_15"]] * len(dataset_name)),
    ("amf_send_20", [["Fm", "Fb", "Sc", "Cs", "send_20"]] * len(dataset_name)),
    ("amf_send_25", [["Fm", "Fb", "Sc", "Cs", "send_25"]] * len(dataset_name)),
    ("amf_send_abs_5", [["Fm", "Fb", "Sc", "Cs", "send_abs_5"]] * len(dataset_name)),
    ("amf_send_abs_10", [["Fm", "Fb", "Sc", "Cs", "send_abs_10"]] * len(dataset_name)),
    ("amf_send_abs_15", [["Fm", "Fb", "Sc", "Cs", "send_abs_15"]] * len(dataset_name)),
    ("amf_send_abs_20", [["Fm", "Fb", "Sc", "Cs", "send_abs_20"]] * len(dataset_name)),
    ("amf_send_abs_25", [["Fm", "Fb", "Sc", "Cs", "send_abs_25"]] * len(dataset_name))
]

mean_std_configs = [
    ("env_1std", 1),
    ("env_2std", 2),
    ("env_3std", 3)
]

# 2. Execute Pipelines & Collect DataFrames
all_new_feature_dfs = [ [] for _ in range(len(dataset_name)) ] # Array of lists

print("=== RUNNING MSC PIPELINES ===")
for exp_label, p_val, feats in msc_configs:
    extracted_dfs = run_msc_pipeline(exp_label, p_val, ref_curves, f"{exp_path}/msc_outlier", dataset_name, kinetic_features, dataset, Y_well, feats)
    for i in range(len(dataset_name)): all_new_feature_dfs[i].append(extracted_dfs[i])

print("=== RUNNING AMF PIPELINES ===")
for exp_label, feats in amf_configs:
    extracted_dfs = run_amf_pipeline(exp_label, feats, ref_curves, f"{exp_path}/amf_outlier", dataset_name, kinetic_features, dataset, Y_well)
    for i in range(len(dataset_name)): all_new_feature_dfs[i].append(extracted_dfs[i])

print("=== RUNNING MEAN/STD PIPELINES ===")
for exp_label, num_std in mean_std_configs:
    extracted_dfs = run_meanstd_pipeline(exp_label, num_std, ref_curves, f"{exp_path}/meanstd_outlier", dataset_name, dataset, Y_well, kinetic_features[0].index)
    for i in range(len(dataset_name)): all_new_feature_dfs[i].append(extracted_dfs[i])

# 3. Concatenate all new features into the kinetic_features
print("=== MERGING FEATURES ===")
for i in range(len(dataset_name)):
    kinetic_features[i] = pd.concat([kinetic_features[i]] + all_new_feature_dfs[i], axis=1)

# 4. Save Final Dataset
training_data["kinetic_features"] = kinetic_features
with open(updated_save_path, 'wb') as f:
    pickle.dump(training_data, f)
print(f"Data saved to {updated_save_path}")