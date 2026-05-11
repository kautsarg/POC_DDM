import os
# Silencing TF warnings at the main level as well
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pickle
import gc
from sklearn.preprocessing import LabelEncoder
from model_utils import evaluate_outlier_filters, plot_ml_results, set_global_determinism

set_global_determinism(0)

exp_path = "/Users/kautsarg/Documents/Final Project/Run Data/trial test data/D20250808_E00_C00_F4500KHz_U_Sample_7"
data_path = os.path.join(exp_path, "curve_for_training_latest.pkl")
results_file_path = os.path.join(exp_path, "classification_performances_with_proba.pkl")

# Ensure the plot directory exists
model_plot_path = os.path.join(exp_path, "model_performance")
os.makedirs(model_plot_path, exist_ok=True)

with open(data_path, 'rb') as f:
    training_data = pickle.load(f)

dataset_name = training_data["dataset_name"]
dataset = training_data["dataset"]
kinetic_features = training_data["kinetic_features"]
Y_well = training_data["Y_well"]

encoder = LabelEncoder()
y_full = encoder.fit_transform(Y_well)

# 1. Dynamically gather all filter columns
all_columns = kinetic_features[0].columns.tolist()
outlier_filters = [None] + [c for c in all_columns if 'label' in c and ('msc_' in c or 'amf_' in c or 'mean_std_' in c)]
outlier_filters = [None, 'msc_label_msc_linear_0.01', 'msc_label_msc_linear_0.05',
					'msc_label_msc_linear_0.001', 'msc_label_msc_baseline_0.01', 
				    'msc_label_msc_baseline_0.05', 'msc_label_msc_baseline_0.001',
                    'amf_label_amf_important', 'amf_label_amf_send_5',
                    'amf_label_amf_send_10', 'amf_label_amf_send_15',
                    'amf_label_amf_send_20', 'amf_label_amf_send_25',
                    'amf_label_amf_send_abs_5', 'amf_label_amf_send_abs_10',
                    'amf_label_amf_send_abs_15', 'amf_label_amf_send_abs_20',
                    'amf_label_amf_send_abs_25', 'mean_std_label_env_1std',
                    'mean_std_label_env_2std', 'mean_std_label_env_3std']

print(f"[*] Found {len(outlier_filters)-1} Dynamic Outlier Filters to test.")

if os.path.exists(results_file_path):
    with open(results_file_path, 'rb') as f:
        all_ml_results = pickle.load(f)
else:
    all_ml_results = {}

total_datasets = len(dataset_name)
total_samples = len(y_full)

# for idx, (name, features_df, curves_2d) in enumerate(zip(dataset_name, kinetic_features, dataset)):
for idx, (name, features_df, curves_2d) in enumerate(reversed(list(zip(dataset_name, kinetic_features, dataset)))):
    clean_title = name.replace("_", " ").title()
    progress_pct = ((idx + 1) / total_datasets) * 100
    
    print(f"\n{'='*75}")
    print(f"[{idx+1}/{total_datasets} | {progress_pct:.1f}%] Processing Dataset: {clean_title}")
    print(f"{'='*75}")

    if clean_title not in all_ml_results: 
        all_ml_results[clean_title] = {}
    
    # --- NATIVE TRAINING ---
    print(f"  [MODE 1/2] NATIVE TRAINING")
    cached_native = all_ml_results[clean_title].get("Native", {})
    
    res_native = evaluate_outlier_filters(
        curves_2d, features_df, y_full, outlier_filters, clean_title, "Native", cached_results=cached_native
    )
    all_ml_results[clean_title]["Native"] = res_native
    with open(results_file_path, 'wb') as f: pickle.dump(all_ml_results, f)
        
    prefix_native = os.path.join(model_plot_path, f"{name}_Native")
    plot_ml_results(
        results_dict=all_ml_results[clean_title]["Native"], 
        outlier_filters=outlier_filters, 
        dataset_name=clean_title, 
        mode_name="Native Training", 
        total_count=total_samples, 
        save_prefix=prefix_native
    )
        
    # --- REFERENCE TRAINING ---
    print(f"\n  [MODE 2/2] REFERENCE TRAINING")
    cached_ref = all_ml_results[clean_title].get("Reference", {})
    
    res_ref = evaluate_outlier_filters(
        dataset[0], features_df, y_full, outlier_filters, clean_title, "Reference", cached_results=cached_ref
    )
    all_ml_results[clean_title]["Reference"] = res_ref
    with open(results_file_path, 'wb') as f: pickle.dump(all_ml_results, f)
        
    prefix_ref = os.path.join(model_plot_path, f"{name}_Reference")
    plot_ml_results(
        results_dict=all_ml_results[clean_title]["Reference"], 
        outlier_filters=outlier_filters, 
        dataset_name=clean_title, 
        mode_name="Reference Training", 
        total_count=total_samples, 
        save_prefix=prefix_ref
    )
        
    gc.collect()

print("\n[*] All Training Complete!")