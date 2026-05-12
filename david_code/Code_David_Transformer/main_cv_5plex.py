"""
10-Fold Stratified Cross-Validation training script for 5-plex dLAMP dataset.

Classes: ad (0), c19 (1), ia (2), ib (3), kp (4)
Sequence length: 35
Dataset: 5Plex_dLAMP_oldData_processed.csv (~54186 samples)
"""

import os
import sys
import argparse
import warnings
import logging
import datetime
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve import paths
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
nested_project_dir = os.path.join(project_root, os.path.basename(project_root))
if os.path.isdir(nested_project_dir):
    sys.path.insert(0, nested_project_dir)

from src.models.transformer import Transformer
from src.data.data_loader import generate_cv_data_loader
from src.training.train_baseline import train_baseline
from src.training.evaluation import eval_best_model, save_aggregated_confusion_matrix


DEFAULT_DATA_DIR      = './data'
DEFAULT_OUTPUT_DIR    = './output_raw'
DEFAULT_DATA_FILENAME = '5Plex_dLAMP_oldData_processed.csv'
ACTIVITIES  = ['ad', 'c19', 'ia', 'ib', 'kp']
CLASS_NUM   = 5
DEFAULT_SEQ_LENGTH = 35


def resolve_device(device_arg):
    cuda_available = torch.cuda.is_available()
    mps_built      = torch.backends.mps.is_built()
    mps_available  = mps_built and torch.backends.mps.is_available()

    if device_arg == 'auto':
        if cuda_available:
            device_name = 'cuda'
        elif mps_available:
            device_name = 'mps'
        else:
            device_name = 'cpu'
    else:
        device_name = device_arg

    if device_name == 'cuda' and not cuda_available:
        raise RuntimeError("CUDA requested but not available.")
    if device_name == 'mps' and not mps_available:
        raise RuntimeError("MPS requested but not available. Run from .venv312.")

    return torch.device(device_name), {
        "cuda_available": cuda_available,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def setup_logging(dir_out, run_timestamp):
    log_file = os.path.join(dir_out, f'cv_log_{run_timestamp}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info(f"Cross-validation log: {log_file}")


def build_config(args, dir_out):
    return {
        "N_seq": args.seq_length,
        "num_iterations": args.num_iterations,
        "test_interval": args.test_interval,
        "dir_out": dir_out,
        "optimizer": {
            "type": optim.AdamW,
            "optim_params": {"lr": args.lr, "weight_decay": 1e-4},
            "lr_type": "cosine_warmup",
            "lr_param": {
                "lr": args.lr,
                "T_max": args.num_iterations,
                "warmup_iters": 500,
                "lr_min": 1e-6,
            },
        },
        "class_num": CLASS_NUM,
    }


def main():
    parser = argparse.ArgumentParser(
        description='10-Fold Stratified CV for 5-plex dLAMP Transformer'
    )
    parser.add_argument('--n_splits',        type=int,   default=10)
    parser.add_argument('--max_folds',       type=int,   default=None,
                        help='Stop after this many folds (default: run all n_splits folds)')
    parser.add_argument('--num_iterations',  type=int,   default=10000)
    parser.add_argument('--test_interval',   type=int,   default=50)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--batch_size',      type=int,   default=64)
    parser.add_argument('--test_batch_size', type=int,   default=128)
    parser.add_argument('--dir_data',        type=str,   default=DEFAULT_DATA_DIR)
    parser.add_argument('--data_filename',   type=str,   default=DEFAULT_DATA_FILENAME)
    parser.add_argument('--seq_length',      type=int,   default=DEFAULT_SEQ_LENGTH)
    parser.add_argument('--dir_out',         type=str,   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--device',          type=str,   default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    f_config = {
        'd_input': 1,
        'd_model': 128,
        'q': 8,
        'v': 8,
        'h': 4,
        'N': 4,
        'attention_size': None,
        'dropout': 0.3,
        'chunk_mode': None,
        'pe': 'original',
        'pe_period': 20,
        'use_bottleneck': True,
        'bottleneck_dim': 256,
        'seq_length': args.seq_length,
    }

    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_dir = os.path.join(args.dir_out, f'5plex_cv_{args.n_splits}fold_{run_timestamp}')
    os.makedirs(cv_dir, exist_ok=True)

    setup_logging(cv_dir, run_timestamp)

    device, device_status = resolve_device(args.device)
    logging.info(f"Dataset : {args.data_filename}  |  classes={CLASS_NUM}  seq_len={args.seq_length}")
    logging.info(f"Device  : {device}  |  cuda={device_status['cuda_available']}  "
                 f"mps_built={device_status['mps_built']}  "
                 f"mps_available={device_status['mps_available']}")
    logging.info(f"Starting {args.n_splits}-fold CV  |  "
                 f"{args.num_iterations} iter/fold  |  lr={args.lr}  bs={args.batch_size}")

    folds_to_run = args.n_splits if args.max_folds is None else min(args.max_folds, args.n_splits)
    fold_accuracies = []
    all_y_true = np.array([], dtype=int)
    all_y_pred = np.array([], dtype=int)

    for fold_k in range(folds_to_run):
        logging.info(f"\n{'='*60}")
        logging.info(f"FOLD {fold_k+1} / {args.n_splits}")
        logging.info(f"{'='*60}")

        fold_dir = os.path.join(cv_dir, f'fold_{fold_k+1:02d}')
        os.makedirs(fold_dir, exist_ok=True)
        fold_timestamp = f"{run_timestamp}_fold{fold_k+1:02d}"

        data_loaders = generate_cv_data_loader(
            fold_k=fold_k,
            n_splits=args.n_splits,
            dir_data=args.dir_data,
            data_filename=args.data_filename,
            train_bs=args.batch_size,
            test_bs=args.test_batch_size,
            use_normalize='standard',
        )

        torch.manual_seed(fold_k * 42)
        model = Transformer(CLASS_NUM, **f_config).to(device)

        if fold_k == 0:
            logging.info(f"Model device: {next(model.parameters()).device}")
            if device.type == 'mps':
                logging.info(f"MPS memory after model init: "
                             f"{torch.mps.current_allocated_memory() / 1e6:.1f} MB")

        config = build_config(args, fold_dir)

        best_model_path = train_baseline(
            config, model, data_loaders, device, fold_dir, fold_timestamp
        )

        if best_model_path and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))

        fold_acc, y_true, y_pred = eval_best_model(
            model, data_loaders, ACTIVITIES,
            f'Fold{fold_k+1:02d}', 'test', device, fold_dir, fold_timestamp,
            return_predictions=True,
        )
        fold_accuracies.append(float(fold_acc))
        all_y_true = np.concatenate([all_y_true, y_true.astype(int)])
        all_y_pred = np.concatenate([all_y_pred, y_pred.astype(int)])
        logging.info(f"Fold {fold_k+1:2d} best accuracy: {fold_acc:.2f}%")

    # ── Summary ──────────────────────────────────────────────────────────
    mean_acc = float(np.mean(fold_accuracies))
    std_acc  = float(np.std(fold_accuracies))

    logging.info(f"\n{'='*60}")
    logging.info(f"{args.n_splits}-Fold CV Summary  (5-plex dLAMP)")
    logging.info(f"{'='*60}")
    for i, acc in enumerate(fold_accuracies):
        logging.info(f"  Fold {i+1:2d}: {acc:.2f}%")
    logging.info(f"{'─'*40}")
    logging.info(f"  Mean : {mean_acc:.2f}%")
    logging.info(f"  Std  : {std_acc:.2f}%")
    logging.info(f"  Final: {mean_acc:.2f}% ± {std_acc:.2f}%")
    logging.info(f"{'='*60}")

    summary_path = os.path.join(cv_dir, 'cv_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"5-plex dLAMP  —  {args.n_splits}-Fold Stratified Cross-Validation\n")
        f.write(f"Classes : {ACTIVITIES}\n")
        f.write(f"Run     : {run_timestamp}\n")
        f.write(f"Device  : {device}  |  iter={args.num_iterations}  "
                f"lr={args.lr}  bs={args.batch_size}\n\n")
        for i, acc in enumerate(fold_accuracies):
            f.write(f"Fold {i+1:2d}: {acc:.4f}%\n")
        f.write(f"\nMean : {mean_acc:.4f}%\n")
        f.write(f"Std  : {std_acc:.4f}%\n")
        f.write(f"Final: {mean_acc:.4f}% ± {std_acc:.4f}%\n")
    logging.info(f"Summary saved to: {summary_path}")

    if len(all_y_true) > 0:
        agg_acc = save_aggregated_confusion_matrix(all_y_true, all_y_pred, ACTIVITIES, cv_dir, run_timestamp)
        logging.info(f"Aggregated CV accuracy ({len(all_y_true)} samples): {agg_acc:.2f}%")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(1, folds_to_run + 1), fold_accuracies, color='steelblue', alpha=0.8)
    plt.axhline(mean_acc, color='red',    linestyle='--', linewidth=1.5, label=f'Mean {mean_acc:.1f}%')
    plt.axhline(mean_acc + std_acc, color='orange', linestyle=':', linewidth=1, label=f'±1 Std')
    plt.axhline(mean_acc - std_acc, color='orange', linestyle=':', linewidth=1)
    for bar, acc in zip(bars, fold_accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{acc:.1f}', ha='center', va='bottom', fontsize=9)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.title(f'5-plex dLAMP  {args.n_splits}-Fold CV  (Mean={mean_acc:.2f}% ± {std_acc:.2f}%)')
    plt.xticks(range(1, folds_to_run + 1))
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    chart_path = os.path.join(cv_dir, 'cv_accuracy_chart.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logging.info(f"Accuracy chart saved to: {chart_path}")


if __name__ == '__main__':
    main()
