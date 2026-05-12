"""
Main training script for Transformer-based PCR amplification curve classification.

Baseline training: Supervised learning on source domain only.
"""

import os
import sys
import argparse
import warnings
import logging
import datetime
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root and src to path so tst.* and src.* imports work
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
# If project contains a nested folder with the same name (created during setup),
# add it to sys.path so `tst` package can be found.
nested_project_dir = os.path.join(project_root, os.path.basename(project_root))
if os.path.isdir(nested_project_dir):
    sys.path.insert(0, nested_project_dir)

from src.models.transformer import Transformer
from src.data.data_loader import generate_data_loader
from src.training.train_baseline import train_baseline
from src.training.evaluation import eval_best_model


# Default directories (modify as needed)
DEFAULT_DATA_DIR = './data'
DEFAULT_OUTPUT_DIR = './output'


def resolve_device(device_arg):
    """Resolve the requested device, preferring CUDA, then MPS, then CPU."""
    cuda_available = torch.cuda.is_available()
    mps_built = torch.backends.mps.is_built()
    mps_available = mps_built and torch.backends.mps.is_available()

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
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    if device_name == 'mps' and not mps_available:
        if not mps_built:
            raise RuntimeError("MPS was requested but this PyTorch build was not compiled with MPS support.")
        raise RuntimeError(
            "MPS was requested but is not available in this environment. "
            "On this machine, PyTorch currently reports MPS unavailable."
        )

    return torch.device(device_name), {
        "cuda_available": cuda_available,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def setup_logging(dir_out):
    """Setup logging configuration."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(dir_out, f'training_log_{timestamp}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True,
    )
    
    logging.info(f"Starting baseline training. Logs saved to: {log_file}")
    return timestamp


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Transformer-based Baseline Training for PCR Classification'
    )
    parser.add_argument('--num_iterations', type=int, default=30000,
                       help='Number of training iterations')
    parser.add_argument('--test_interval', type=int, default=500,
                       help='Interval for test evaluation')
    parser.add_argument('--dir_data', type=str, default=DEFAULT_DATA_DIR,
                       help='Directory containing data files')
    parser.add_argument('--dir_out', type=str, default=DEFAULT_OUTPUT_DIR,
                       help='Output directory for models and logs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=128,
                       help='Batch size for testing')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Training device to use')
    
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    
    # Setup output directory
    os.makedirs(args.dir_out, exist_ok=True)
    timestamp = setup_logging(args.dir_out)
    
    # Device
    device, device_status = resolve_device(args.device)
    logging.info(f"Device for training: {device}")
    logging.info(
        "Backend availability: "
        f"cuda={device_status['cuda_available']}, "
        f"mps_built={device_status['mps_built']}, "
        f"mps_available={device_status['mps_available']}"
    )
    
    # Configuration
    config = {
        "N_seq": 473,  # Sequence length (PCR cycles) - updated for 3plex_2000_truncated.csv (473 numeric columns)
        "num_iterations": args.num_iterations,
        "test_interval": args.test_interval,
        "dir_out": args.dir_out,
        "optimizer": {
            "type": optim.AdamW,
            "optim_params": {
                'lr': args.lr,
                "weight_decay": 0.0001
            },
            "lr_type": "inv",
            "lr_param": {
                "lr": args.lr,
                "gamma": 0.001,
                "power": 0.75
            }
        },
        "data": {
            "dir_data": args.dir_data,
            "source": {
                "name": ['3plex_2000_truncated.csv'],  # Using new dataset with truncated time points
                "batch_size": args.batch_size
            },
            "target": {
                "name": "3plex_2000_truncated.csv",  # Same file for baseline training
                "batch_size": args.batch_size
            },
            "test": {
                "name": "3plex_2000_truncated.csv",  # Same file for testing
                "batch_size": args.test_batch_size
            },
            "normalize": "min_max"
        },
        "class_num": 3  # Number of classes - 'HAdv' (0), 'IAV' (1), 'IBV' (2)
    }
    
    # Transformer configuration
    F_config = {
        'd_input': 1,
        'd_model': 128,
        'q': 8,
        'v': 8,
        'h': 4,
        'N': 4,
        'attention_size': None,
        'dropout': 0.3,
        'chunk_mode': None,
        'pe': "regular",
        'pe_period': 20,
        'use_bottleneck': True,
        'bottleneck_dim': 256,
        'seq_length': 473  # Updated for 3plex_2000_truncated.csv (473 numeric columns)
    }
    
    logging.info("Configuration: " + str(config))
    logging.info("Transformer Configuration: " + str(F_config))
    
    torch.manual_seed(42)
    
    # Load data
    logging.info("Loading data...")
    data_config = config["data"]
    data_loaders = generate_data_loader(
        data_config["dir_data"],
        data_config["source"]["batch_size"],
        data_config["target"]["batch_size"],
        data_config["test"]["batch_size"],
        data_config["source"]["name"],
        data_config["target"]["name"],
        data_config["test"]["name"],
        device,
        use_normalize='None'  # Or 'min_max' if normalization needed
    )
    logging.info("Data loading complete!")
    
    # Class names for evaluation
    Activities = ['HAdv', 'IAV', 'IBV']  # Three classes: HAdv (0), IAV (1), IBV (2)
    
    # Train Baseline Model
    logging.info("\n\n=========== STARTING BASELINE MODEL TRAINING ===========")
    baseline_transformer = Transformer(
        config['class_num'],
        **F_config
    ).to(device)

    # Verify model is on the expected device
    actual_device = next(baseline_transformer.parameters()).device
    logging.info(f"Model parameters device: {actual_device}")
    if device.type == 'mps':
        logging.info(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1e6:.1f} MB")

    best_model_path = train_baseline(
        config, baseline_transformer, data_loaders,
        device, args.dir_out, timestamp
    )

    # Load the best saved model
    if best_model_path and os.path.exists(best_model_path):
        baseline_transformer.load_state_dict(torch.load(best_model_path, map_location=device))
        logging.info(f"Loaded best model from: {best_model_path}")

    baseline_best_model = baseline_transformer

    # Evaluate the model
    baseline_acc = eval_best_model(
        baseline_best_model, data_loaders, Activities,
        'Baseline', 'test', device, args.dir_out, timestamp
    )
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Baseline Model Final Accuracy: {baseline_acc:.2f}%")
    logging.info(f"{'='*60}")
    logging.info("Training complete!")


if __name__ == "__main__":
    main()
