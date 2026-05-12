# Project Structure - Baseline Training Only

```
Code_David_Transformer/
├── main.py                          # Main training script (baseline only)
├── README.md                        # Project documentation
├── QUICKSTART.md                    # Quick start guide
├── PROJECT_STRUCTURE.md             # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── models/                     # Model architectures
│   │   ├── __init__.py
│   │   ├── transformer.py         # Transformer model
│   │   └── utils.py                # Model utilities (weight initialization)
│   │
│   ├── data/                       # Data handling
│   │   ├── __init__.py
│   │   ├── prepare_data.py        # Data preparation (label encoding)
│   │   └── data_loader.py         # PyTorch data loaders
│   │
│   ├── training/                   # Training scripts
│   │   ├── __init__.py
│   │   ├── train_baseline.py       # Baseline training (supervised learning)
│   │   └── evaluation.py           # Model evaluation
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       └── lr_schedule.py          # Learning rate scheduling
│
├── tst/                            # Transformer sub-modules
│   ├── __init__.py
│   ├── encoder.py                  # Transformer encoder
│   ├── decoder.py                  # Transformer decoder (not used in baseline)
│   ├── multiHeadAttention.py       # Multi-head attention
│   ├── positionwiseFeedForward.py  # Feed-forward layers
│   ├── utils.py                    # PE generation utilities
│   └── loss.py                     # Transformer-specific losses
│
├── configs/                        # Configuration files (optional)
├── data/                           # Data directory (place CSV files here)
│   └── prepared/                   # Prepared data (after label encoding)
└── output/                         # Training outputs
    ├── models/                     # Saved models (.pth files)
    ├── logs/                       # Training logs
    └── plots/                      # Training curves, confusion matrices
```

## Key Files

### Main Entry Point
- **main.py**: Main training script - baseline training only

### Data Preparation
- **src/data/prepare_data.py**: Converts string labels to integers, normalizes data
- **src/data/data_loader.py**: Creates PyTorch DataLoaders for training

### Models
- **src/models/transformer.py**: Transformer architecture for sequence classification

### Training
- **src/training/train_baseline.py**: Supervised learning on source domain only
- **src/training/evaluation.py**: Model evaluation and confusion matrix

### Utilities
- **src/utils/lr_schedule.py**: Learning rate scheduling (inverse decay)

## Removed Components (CDAN/BYOL)

The following components were removed as they're not needed for baseline training:
- ❌ `src/training/train_cdan.py` - CDAN training
- ❌ `src/training/train_byol_cdan.py` - BYOL+CDAN training
- ❌ `src/models/adversarial_network.py` - Adversarial networks
- ❌ `src/utils/loss.py` - CDAN/DANN loss functions

## Data Format

### Input CSV Files
- Must contain `Target` column with string class names
- Numeric columns representing PCR cycles (e.g., '1', '2', ..., '50')
- Optional metadata columns

### After Preparation
- `Target_cat` column added with integer labels (0 to num_classes-1)
- Files saved to `data/prepared/`

## Usage

1. **Prepare Data**:
   ```bash
   python src/data/prepare_data.py --data_dir data/ --output_dir data/prepared/
   ```

2. **Train Baseline Model**:
   ```bash
   python main.py --num_iterations 30000
   ```

3. **Modify Configuration**:
   - Edit `main.py` to change data file names, batch sizes, etc.
   - Or pass via command-line arguments

## Training Method

**Baseline Training**:
- Supervised learning on source domain only
- Uses CrossEntropyLoss for classification
- No domain adaptation
- Simple and effective for same-domain classification

## Dependencies

See `requirements.txt` for full list. Key packages:
- PyTorch >= 1.9.0
- NumPy, Pandas
- scikit-learn
- Matplotlib
