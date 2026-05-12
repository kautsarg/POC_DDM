# Transformer-Based Baseline Training for PCR Amplification Curve Classification

A deep learning framework for classifying PCR amplification curves using Transformer architecture with supervised learning (baseline method).

## Project Structure

```
Code_David_Transformer/
├── main.py                          # Main training script
├── README.md                        # Project documentation
├── QUICKSTART.md                    # Quick start guide
├── requirements.txt                 # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code
│   ├── models/                     # Model architectures
│   │   ├── transformer.py         # Transformer model
│   │   └── utils.py                # Model utilities
│   │
│   ├── data/                       # Data handling
│   │   ├── prepare_data.py        # Data preparation (label encoding)
│   │   └── data_loader.py         # PyTorch data loaders
│   │
│   ├── training/                   # Training scripts
│   │   ├── train_baseline.py       # Baseline training
│   │   └── evaluation.py           # Model evaluation
│   │
│   └── utils/                      # Utilities
│       └── lr_schedule.py         # Learning rate scheduling
│
├── tst/                            # Transformer sub-modules
│   ├── encoder.py                 # Transformer encoder
│   ├── multiHeadAttention.py      # Multi-head attention
│   └── ...
│
├── data/                           # Your data goes here
└── output/                         # Training outputs
```

## Features

- **Baseline Training**: Supervised learning on source domain only
- **Transformer Architecture**: Multi-head self-attention with positional encoding
- **Data Preparation**: Automatic label encoding from strings to integers
- **Evaluation**: Confusion matrices, classification reports, accuracy metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your CSV files in the `data/` directory. Each CSV should contain:
- `Target`: String labels (e.g., 'Spneu', 'Cneo', etc.)
- Numeric columns: Cycle values (1, 2, 3, ..., 50) representing fluorescence
- Other metadata columns as needed

Run data preparation:
```bash
python src/data/prepare_data.py --data_dir data/ --output_dir data/prepared/
```

This will create `Target_cat` column with integer labels (0-7).

### 3. Update Configuration

Edit `main.py` to match your data file names:
- Line 95: Source data file name
- Line 99: Test data file name  
- Line 175: Class names for evaluation
- Line 143: Number of classes

### 4. Train Model

```bash
python main.py --num_iterations 30000 --test_interval 500
```

### 5. Customize Training

```bash
python main.py \
    --num_iterations 50000 \
    --lr 5e-4 \
    --batch_size 128 \
    --test_interval 1000 \
    --dir_data ./data \
    --dir_out ./output
```

## Output

Training produces:
- **Model**: `output/baseline_best_model_{timestamp}.pth`
- **Logs**: `output/training_log_{timestamp}.txt`
- **Plots**: 
  - Training/test accuracy curves
  - Confusion matrix

## Model Architecture

- **Input**: 50-cycle PCR curves (sequence length = 50)
- **Transformer**: 4 encoder layers, 4 attention heads, 128-dim hidden
- **Classification**: Flatten → Linear(6400 → 128) → Bottleneck(128 → 256) → Output(256 → num_classes)

## Data Format

### Input CSV Files
- **Target column**: String class names
- **Numeric columns**: Cycle values (e.g., columns named '1', '2', ..., '50')
- **Optional metadata**: Concentration, Sample_ID, etc.

### After Preparation
- **Target_cat column**: Integer labels (0 to num_classes-1)
- Files saved to `data/prepared/`

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 1.9.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- scikit-learn >= 0.24.0
- Matplotlib >= 3.3.0

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_iterations | 30000 | Total training iterations |
| test_interval | 500 | Evaluation frequency |
| lr | 1e-3 | Learning rate |
| batch_size | 64 | Training batch size |
| test_batch_size | 128 | Test batch size |
| d_model | 128 | Transformer hidden dimension |
| N | 4 | Number of encoder layers |
| h | 4 | Number of attention heads |

## Troubleshooting

1. **Import Errors**: Make sure you're running from the project root directory
2. **CUDA Out of Memory**: Reduce batch size with `--batch_size 32`
3. **Missing Data**: Check that CSV files exist and have correct column names
4. **Label Errors**: Ensure `Target_cat` column exists (run data preparation first)

## Citation

If you use this code, please cite the original Transformer and domain adaptation papers.
