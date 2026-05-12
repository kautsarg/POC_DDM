# Quick Start Guide - Baseline Training Only

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Your Data**:
   - Place CSV files in `data/` directory
   - Each CSV should have:
     - `Target` column with string class names
     - Numeric columns (1, 2, 3, ..., 50) for PCR cycles
   
   Run data preparation:
   ```bash
   python src/data/prepare_data.py --data_dir data/ --output_dir data/prepared/
   ```

3. **Update Configuration in main.py**:
   - Line 95: Change source data file name
   - Line 99: Change test data file name
   - Line 175: Update class names list
   - Line 143: Change number of classes if needed

## Training

### Basic Training:
```bash
python main.py
```

### With Custom Parameters:
```bash
python main.py \
    --num_iterations 50000 \
    --lr 5e-4 \
    --batch_size 128 \
    --test_interval 1000
```

### Full Example:
```bash
python main.py \
    --num_iterations 30000 \
    --test_interval 500 \
    --lr 1e-3 \
    --batch_size 64 \
    --test_batch_size 128 \
    --dir_data ./data \
    --dir_out ./output
```

## Outputs

All outputs are saved to `output/` directory:
- **Model**: `output/baseline_best_model_{timestamp}.pth`
- **Logs**: `output/training_log_{timestamp}.txt`
- **Plots**: 
  - `output/training_curve_baseline_model_{timestamp}.png` - Training/test accuracy curves
  - `output/confusion_matrix_Baseline_{timestamp}.png` - Confusion matrix

## Customization

### Change Number of Classes:
Edit `main.py` line 143:
```python
"class_num": 8  # Change to your number of classes
```

### Change Class Names:
Edit `main.py` line 175:
```python
Activities = ['Class1', 'Class2', 'Class3', ...]  # Your class names
```

### Modify Transformer Architecture:
Edit `main.py` lines 147-162 (F_config dictionary):
- `d_model`: Hidden dimension (default: 128)
- `N`: Number of encoder layers (default: 4)
- `h`: Number of attention heads (default: 4)
- `dropout`: Dropout rate (default: 0.3)

## What is Baseline Training?

Baseline training uses **supervised learning** on the source domain only:
- Trains on labeled source data
- Uses CrossEntropyLoss for classification
- No domain adaptation
- Simple and effective for same-domain classification

## Troubleshooting

1. **Import Errors**: Run from project root: `python main.py`
2. **CUDA Out of Memory**: Reduce batch size: `--batch_size 32`
3. **Missing Data**: Check CSV files exist in `data/` directory
4. **Label Errors**: Run data preparation first to create `Target_cat` column
