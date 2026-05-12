import sys
import os
sys.path.append('src')
from src.data.data_loader import generate_data_loader

print("=== TESTING DETAILED DATA LOADER ===")
data_loaders = generate_data_loader(
    './data', 16, 16, 32,
    ['3plex_2000_truncated.csv'],
    '3plex_2000_truncated.csv',
    '3plex_2000_truncated.csv',
    'cpu', 'None'
)

print("\n=== CHECKING FIRST 5 BATCHES FROM TEST LOADER ===")
batch_count = 0
for batch_data, batch_labels in data_loaders['test']:
    batch_count += 1
    print(f"Batch {batch_count}:")
    print(f"  Data shape: {batch_data.shape}")
    print(f"  Labels shape: {batch_labels.shape}")
    print(f"  Unique labels: {batch_labels.unique()}")
    print(f"  Label counts: {batch_labels.bincount()}")
    
    if batch_count >= 5:
        break

print(f"\nTotal batches in test loader: {len(data_loaders['test'])}")
print(f"Test dataset size: {len(data_loaders['test'].dataset)}")
print(f"Batch size: {data_loaders['test'].batch_size}")

# Check the dataset directly
test_dataset = data_loaders['test'].dataset
print("\n=== DIRECT DATASET CHECK ===")
print(f"Dataset length: {len(test_dataset)}")
print(f"First 10 labels: {test_dataset.y[:10]}")
print(f"Last 10 labels: {test_dataset.y[-10:]}")
print(f"All unique labels in dataset: {test_dataset.y.unique()}")
print(f"Label distribution in dataset: {test_dataset.y.bincount()}")
