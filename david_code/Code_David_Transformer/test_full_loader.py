import sys
import os
sys.path.append('src')
from src.data.data_loader import generate_data_loader

print("=== TESTING FULL DATA LOADER ===")
try:
    data_loaders = generate_data_loader(
        './data', 16, 16, 32,
        ['3plex_2000_truncated.csv'],
        '3plex_2000_truncated.csv', 
        '3plex_2000_truncated.csv',
        'cpu', 'None'
    )
    
    print("\n=== CHECKING SOURCE LOADER ===")
    for batch_data, batch_labels in data_loaders['source']:
        print(f'Batch data shape: {batch_data.shape}')
        print(f'Batch labels shape: {batch_labels.shape}')
        print(f'Unique labels: {batch_labels.unique()}')
        print(f'Label counts: {batch_labels.bincount()}')
        break
        
    print("\n=== CHECKING TEST LOADER ===")
    for batch_data, batch_labels in data_loaders['test']:
        print(f'Batch data shape: {batch_data.shape}')
        print(f'Batch labels shape: {batch_labels.shape}')
        print(f'Unique labels: {batch_labels.unique()}')
        print(f'Label counts: {batch_labels.bincount()}')
        break
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
