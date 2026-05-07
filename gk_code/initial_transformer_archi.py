import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Optimization: Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("MPS not found, falling back to CPU")

class SignalTransformer(nn.Module):
    def __init__(self, seq_len, num_classes, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.input_projection = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, 
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.input_projection(x.unsqueeze(-1)) + self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        return self.classifier(x)

class ChipDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_loaders(df, filter_col=None, batch_size=32):
    # 1. Filter Outliers if specified
    if filter_col:
        df = df[df[filter_col] == 1].copy()
    
    # 2. Extract Features and Encode Labels
    X = df.filter(like="Cycle").values
    le = LabelEncoder()
    y = le.fit_transform(df['well'])
    
    # 3. Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    train_loader = DataLoader(
        ChipDataset(X_train, y_train), 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0 # Set to 4 if running as a standalone script (.py)
    )
    
    val_loader = DataLoader(
        ChipDataset(X_val, y_val), 
        batch_size=batch_size,
        num_workers=0
    )
    
    return train_loader, val_loader, le, X.shape[1], len(le.classes_)

def train_model(model, train_loader, val_loader, epochs=50, label_smoothing=0.1, lr=0.001, weight_decay=1e-2):
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.inference_mode():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    outputs = model(signals)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            print(f"Epoch {epoch+1} | Val Acc: {correct/total:.2%}")

def predict(model, data, label_encoder):
    model.eval()
    inputs = torch.as_tensor(data, dtype=torch.float32).to(device)
    
    with torch.inference_mode():
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        classes = logits.argmax(dim=1).cpu().numpy()
    
    return label_encoder.inverse_transform(classes), probs.cpu().numpy()

# --- 1. Setup Data ---
# Let's use your best performing outlier filter from previous tests
best_filter = 'mean_outlier_label_0.025' 
train_loader, val_loader, encoder, seq_len, num_wells = prepare_loaders(chip_data, filter_col=best_filter)

# --- 2. Initialize and Train ---
transformer = SignalTransformer(seq_len=seq_len, num_classes=num_wells)
train_model(transformer, train_loader, val_loader, epochs=100)

# --- 3. Run Inference ---
# Taking a few raw samples for testing
sample_curves = chip_data.filter(like="Cycle").values[:5]
predicted_wells, probabilities = predict(transformer, sample_curves, encoder)

print("\n--- Predictions ---")
for well, prob in zip(predicted_wells, probabilities):
    print(f"Predicted: {well} | Confidence: {np.max(prob):.2%}")