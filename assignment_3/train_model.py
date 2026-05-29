import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import os

# 1. Dataset Class
class HAR_Dataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
            
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx].permute(1, 0), self.y[idx] # (C, L)
        else:
            return self.X[idx].permute(1, 0)

# 2. Model Architecture: 1D CNN + LSTM
class HAR_Model(nn.Module):
    def __init__(self, num_classes=6):
        super(HAR_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # After 3 MaxPool1d(2), length 300 -> 150 -> 75 -> 37
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # x is (Batch, 256, 37). Convert to (Batch, 37, 256) for LSTM
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        # Take the last output of LSTM
        x = lstm_out[:, -1, :]
        
        x = self.fc(x)
        return x

# 3. Training Function
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    epoch_loss = running_loss / len(loader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_f1

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            running_loss += loss.item() * X.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    val_loss = running_loss / len(loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return val_loss, val_f1

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    data_path = "assignment_3/nycu-data-mining-assignment-3/train_data.npz"
    data = np.load(data_path)
    X = data['X'] # (11020, 300, 6)
    y = data['y']
    groups = data['user_ids']
    
    # Scaling
    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, 6)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)
    
    gkf = GroupKFold(n_splits=5)
    
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        print(f"\n--- Fold {fold+1} ---")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_ds = HAR_Dataset(X_train, y_train)
        val_ds = HAR_Dataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        model = HAR_Model().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        best_val_f1 = 0.0
        epochs = 10
        
        for epoch in range(epochs):
            train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")
            
            scheduler.step(val_f1)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"assignment_3/best_model_fold{fold+1}.pth")
        
        fold_f1s.append(best_val_f1)
        break # Testing one fold first
        
    print(f"\nAverage Cross-Validation F1 Score: {np.mean(fold_f1s):.4f}")
