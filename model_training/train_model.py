# Training script for CNN+LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from cnn_lstm_model import CNNLSTM

# Define dataset
class EEGDataset(Dataset):
    def __init__(self, csv_file, window_size=30):
        # Read CSV with strict column checking (12 expected)
        df = pd.read_csv(csv_file, on_bad_lines='warn')  # pandas 1.3+


# Drop rows that don't have exactly 12 columns (including attention state)
        df = df.dropna()
        if df.shape[1] != 12:
          raise ValueError(f"Expected 12 columns in dataset, but got {df.shape[1]}")
        df = df.dropna()
        df = df[df['attention_state'].notna()]

        label_map = {'Focused': 0, 'Relaxed': 1, 'Drowsy': 2, 'Neutral': 3}
        df['label'] = df['attention_state'].map(label_map)

        features = df[['mental_param_1', 'mental_param_2', 'mental_param_3',
                       'spectral_delta', 'spectral_theta', 'spectral_alpha',
                       'spectral_beta', 'spectral_gamma']].values
        labels = df['label'].values

        self.X = []
        self.y = []

        for i in range(len(features) - window_size):
            self.X.append(features[i:i+window_size])
            self.y.append(labels[i+window_size-1])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Hyperparameters
input_size = 8
hidden_size = 128
num_classes = 4
window_size = 30
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Dataset and loader
dataset = EEGDataset('../dataset/brain_data_log.csv', window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
model = CNNLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'model_weights.pth')
print("Model saved to model_weights.pth")
