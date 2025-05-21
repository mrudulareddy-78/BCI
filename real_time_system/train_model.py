# Training script for CNN+LSTM with accuracy + confusion matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from cnn_lstm_model import CNNLSTM
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset class
class EEGDataset(Dataset):
    def __init__(self, csv_file, window_size=30):
        df = pd.read_csv(csv_file, on_bad_lines='warn')
        df = df.dropna()
        if df.shape[1] != 12:
            raise ValueError(f"Expected 12 columns, got {df.shape[1]}")
        df = df[df['attention_state'].notna()]

        label_map = {'Focused': 0, 'Relaxed': 1, 'Drowsy': 2, 'Neutral': 3}
        df['label'] = df['attention_state'].map(label_map)

        features = df[['mental_param_1', 'mental_param_2', 'mental_param_3',
                       'spectral_delta', 'spectral_theta', 'spectral_alpha',
                       'spectral_beta', 'spectral_gamma']].values
        labels = df['label'].values

        self.X, self.y = [], []
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

# Load dataset
dataset = EEGDataset('../dataset/brain_data_log.csv', window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model setup
model = CNNLSTM(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with accuracy tracking
all_preds = []
all_labels = []

for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    epoch_preds = []
    epoch_labels = []

    for inputs, labels in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        epoch_preds.extend(preds.cpu().numpy())
        epoch_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(epoch_labels, epoch_preds)
    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {total_loss:.4f} — Accuracy: {acc:.4f}")
    all_preds.extend(epoch_preds)
    all_labels.extend(epoch_labels)

# Save model
torch.save(model.state_dict(), 'model_weights.pth')
print("✅ Model saved to model_weights.pth")

# Final evaluation report
print("\n🧠 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Focused', 'Relaxed', 'Drowsy', 'Neutral']))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Focused', 'Relaxed', 'Drowsy', 'Neutral'],
            yticklabels=['Focused', 'Relaxed', 'Drowsy', 'Neutral'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("🧠 Confusion Matrix")
plt.tight_layout()
plt.show()
