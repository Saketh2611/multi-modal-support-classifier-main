# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from process import TicketDataset, df
from model import MultiModalClassifier
import torch.optim as optim
from tqdm import tqdm

# --- Hyperparameters ---
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

# --- Dataset and DataLoader ---
dataset = TicketDataset(df)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Model Setup ---
NUM_CLASSES = len(df["label"].unique())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalClassifier(num_classes=NUM_CLASSES).to(device)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct = 0, 0
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = total_correct / len(train_dataset)
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.4f}")

    # --- Validation ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, images)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(val_dataset)
    print(f"Validation Accuracy: {val_acc:.4f}\n")

# --- Save the Trained Model ---
torch.save(model.state_dict(), "multimodal_classifier.pth")
print("✅ Model saved as 'multimodal_classifier.pth'")
