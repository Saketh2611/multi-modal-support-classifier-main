# preprocess.py
import numpy as np
print(np.__version__)
print(np.array([1, 2, 3]))

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# --- Load Data and Initialize ---
df = pd.read_csv("tickets_final.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Save encoder classes for later use
pd.Series(label_encoder.classes_).to_csv("label_classes.csv", index=False,header=False)

# --- Image Transform ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Custom Dataset ---
class TicketDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["ticket_text"]
        image_path = row["ticket_picture"]
        label = row["label_encoded"]

        # Process text
        text_tokens = tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids = text_tokens['input_ids'].squeeze(0)
        attention_mask = text_tokens['attention_mask'].squeeze(0)

        # Process image
        image = Image.open(image_path).convert("RGB")
        image_tensor = image_transform(image)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image_tensor,
            "label": torch.tensor(label)
        }

# --- Save Dataset Loader Preview ---
if __name__ == "__main__":
    dataset = TicketDataset(df)
    sample = dataset[0]
    print("Input IDs shape:", sample["input_ids"].shape)
    print("Attention Mask shape:", sample["attention_mask"].shape)
    print("Image shape:", sample["image"].shape)
    print("Label:", sample["label"])
