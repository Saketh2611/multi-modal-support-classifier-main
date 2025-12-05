# app.py

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision.models import resnet18
from torchvision import transforms
import pandas as pd
import whisper
from datetime import datetime
import csv

# --- Load Label Encoder ---
label_classes = pd.read_csv("label_classes.csv", header=None)[0].tolist()

# --- Text Tokenizer ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# --- Image Transform ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# --- MultiModal Classifier ---
class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalClassifier, self).__init__()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)

        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)

        self.fusion = nn.Linear(256 + 256, 128)
        self.classifier = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, image):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = self.text_fc(text_out.pooler_output)

        image_embed = self.cnn(image)

        combined = torch.cat((text_embed, image_embed), dim=1)
        x = self.relu(self.fusion(combined))
        x = self.dropout(x)
        out = self.classifier(x)
        return out

# --- Load Trained Model ---
model = MultiModalClassifier(num_classes=len(label_classes))
model.load_state_dict(torch.load("multimodal_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# --- Whisper for Audio ---
whisper_model = whisper.load_model("base")

# --- Streamlit UI ---
st.set_page_config(page_title="🎧 Multi-modal Support Classifier", layout="centered")
st.title("🧠 Multi-modal Support Ticket Classifier")
st.markdown("Upload ticket text, screenshot, or voice message to classify it.")

text_input = st.text_area("📄 Ticket Description")
image_input = st.file_uploader("🖼 Upload Screenshot", type=["png", "jpg", "jpeg"])
audio_input = st.file_uploader("🎤 Upload Voice Message (optional)", type=["mp3", "wav", "m4a"])

if st.button("Classify"):
    # --- Transcribe Audio if Present ---
    if audio_input is not None:
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_input.read())
        result = whisper_model.transcribe("temp_audio.mp3")
        text_input = result["text"]
        st.info(f"🎧 Transcribed Audio: {text_input}")

    if not text_input or not image_input:
        st.error("Please provide both text and image (or transcribed audio).")
    else:
        # --- Process Text ---
        tokens = tokenizer(text_input, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # --- Process Image ---
        image = Image.open(image_input).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0)

        # --- Inference ---
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item() * 100

        # --- Display Result ---
        st.success(f"✅ **Predicted Category:** {label_classes[pred]} ({confidence:.2f}% confidence)")
        st.subheader("🔢 Class Probabilities")
        for i, p in enumerate(probs[0]):
            st.write(f"- {label_classes[i]}: {p.item():.4f}")

        # --- Log Prediction ---
        with open("prediction_logs.csv", mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                text_input,
                image_input.name if image_input else "None",
                audio_input.name if audio_input else "None",
                label_classes[pred],
                *[f"{p.item():.4f}" for p in probs[0]]
            ])
