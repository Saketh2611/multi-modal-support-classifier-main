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
# Ensure label_classes.csv exists or handle the error if not for this demo
try:
    label_classes = pd.read_csv("label_classes.csv", header=None)[0].tolist()
except FileNotFoundError:
    label_classes = ["Billing", "Technical Support", "Hardware", "Feature Request"] # Fallback

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
# Added try-except to prevent crash if model file is missing during testing
try:
    model.load_state_dict(torch.load("multimodal_classifier.pth", map_location=torch.device("cpu")))
except FileNotFoundError:
    st.warning("⚠️ 'multimodal_classifier.pth' not found. Using initializing weights for demo.")
model.eval()

# --- Whisper for Audio ---
# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# --- Streamlit UI ---
st.set_page_config(page_title="🎧 Multi-modal Support Classifier", layout="centered")
st.title("🧠 Multi-modal Support Ticket Classifier")
st.markdown("Upload ticket text, screenshot, or record a voice message to classify it.")

# 1. Inputs
text_input = st.text_area("📄 Ticket Description")
image_input = st.file_uploader("🖼 Upload Screenshot", type=["png", "jpg", "jpeg"])

st.markdown("### 🎤 Audio Input")
st.caption("Upload a file OR record directly.")
col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])
with col2:
    audio_mic = st.audio_input("Record Voice Message")

if st.button("Classify"):
    # 2. Determine Audio Source (Mic takes priority if both exist, or you can check either)
    audio_source = audio_mic if audio_mic else audio_file

    # --- Transcribe Audio if Present ---
    if audio_source is not None:
        with st.spinner("Transcribing audio..."):
            # Save to temp file because Whisper expects a file path
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_source.read())
            
            result = whisper_model.transcribe("temp_audio.mp3")
            text_input = result["text"]
            st.info(f"🎧 Transcribed Audio: {text_input}")

    # 3. Validation
    if not text_input or not image_input:
        st.error("Please provide both text (or audio) and an image.")
    else:
        with st.spinner("Analyzing..."):
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
            
            # Create a nice bar chart instead of just text
            prob_dict = {label_classes[i]: probs[0][i].item() for i in range(len(label_classes))}
            st.bar_chart(prob_dict)

            # --- Log Prediction ---
            with open("prediction_logs.csv", mode="a", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    datetime.now().isoformat(),
                    text_input,
                    image_input.name if image_input else "None",
                    "Mic Input" if audio_mic else (audio_file.name if audio_file else "None"),
                    label_classes[pred],
                    *[f"{p.item():.4f}" for p in probs[0]]
                ])