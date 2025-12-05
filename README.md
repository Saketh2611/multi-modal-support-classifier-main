# 🎧 Multi-modal Support Ticket Classifier

This is an intelligent support ticket classification system that uses **text**, **image**, and optionally **audio** inputs to automatically categorize incoming support tickets.

It leverages:
- **BERT** for text embeddings
- **ResNet18** for image feature extraction
- **Whisper** for transcribing voice messages to text
- A **fusion classifier** for combining the modalities and predicting the support category

---

## 📁 Project Contents

Here are the important files included in this repository:

| File | Description |
|------|-------------|
| `App.py` | The Streamlit web application for uploading and classifying tickets |
| `label_classes.csv` | Contains the class labels used during training (one label per line) |
| `multimodal_classifier.pth` | The trained PyTorch model for inference |
| `train.py` | Script used to train the multimodal classifier |
| `process.py` | Script used to process and prepare the dataset |
| `all_tickets_processed_improved_v3.csv` | Sample dataset used for training |
|`dataset`|Used to create the actual dataset for audio and images|

> **Note**: `requirements.txt` is not included. You can install the needed packages manually (see steps below).

---

## 🚀 Getting Started

Follow the steps below to set up and run the project on your local machine.

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/multi-modal-support-classifier.git
cd multi-modal-support-classifier
```

---

### ✅ Step 2: Set Up a Virtual Environment

It's recommended to use a Python virtual environment.

```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On macOS/Linux
```

---

### ✅ Step 3: Install Required Dependencies

Since `requirements.txt` is not provided, install the packages manually:

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install streamlit
pip install pandas pillow
pip install openai-whisper
```

> ❗ **Fix NumPy Issues**  
If you get `RuntimeError: Numpy is not available`, run:

```bash
pip install numpy==1.24.4
```

> ❗ **Ensure FFmpeg for Whisper**  
Whisper needs FFmpeg to process audio:
- **Windows**: Download from https://ffmpeg.org and add to PATH
- **Linux/macOS**: Run `sudo apt install ffmpeg` or `brew install ffmpeg`

---

### ✅ Step 4: Run the Streamlit App

```bash
streamlit run App.py
```

This will launch the web app in your browser.

---

## 🖼️ How to Use the App

1. **Enter Text**: Type the issue in the text box.
2. **Upload Image**: Choose a screenshot or related image.
3. **(Optional) Upload Audio**: Upload a `.mp3`, `.wav`, or `.m4a` file.
4. **Click "Classify"**: The model will:
   - Transcribe audio (if provided)
   - Process the text and image
   - Predict the category and show confidence levels
5. **All predictions are logged** in `prediction_logs.csv`.

---

## 🧠 Model Architecture

- **Text** → BERT (`bert-base-uncased`) → 256-dim linear projection  
- **Image** → ResNet18 → 256-dim linear projection  
- **Fusion** → Concatenation → Linear(512 → 128) → Dropout → Classifier (Linear → num_classes)

---

## 🛠️ Troubleshooting

- Whisper errors? Make sure FFmpeg is installed.
- Torch/NumPy issues? Downgrade NumPy to `<2.0.0` and use PyTorch 2.1.0.
- Audio not required? You can skip uploading an audio file — text + image are sufficient.

---

## ✨ Future Enhancements

- Add live microphone support via WebRTC (optional)
- Deploy publicly via Streamlit Cloud, Render, or HuggingFace Spaces
- Add log visualization dashboard

---
