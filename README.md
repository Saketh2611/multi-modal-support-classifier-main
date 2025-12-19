# 🎧 Multi-modal Support Ticket Classifier

This is an intelligent support ticket classification system that uses **text**, **image**, and **audio** inputs to automatically categorize incoming support tickets.

It leverages:
- **BERT** for text embeddings
- **ResNet18** for image feature extraction
- **Whisper** for transcribing voice messages to text
- A **fusion classifier** for combining the modalities and predicting the support category

---

## PROJECT WORKFLOW 
```
https://miro.com/app/board/uXjVGZM3-Z4=/?share_link_id=475132053421
```
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
| `dataset/` | Folder containing raw images and audio for training |
| `prediction_logs.csv` | Automatically generated log of all user predictions |

> **Note**: `requirements.txt` is not included. You can install the needed packages manually (see steps below).

---

## 🚀 Getting Started

Follow the steps below to set up and run the project on your local machine.

### ✅ Step 1: Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/multi-modal-support-classifier.git](https://github.com/YOUR_USERNAME/multi-modal-support-classifier.git)
cd multi-modal-support-classifier
```

### ✅ Step 2: Set Up a Virtual Environment

It is highly recommended to use a Python virtual environment to avoid conflicts.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### ✅ Step 3: Install Required Dependencies

Run the following commands to install the necessary libraries:

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install transformers
pip install streamlit
pip install pandas pillow
pip install openai-whisper
```

> **Note on PyTorch**: The command above installs the CPU version to save space. If you have a GPU, remove `--index-url ...` to get the CUDA version.

### ✅ Step 4: Install FFmpeg (CRITICAL)

The **Whisper** model requires FFmpeg to process audio files. If this is not installed, the app will crash when you try to record or upload audio.

* **Windows (Easiest Method):**
  Open PowerShell as Administrator and run:
  ```powershell
  winget install -e --id Gyan.FFmpeg
  ```
  *Restart your terminal/editor after installing.*

* **Linux/macOS:**
  ```bash
  sudo apt install ffmpeg   # Linux
  brew install ffmpeg       # macOS
  ```

---

## 🏃‍♂️ How to Run

Once everything is installed, launch the application:

```bash
streamlit run App.py
```

This will automatically open the web app in your default browser (usually at `http://localhost:8501`).

---

## 🖼️ How to Use the App

1. **Enter Text**: Type a description of the issue in the text box.
2. **Upload Image**: Upload a screenshot or image related to the issue.
3. **Audio Input** (Choose one):
   * **Upload File**: Upload an `.mp3`, `.wav`, or `.m4a` file.
   * **Record Live**: Use the **"Record Voice Message"** button to speak directly into your microphone.
4. **Click "Classify"**: The model will:
   * Transcribe the audio (using Whisper).
   * Fuse the text (from description + transcription) with the image features.
   * Predict the category and display confidence levels.
5. **Logs**: All predictions are saved to `prediction_logs.csv`.

---

## 🧠 Model Architecture

The model uses a **Late Fusion** approach:

1. **Text Branch**:
   * Input: Ticket text + Audio Transcription
   * Model: BERT (`bert-base-uncased`)
   * Output: 768-dim vector → Projected to 256-dim
2. **Image Branch**:
   * Input: Ticket Screenshot
   * Model: ResNet18 (Pre-trained)
   * Output: 512-dim vector → Projected to 256-dim
3. **Fusion Layer**:
   * Concatenates text (256) + image (256) features.
   * Passes through a Fully Connected Layer (128 units) with ReLU and Dropout.
   * Final Softmax Classifier predicts the class.

---

## 🛠️ Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: ... ffmpeg` | FFmpeg is missing. See **Step 4** above to install it. |
| `RuntimeError: Numpy is not available` | Your NumPy version might be too new for the installed PyTorch. Run: `pip install "numpy<2.0"` |
| `ModuleNotFoundError: No module named 'streamlit'` | Activate your virtual environment first! (`.venv\Scripts\activate`) |
| `AttributeError: module 'streamlit' has no attribute 'audio_input'` | Update Streamlit: `pip install -U streamlit` |

---

## ✨ Future Enhancements

* [ ] Deploy publicly via Streamlit Cloud or Hugging Face Spaces.
* [ ] Add a dashboard to visualize trends in `prediction_logs.csv` (e.g., "Most common issues this week").
* [ ] Implement user feedback loop to retrain the model on incorrect predictions.
