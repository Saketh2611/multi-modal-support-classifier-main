# Use a lightweight Python version (Slim) to keep image size down
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for Whisper (FFmpeg) and Image processing
# - ffmpeg: Required by OpenAI Whisper for audio processing
# - git: Sometimes needed if installing specific pip packages from source
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (to cache dependencies)
COPY requirements.txt .

# Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.address=0.0.0.0"]