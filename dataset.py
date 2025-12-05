from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os

# Load dataset
df = pd.read_csv("tickets.csv")

# Create output directories
audio_dir = "audio"
picture_dir = "pictures"
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(picture_dir, exist_ok=True)

# Load font
font = ImageFont.load_default()

# Process each row
for idx, row in df.iterrows():
    text = row["ticket_text"]

    # ----- Create Audio -----
    audio_path = os.path.join(audio_dir, f"ticket_{idx}.mp3")
    tts = gTTS(text=text)
    tts.save(audio_path)
    df.at[idx, "ticket_audio"] = audio_path

    # ----- Create Picture -----
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Wrap text
    words = text.split()
    lines, line = [], ""
    for word in words:
        if len(line + word) < 50:
            line += word + " "
        else:
            lines.append(line.strip())
            line = word + " "
    lines.append(line.strip())
    lines = lines[:12]

    y = 10
    for l in lines:
        draw.text((10, y), l, fill=(0, 0, 0), font=font)
        y += 20

    picture_path = os.path.join(picture_dir, f"ticket_{idx}.png")
    img.save(picture_path)
    df.at[idx, "ticket_picture"] = picture_path

# ----- Save Final CSV -----
df.to_csv("tickets_final.csv", index=False)
print("✅ All done! Final CSV saved as 'tickets_final.csv' with audio + picture paths.")

# Drop 'ticket_image' column if it exists
if 'ticket_image' in df.columns:
    df.drop(columns=['ticket_image'], inplace=True)












