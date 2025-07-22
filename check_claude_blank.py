import os
import hashlib
import base64
import pandas as pd
from PIL import Image
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ==== CONFIGURATION ====
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Replace with your actual key
DATA_ROOT = "/Users/shivampatel/Research/Chest_X_Ray"
NORMAL_DIR = os.path.join(DATA_ROOT, "blank_images")
PNEUMONIA_DIR = os.path.join(DATA_ROOT, "blank_images_2")
OUTPUT_CSV = "/Users/shivampatel/Research/Chest_X_Ray/blank_claude_diagnostic_results.csv"
TMP_IMAGE_DIR = "blank_claude_tmp_images"  # Directory to save decoded base64 previews

# ==== Claude Client ====
claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-20250514"

# ==== Prompt ====
PROMPT = (
    "You are a radiologist taking a licensing exam.\n"
    "Classify the chest X-ray image as follows:\n"
    "0 = Normal\n"
    "1 = Bacterial Pneumonia\n\n"
    "Respond in this format:\n"
    "Prediction: <0 or 1>\n"
    "Reason: <short explanation of the findings>"
)

# ==== Utility Functions ====
def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def save_temp_decoded_image(encoded_data, filename):
    os.makedirs(TMP_IMAGE_DIR, exist_ok=True)
    img_bytes = base64.b64decode(encoded_data)
    path = os.path.join(TMP_IMAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(img_bytes)

def query_claude(image_path):
    try:
        base64_img = encode_image(image_path)
        filename = os.path.basename(image_path)

        # Send to Claude
        message = claude_client.messages.create(
            model=MODEL,
            max_tokens=100,
            temperature=0.1,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_img
                        }
                    },
                    {
                        "type": "text",
                        "text": PROMPT
                    }
                ]
            }]
        )

        response = message.content[0].text.strip()
        save_temp_decoded_image(base64_img, filename)  # save what Claude sees
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# ==== Main Processing ====
def process_folder(folder_path, label):
    results = []
    print(f"\n--- Testing Claude on {label.upper()} images ---")

    for fname in sorted(os.listdir(folder_path))[:10]:  # Test on 10 for now
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder_path, fname)
        sha = sha256_file(path)
        response = query_claude(path)

        print(f"[{label}] {fname}")
        print(f"  SHA256: {sha}")
        print(f"🧠 Claude Raw Response: {response}\n")

        results.append({
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "filename": fname,
            "sha256": sha,
            "claude_response": response
        })

    return results

def main():
    all_results = process_folder(NORMAL_DIR, "NORMAL")
    all_results += process_folder(PNEUMONIA_DIR, "PNEUMONIA")

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved results to {OUTPUT_CSV}")
    print(f"🖼️ Decoded images saved in: {TMP_IMAGE_DIR}/")

if __name__ == "__main__":
    main()