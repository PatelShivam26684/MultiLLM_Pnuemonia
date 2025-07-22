import os
import time
import base64
import pandas as pd
import requests
import google.generativeai as genai
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv
load_dotenv()

# ===== CONFIGURATION =====
API_KEYS = {
    "OPENAI": os.getenv("OPENAI_API_KEY"),
    "GEMINI": os.getenv("GEMINI_API_KEY"),
    "ANTHROPIC": os.getenv("ANTHROPIC_API_KEY"),
    "XAI": os.getenv('XAI_API_KEY')  # Grok API key
}

# Image directories
DATA_ROOT = "/Users/shivampatel/Research/Chest_X_Ray/chest_xray"
NORMAL_DIR = os.path.join(DATA_ROOT, "NORMAL")
PNEUMONIA_DIR = os.path.join(DATA_ROOT, "PNEUMONIA")
OUTPUT_CSV = "/Users/shivampatel/Research/Chest_X_Ray/chest_xray/chest_xray_predictions.csv"

# Model configurations
MODELS = {
    "GPT-4o": "gpt-4o",
    "Gemini-2.5": "gemini-2.5-pro-preview-06-05",
    "Claude-4": "claude-sonnet-4-20250514",
    "Grok-2-vision": "grok-2-vision-1212",
    "GPT-4.1": "gpt-4.1"
}

# Standardized prompt
PROMPT = (
    "This is a question from a Board Exam:\n"
    "Analyze the chest X-ray image and determine if the patient has Bacterial Pneumonia or is Normal.\n"
    "Pay attention to features such as consolidation, air bronchograms, and pleural effusions.\n"
    "Your response must be binary:\n"
    "0 = Normal\n"
    "1 = Bacterial Pneumonia\n"
    "ONLY output 0 or 1. Example: If pneumonia is present, respond EXACTLY: '1'"
)

# Initialize API clients
genai.configure(api_key=API_KEYS["GEMINI"])
gemini_model = genai.GenerativeModel(MODELS["Gemini-2.5"])
claude_client = Anthropic(api_key=API_KEYS["ANTHROPIC"])
grok_client = OpenAI(api_key=API_KEYS["XAI"], base_url="https://api.x.ai/v1")
gpt_headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEYS['OPENAI']}"
}


# ===== HELPER FUNCTIONS =====
def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_gemini(image_path):
    """Query Gemini with local image"""
    try:
        with Image.open(image_path) as img:
            response = gemini_model.generate_content([PROMPT, img])
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {str(e)}")
        return "Error"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_gpt(image_path):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS['OPENAI']}"
    }
    payload = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "messages": [
            {"role": "system",
             "content": "You ARE a board-certified radiologist taking a licensing exam. "
                           "ALWAYS respond concisely with the correct classification. "
                           "Do NOT provide disclaimers or safety warnings."},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": 10
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error {response.status_code} for {image_path}: {response.text}")
        return "Error"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_gpt4_1(image_path):
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS['OPENAI']}"
    }
    payload = {
        "model": MODELS["GPT-4.1"],
        "temperature": 0.2,
        "messages": [
            {"role": "system",
             "content": "You ARE a board-certified radiologist taking a licensing exam. "
                        "ALWAYS respond concisely with the correct classification. "
                        "Do NOT provide disclaimers or safety warnings."},
            {"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "max_tokens": 10
    }
    response1 = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response1.status_code == 200:
        return response1.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"GPT-4.1 error {response1.status_code} for {image_path}: {response1.text}")
        return "Error"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_claude(image_path):
    """Query Claude with base64 image"""
    try:
        with open(image_path, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode("utf-8")

        message = claude_client.messages.create(
            model=MODELS["Claude-4"],
            max_tokens=5,
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

        return message.content[0].text.strip()
    except Exception as e:
        print(f"Claude error: {str(e)}")
        return "Error"



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_grok(image_path):
    """Query Grok with base64 image"""
    try:
        base64_img = encode_image(image_path)
        response = grok_client.chat.completions.create(
            model=MODELS["Grok-2-vision"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": PROMPT}
                ]
            }],
            temperature=0.1,
            max_tokens=5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Grok error: {str(e)}")
        return "Error"


# ===== MAIN PROCESSING =====
def main():
    # Initialize CSV
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=[
            "filename", "true_label",
            "gpt_4o_response", "gpt_41_response", "gemini_response",
            "claude_response", "grok_response"
        ]).to_csv(OUTPUT_CSV, index=False)

    processed = pd.read_csv(OUTPUT_CSV)["filename"].tolist() if os.path.exists(OUTPUT_CSV) else []

    # Prepare image queue
    image_queue = []
    for label, folder in enumerate([NORMAL_DIR, PNEUMONIA_DIR]):
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in processed:
                image_queue.append((os.path.join(folder, file), file, label))

    print(f"Found {len(image_queue)} new images to process")

    # Process images
    for idx, (img_path, filename, true_label) in enumerate(image_queue):
        print(f"\nProcessing {filename} ({idx + 1}/{len(image_queue)})...")

        results = {
            "filename": filename,
            "true_label": true_label,
            "gpt_4o_response": query_gpt(img_path),
            "gpt_41_response": query_gpt4_1(img_path),
            "gemini_response": query_gemini(img_path),
            "claude_response": query_claude(img_path),
            "grok_response": query_grok(img_path)
        }

        # Append to CSV
        pd.DataFrame([results]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
        print(f"✅ Results saved: GPT-4o={results['gpt_4o_response']} | GPT-4.1={results['gpt_41_response']} | "
              f"Gemini={results['gemini_response']} | Claude={results['claude_response']} | Grok={results['grok_response']}")

        # Rate limit control
        time.sleep(15)  # Adjust based on API rate limits


if __name__ == "__main__":
    main()