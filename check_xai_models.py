from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os


api_key = os.getenv("XAI_API_KEY")
grok_client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
models = grok_client.models.list()
print("Available Grok models:")
for m in models.data:
    print(f"- {m.id}")