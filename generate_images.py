import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}

def generate_image(topic, explanation, image_path):
    prompt = f"Educational diagram about {topic}, clear illustration, high quality"

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})

    if response.status_code != 200:
        print("HF Error:", response.text)
        return False

    content_type = response.headers.get("content-type", "")

    if "image" not in content_type:
        print("HF Not Image:", response.text)
        return False

    with open(image_path, "wb") as f:
        f.write(response.content)

    return True
