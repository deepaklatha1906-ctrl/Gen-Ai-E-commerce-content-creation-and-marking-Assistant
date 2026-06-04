import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize client once - handles SSL/SNI/DNS automatically
client = InferenceClient(
    model="black-forest-labs/FLUX.1-schnell",
    token=HUGGINGFACE_API_KEY,
    timeout=120
)


def generate_marketing_image(prompt: str):
    """Generate marketing image using HF FLUX.1-schnell via InferenceClient"""
    try:
        # InferenceClient manages TLS handshake properly
        image = client.text_to_image(
            prompt,
            num_inference_steps=4
        )
        return image  # Already a PIL.Image object
    except Exception as e:
        error_msg = str(e).lower()
        if "loading" in error_msg or "503" in error_msg:
            raise Exception("Model is currently loading on Hugging Face. Please wait 30 seconds and try again.")
        raise Exception(f"Image generation failed: {str(e)}")