import os
import base64
import traceback
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1")

# Initialize client once
client = genai.Client(api_key=GEMINI_API_KEY)


def analyze_image(image_bytes: bytes) -> str:
    """Analyze product image using Gemini Vision API"""
    try:
        # Encode image for Gemini
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Describe this e-commerce product image in detail for SEO purposes. "
                "Include: product type, color, material, style, key features visible, "
                "and suggested keywords. Be concise but comprehensive.",
                image_part
            ]
        )

        if response.text:
            return response.text.strip()
        return "Could not analyze image."

    except Exception as e:
        full_error = traceback.format_exc()
        print(f"\n{'='*60}")
        print(f"GEMINI IMAGE ANALYSIS ERROR:\n{full_error}")
        print(f"{'='*60}\n")
        raise Exception(f"Image analysis failed: {str(e)}")