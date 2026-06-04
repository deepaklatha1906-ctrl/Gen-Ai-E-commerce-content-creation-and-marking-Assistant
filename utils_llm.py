import os
import socket
import traceback
from dotenv import load_dotenv
from google import genai

# === DNS FALLBACK FOR GEMINI ===
_original_getaddrinfo = socket.getaddrinfo

def _patched_getaddrinfo(*args, **kwargs):
    try:
        return _original_getaddrinfo(*args, **kwargs)
    except socket.gaierror:
        host = args[0] if args else kwargs.get('host', '')
        if host == 'generativelanguage.googleapis.com':
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('142.250.80.42', 443))]
        raise

socket.getaddrinfo = _patched_getaddrinfo
# ================================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_2")
client = genai.Client(api_key=GEMINI_API_KEY)


def generate_text(prompt: str) -> str:
    """Generate marketing text using Gemini 2.0 Flash"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip() if response.text else "Could not generate content."
    except Exception as e:
        print(f"\nTEXT GEN ERROR:\n{traceback.format_exc()}\n")
        raise Exception(f"Text generation failed: {str(e)}")


def generate_chat_response(messages: list) -> str:
    """Multi-turn chat using Gemini 2.0 Flash"""
    try:
        gemini_contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_contents
        )
        return response.text.strip() if response.text else "Could not generate response."
    except Exception as e:
        print(f"\nCHAT ERROR:\n{traceback.format_exc()}\n")
        raise Exception(f"Chat response failed: {str(e)}")