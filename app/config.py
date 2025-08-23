# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Main API key - using Gemini now
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
