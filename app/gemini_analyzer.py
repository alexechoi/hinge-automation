# app/gemini_analyzer.py

import os
from google import genai
from google.genai import types
from PIL import Image
import requests


def extract_text_from_image_gemini(image_path: str, gemini_api_key: str = None) -> str:
    """
    Uses Google's Gemini API to extract and analyze text from dating profile images.
    
    Args:
        image_path: Path to the screenshot image
        gemini_api_key: Google GenAI API key (optional, will use env var if not provided)
    
    Returns:
        Extracted text from the image
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    try:
        # Initialize the client
        client = genai.Client(api_key=gemini_api_key)
        
        # Load and prepare the image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Create the image part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'  # Assuming screenshots are PNG
        )
        
        # Prompt specifically for dating profile text extraction
        prompt = """
        Extract all visible text from this dating profile screenshot. 
        Focus on:
        - Profile bio/description text
        - Name and age information
        - Any prompts and answers
        - Interests or hobbies mentioned
        - Location information if visible
        
        Return only the extracted text content, formatted cleanly without any analysis or commentary.
        """
        
        # Generate content
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part]
        )
        
        return response.text.strip() if response.text else ""
        
    except Exception as e:
        print(f"Error extracting text with Gemini API: {e}")
        return ""


def analyze_profile_with_gemini(image_path: str, gemini_api_key: str = None) -> dict:
    """
    Uses Gemini to extract and analyze a dating profile image comprehensively.
    
    Returns:
        Dictionary containing extracted text, interests, sentiment analysis, etc.
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        prompt = """
        Analyze this dating profile screenshot and provide a comprehensive analysis in JSON format:
        
        {
            "extracted_text": "All visible text from the profile",
            "name_and_age": "Person's name and age if visible",
            "bio_text": "Main bio/description text",
            "interests": ["list", "of", "interests", "mentioned"],
            "prompts_and_answers": ["Any dating app prompts and their answers"],
            "location": "Location if mentioned",
            "sentiment": "positive/neutral/negative overall tone",
            "key_topics": ["main", "conversation", "starters"],
            "personality_traits": ["traits", "that", "stand", "out"]
        }
        
        Be thorough but concise. If information is not available, use empty strings or arrays.
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        import json
        return json.loads(response.text) if response.text else {}
        
    except Exception as e:
        print(f"Error analyzing profile with Gemini API: {e}")
        return {}