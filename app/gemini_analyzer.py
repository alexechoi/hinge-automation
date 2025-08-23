# app/gemini_analyzer.py

import os
from google import genai
from google.genai import types
import json


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


def generate_comment_gemini(profile_text: str, gemini_api_key: str = None) -> str:
    """
    Generate a dating app comment using Gemini instead of OpenAI.
    
    Args:
        profile_text: The extracted text from the dating profile
        gemini_api_key: Google GenAI API key (optional, will use env var if not provided)
    
    Returns:
        Generated comment string
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        prompt = f"""
        Based on the following dating profile description, generate a 1-line friendly and personalized comment asking them to go out with you:

        Profile Description:
        {profile_text}

        Requirements:
        - Be witty and humorous
        - Reference something specific from their profile
        - Keep it under 50 words
        - Make it feel personal, not generic
        - End with a question or suggestion to meet up
        - Be respectful and not overly aggressive

        Generate only the comment text, nothing else:
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        
        return response.text.strip() if response.text else "Hey, I'd love to meet up!"
        
    except Exception as e:
        print(f"Error generating comment with Gemini API: {e}")
        return "Hey, I'd love to meet up!"


def generate_advanced_comment_gemini(profile_text: str, style: str = "balanced", gemini_api_key: str = None) -> str:
    """
    Generate an advanced dating comment with different styles using Gemini.
    
    Args:
        profile_text: The extracted text from the dating profile
        style: "comedic", "flirty", "straightforward", or "balanced"
        gemini_api_key: Google GenAI API key
    
    Returns:
        Generated comment string
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    style_prompts = {
        "comedic": "Write a witty, humorous comment that makes them laugh while referencing their profile.",
        "flirty": "Write a playful, charming comment that's flirty but respectful.",
        "straightforward": "Write a direct, honest comment that shows genuine interest in getting to know them.",
        "balanced": "Write a friendly comment that combines humor with genuine interest."
    }
    
    style_instruction = style_prompts.get(style, style_prompts["balanced"])
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        prompt = f"""
        Based on this dating profile, {style_instruction}

        Profile:
        {profile_text}

        Guidelines:
        - Keep it under 50 words
        - Reference something specific from their profile
        - End with a question or suggestion to connect
        - Be authentic and engaging
        - Avoid generic pickup lines

        Generate only the comment:
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        
        return response.text.strip() if response.text else "Hey, I'd love to meet up!"
        
    except Exception as e:
        print(f"Error generating advanced comment with Gemini API: {e}")
        return "Hey, I'd love to meet up!"