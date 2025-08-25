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


def analyze_dating_ui_with_gemini(image_path: str, gemini_api_key: str = None) -> dict:
    """
    Use Gemini to analyze the dating app UI and determine what actions are available.
    
    Returns:
        Dictionary with UI analysis including like button location, profile content, etc.
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
        Analyze this dating app screenshot and provide a comprehensive UI analysis in JSON format:
        
        {
            "has_like_button": true/false,
            "like_button_visible": true/false,
            "profile_quality_score": 1-10,
            "should_like": true/false,
            "reason": "detailed reason for recommendation",
            "ui_elements_detected": ["list", "of", "visible", "elements"],
            "profile_attractiveness": 1-10,
            "text_content_quality": 1-10,
            "conversation_potential": 1-10,
            "red_flags": ["any", "concerning", "elements"],
            "positive_indicators": ["good", "signs", "to", "like"]
        }
        
        Base your recommendation on:
        - Profile photo quality and attractiveness
        - Bio text content (if visible)
        - Overall profile completeness
        - Any red flags or positive indicators
        - Whether this seems like a good potential match
        
        Be honest in your assessment.
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        return json.loads(response.text) if response.text else {}
        
    except Exception as e:
        print(f"Error analyzing UI with Gemini API: {e}")
        return {
            "has_like_button": False,
            "should_like": False,
            "reason": "Analysis failed",
            "profile_quality_score": 5
        }


def find_ui_elements_with_gemini(image_path: str, element_type: str = "like_button", gemini_api_key: str = None) -> dict:
    """
    Use Gemini to find UI elements and their approximate locations.
    
    Args:
        image_path: Path to screenshot
        element_type: Type of element to find ("like_button", "dislike_button", etc.)
        gemini_api_key: API key
    
    Returns:
        Dictionary with element location info
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        prompt = f"""
        Analyze this dating app screenshot and find the {element_type}.
        
        Look carefully for:
        - Like button: Heart icon, usually green/pink, often at bottom right area (around 70-90% from left, 80-95% from top)
        - Dislike button: X icon or cross, usually red, often at bottom left area (around 10-30% from left, 80-95% from top)
        - Scroll area: The main profile content area that can be scrolled (usually center 20-80% of screen)
        
        Provide precise location in JSON format:
        {{
            "element_found": true/false,
            "approximate_x_percent": 0.0-1.0,
            "approximate_y_percent": 0.0-1.0,
            "confidence": 0.0-1.0,
            "description": "detailed description of what you see",
            "visual_context": "describe surrounding elements",
            "tap_area_size": "small/medium/large"
        }}
        
        Be very precise with coordinates. Dating app buttons are usually in consistent locations.
        Express coordinates as percentages where 0.0 = left/top edge, 1.0 = right/bottom edge.
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        return json.loads(response.text) if response.text else {"element_found": False}
        
    except Exception as e:
        print(f"Error finding UI elements with Gemini: {e}")
        return {"element_found": False}


def analyze_profile_scroll_content(image_path: str, gemini_api_key: str = None) -> dict:
    """
    Analyze if there's more content to scroll through on a profile.
    
    Returns:
        Dictionary with scroll analysis
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        prompt = """
        Analyze this dating profile screenshot to determine scrolling needs:
        
        {
            "has_more_content": true/false,
            "scroll_direction": "up/down/none",
            "content_completion": 0.0-1.0,
            "visible_profile_elements": ["photos", "bio", "prompts", "interests"],
            "should_scroll_down": true/false,
            "scroll_area_center_x": 0.0-1.0,
            "scroll_area_center_y": 0.0-1.0,
            "analysis": "description of what's visible and what might be below",
            "scroll_confidence": 0.0-1.0,
            "estimated_content_below": "description of likely content below"
        }
        
        Look carefully for:
        - Text that appears cut off at the bottom edge
        - Photos that are partially visible
        - Section headers followed by minimal content
        - Prompts or questions with incomplete answers
        - Bio text that seems to continue beyond visible area
        - Any visual indicators of more content (scroll bars, etc.)
        
        Only suggest scrolling down if you're confident there's meaningful content below.
        Be conservative - don't suggest scrolling if the profile appears complete.
        
        The scroll area should be in the center of the profile content, avoiding buttons at bottom.
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        return json.loads(response.text) if response.text else {"has_more_content": False}
        
    except Exception as e:
        print(f"Error analyzing scroll content: {e}")
        return {"has_more_content": False}


def get_profile_navigation_strategy(image_path: str, gemini_api_key: str = None) -> dict:
    """
    Determine the best navigation strategy to avoid getting stuck.
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        prompt = """
        Analyze this dating app screen to determine navigation strategy:
        
        {
            "screen_type": "profile/card_stack/other",
            "stuck_indicator": true/false,
            "navigation_action": "swipe_left/swipe_right/scroll_down/tap_next/go_back",
            "swipe_direction": "left/right/up/down",
            "swipe_start_x": 0.0-1.0,
            "swipe_start_y": 0.0-1.0,
            "swipe_end_x": 0.0-1.0,
            "swipe_end_y": 0.0-1.0,
            "confidence": 0.0-1.0,
            "reason": "why this navigation is recommended"
        }
        
        Identify if this looks like:
        - A profile view (detailed profile page) - needs swipe or back button
        - Card stack view (swipeable profiles) - needs horizontal swipes
        - Error/stuck state - needs different navigation
        
        For getting unstuck, recommend larger swipe distances and different directions.
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        return json.loads(response.text) if response.text else {"navigation_action": "swipe_left"}
        
    except Exception as e:
        print(f"Error getting navigation strategy: {e}")
        return {"navigation_action": "swipe_left", "reason": "fallback"}


def detect_comment_ui_elements(image_path: str, gemini_api_key: str = None) -> dict:
    """
    Detect comment interface elements like text field and send button.
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        prompt = """
        Analyze this dating app comment interface screenshot and find UI elements:
        
        {
            "comment_field_found": true/false,
            "comment_field_x": 0.0-1.0,
            "comment_field_y": 0.0-1.0,
            "comment_field_confidence": 0.0-1.0,
            "send_button_found": true/false,
            "send_button_x": 0.0-1.0,
            "send_button_y": 0.0-1.0,
            "send_button_confidence": 0.0-1.0,
            "cancel_button_found": true/false,
            "cancel_button_x": 0.0-1.0,
            "cancel_button_y": 0.0-1.0,
            "interface_state": "comment_ready/sending/error/unknown",
            "description": "what you see in the interface"
        }
        
        Look for:
        - Comment text field (might say "Add a comment" or be an empty text input)
        - Send button (might say "Send Like", "Send", or have an arrow icon)
        - Cancel button (usually says "Cancel" or has an X)
        
        Focus on elements in the bottom half of the screen.
        Express coordinates as percentages (0.0 = left/top, 1.0 = right/bottom).
        """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        return json.loads(response.text) if response.text else {}
        
    except Exception as e:
        print(f"Error detecting comment UI elements: {e}")
        return {"comment_field_found": False, "send_button_found": False}


def verify_action_success(image_path: str, action_type: str, gemini_api_key: str = None) -> dict:
    """
    Verify if a specific action (like, comment, etc.) was successful.
    
    Args:
        image_path: Path to screenshot after action
        action_type: "like_tap", "comment_sent", "profile_change"
        gemini_api_key: API key
    
    Returns:
        Dictionary with verification results
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/png'
        )
        
        if action_type == "like_tap":
            prompt = """
            Analyze this dating app screenshot to verify if a LIKE action was successful:
            
            {
                "like_successful": true/false,
                "interface_state": "comment_modal/main_profile/next_profile/error",
                "visible_indicators": ["like_confirmation", "comment_interface", "match_notification"],
                "next_action_available": true/false,
                "confidence": 0.0-1.0,
                "description": "what you see that indicates like success or failure"
            }
            
            Look for indicators of successful like:
            - Comment interface appeared (means like worked)
            - Match notification/celebration screen
            - Profile changed or advanced
            - Like button disappeared or changed state
            
            Signs of failure:
            - Still see the same like button in same position
            - Error message
            - Interface unchanged
            """
            
        elif action_type == "comment_sent":
            prompt = """
            Analyze this screenshot to verify if a COMMENT was successfully sent:
            
            {
                "comment_sent": true/false,
                "interface_state": "back_to_profile/match_screen/conversation_started/error",
                "visible_indicators": ["match_notification", "conversation_preview", "success_message"],
                "comment_interface_gone": true/false,
                "confidence": 0.0-1.0,
                "description": "what indicates comment was sent successfully"
            }
            
            Look for successful comment indicators:
            - Comment interface disappeared
            - Match notification appeared
            - Conversation/chat interface visible
            - Success confirmation message
            - Profile advanced to next person
            
            Signs of failure:
            - Still in comment interface
            - Error message
            - Send button still visible and active
            """
            
        elif action_type == "profile_change":
            prompt = """
            Analyze this screenshot to verify if we successfully moved to a NEW profile:
            
            {
                "profile_changed": true/false,
                "interface_state": "new_profile/same_profile/loading/error",
                "profile_elements_visible": ["new_photos", "new_name", "new_bio"],
                "stuck_indicator": true/false,
                "confidence": 0.0-1.0,
                "description": "evidence of profile change or staying on same profile"
            }
            
            Look for profile change indicators:
            - Different person's photos
            - Different name visible
            - Different bio/text content
            - New profile layout
            
            Signs we're stuck:
            - Same person's photos
            - Identical interface
            - Same name/age
            - No visual changes
            """
        
        else:
            # Generic verification
            prompt = f"""
            Analyze this screenshot for general action verification of type: {action_type}
            
            {{
                "action_successful": true/false,
                "interface_state": "unknown",
                "confidence": 0.0-1.0,
                "description": "general analysis of interface state"
            }}
            """
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=config
        )
        
        result = json.loads(response.text) if response.text else {}
        result['verification_type'] = action_type
        return result
        
    except Exception as e:
        print(f"Error verifying action {action_type}: {e}")
        return {
            "verification_type": action_type,
            "action_successful": False,
            "confidence": 0.0,
            "description": f"Verification failed: {e}"
        }