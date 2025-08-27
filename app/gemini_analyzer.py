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



def generate_comment_gemini(profile_text: str, gemini_api_key: str = None) -> str:
    """
    Generate a flirty, witty dating app comment focused on getting a date.
    
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
        Based on this dating profile, generate a FLIRTY, WITTY comment that's designed to get a date.

        Profile Content:
        {profile_text}

        STYLE REQUIREMENTS:
        - Be confident and playfully flirty (not aggressive or creepy)
        - Use clever wordplay, puns, or witty observations
        - Reference something specific from their profile to show you actually read it
        - Create intrigue and make them want to respond
        - Suggest meeting up in a clever/indirect way
        - Sound like you're genuinely interested in them as a person
        - Keep it under 40 words for maximum impact

        TONE EXAMPLES:
        - Playful teasing about something they mentioned
        - Clever callbacks to their interests/hobbies
        - Confident but not arrogant
        - Fun and lighthearted
        - Slightly challenging or intriguing

        AVOID:
        - Generic compliments about looks
        - Boring "hey how are you" openers  
        - Overly sexual or inappropriate content
        - Trying too hard to be funny
        - Being too serious or formal

        GOAL: Make them think "this person seems fun and interesting, I want to know more"

        Generate ONE flirty, witty comment that will get them excited to meet up:
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        
        comment = response.text.strip() if response.text else ""
        
        # Clean up the comment (remove quotes if present)
        comment = comment.strip('"\'')
        
        # Fallback if generation fails or is too generic
        if not comment or len(comment) < 10 or "hey" in comment.lower()[:10]:
            return _generate_fallback_flirty_comment(profile_text)
        
        return comment
        
    except Exception as e:
        print(f"Error generating comment with Gemini API: {e}")
        return _generate_fallback_flirty_comment(profile_text)


def _generate_fallback_flirty_comment(profile_text: str) -> str:
    """
    Generate fallback flirty comments when main generation fails
    """
    import random
    
    # Try to match fallback to profile content
    profile_lower = profile_text.lower()
    
    if any(word in profile_lower for word in ['coffee', 'caffeine', 'espresso', 'latte']):
        return "I have a theory that our first coffee date is going to turn into an all-day adventure"
    
    if any(word in profile_lower for word in ['travel', 'adventure', 'explore', 'wanderlust']):
        return "Your wanderlust is showing - want to explore the city together?"
    
    if any(word in profile_lower for word in ['food', 'foodie', 'cooking', 'restaurant', 'pizza']):
        return "I'm getting serious 'let's debate the best restaurants over dinner' energy from you"
    
    if any(word in profile_lower for word in ['music', 'concert', 'festival', 'band']):
        return "Plot twist: what if our music taste is as compatible as I think? Testing required"
    
    if any(word in profile_lower for word in ['workout', 'gym', 'fitness', 'yoga', 'hike']):
        return "Challenge accepted - but first, let's grab drinks and see if you're as competitive as me"
    
    # Generic flirty fallbacks
    flirty_fallbacks = [
        "I'm getting major 'let's grab drinks and see if you're as interesting in person' vibes",
        "Your profile just convinced me we need to test our compatibility over coffee",
        "I have a theory that we'd have amazing chemistry - care to help me test it?",
        "Warning: I'm about to suggest we skip the small talk and go straight to an adventure",
        "Plot twist: what if we actually met up instead of just matching? Wild concept, I know",
        "I'm calling it - we're going to have one of those 'can't believe we met on an app' stories",
        "Fair warning: I'm really good at first dates. Want to find out?",
        "I have a feeling you're trouble in the best way possible. Prove me right?",
        "Your profile is giving me 'let's skip to the fun part' energy - drinks this week?",
        "I'm convinced we're going to have one of those conversations that goes until 3am",
        "Something tells me you'd be dangerous to take on a date - I'm intrigued",
        "Your vibe is immaculate - when can we test the in-person chemistry?"
    ]
    
    return random.choice(flirty_fallbacks)


def generate_contextual_date_comment(profile_analysis: dict, profile_text: str, gemini_api_key: str = None) -> str:
    """
    Generate highly contextual, flirty comments based on detailed profile analysis
    """
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    try:
        client = genai.Client(api_key=gemini_api_key)
        
        interests = profile_analysis.get('interests', [])
        personality_traits = profile_analysis.get('personality_traits', [])
        profession = profile_analysis.get('profession', '')
        location = profile_analysis.get('location', '')
        
        context_info = f"""
        PROFILE ANALYSIS:
        - Interests: {', '.join(interests[:5])}
        - Personality: {', '.join(personality_traits[:3])}
        - Profession: {profession}
        - Location: {location}
        
        FULL PROFILE TEXT:
        {profile_text[:500]}...
        """
        
        prompt = f"""
        Create an IRRESISTIBLE, flirty comment that will make them want to meet up ASAP.
        
        {context_info}
        
        ADVANCED REQUIREMENTS:
        - Use their specific interests/job/personality to create a unique opener
        - Be confident and slightly cocky (but charming)
        - Create instant chemistry and intrigue
        - Suggest a specific type of date that matches their interests
        - Make them feel like you "get" them
        - Use humor, wit, or clever observations
        - Maximum 35 words for punch and impact
        
        FLIRTY COMMENT FORMULAS (pick one style):
        1. "Your [specific interest] obsession + my [related skill/interest] = [fun date idea]. When are we testing this theory? 😏"
        2. "I see you're into [interest]. Coincidence: I know the best [related place/activity] in town. Suspicious? 🤔"
        3. "Your [personality trait] energy is dangerous - exactly my type. [Date suggestion] this week? 😈"
        4. "Plot twist: someone who [references their content] definitely needs to meet someone who [your implied trait] 🎭"
        5. "[Witty observation about their profile] - clearly we need to continue this conversation over [relevant activity] ✨"
        
        Generate ONE comment that's impossible to ignore:
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        
        comment = response.text.strip().strip('"\'') if response.text else ""
        
        if not comment or len(comment) < 15:
            return generate_comment_gemini(profile_text, gemini_api_key)
        
        return comment
        
    except Exception as e:
        print(f"Error generating contextual comment: {e}")
        return generate_comment_gemini(profile_text, gemini_api_key)




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
        - Like button: Heart icon, usually with a hollow interior, always on the right hand side of the screen often on one of the elements
        - Dislike button: X icon or cross, often at bottom left area (around 10-30% from left, 80-95% from top)
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
        
        Be very precise with coordinates..
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