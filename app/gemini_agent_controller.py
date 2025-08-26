# app/gemini_agent_controller.py

import json
import time
import uuid
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from helper_functions import (
    connect_device, get_screen_resolution, open_hinge,
    capture_screenshot, tap, tap_with_confidence, swipe,
    input_text, dismiss_keyboard
)
from gemini_analyzer import (
    extract_text_from_image_gemini, analyze_dating_ui_with_gemini,
    find_ui_elements_with_gemini, analyze_profile_scroll_content,
    get_profile_navigation_strategy, detect_comment_ui_elements,
    verify_action_success, generate_comment_gemini
)
from data_store import store_generated_comment, calculate_template_success_rates
from prompt_engine import update_template_weights


class GeminiAgentController:
    """
    Gemini-powered agent controller that analyzes screenshots and intelligently
    selects and executes tools for Hinge automation.
    """
    
    def __init__(self, max_profiles: int = 10, config=None):
        from agent_config import DEFAULT_CONFIG
        
        self.max_profiles = max_profiles
        self.config = config or DEFAULT_CONFIG
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Available tools that Gemini can call
        self.available_tools = {
            "capture_screenshot": self.capture_screenshot_tool,
            "analyze_profile": self.analyze_profile_tool,
            "scroll_profile": self.scroll_profile_tool,
            "detect_like_button": self.detect_like_button_tool,
            "execute_like": self.execute_like_tool,
            "generate_comment": self.generate_comment_tool,
            "handle_comment_interface": self.handle_comment_interface_tool,
            "execute_dislike": self.execute_dislike_tool,
            "navigate_to_next": self.navigate_to_next_tool,
            "recover_from_stuck": self.recover_from_stuck_tool,
            "verify_action": self.verify_action_tool
        }
        
        # Session state
        self.device = None
        self.width = 0
        self.height = 0
        self.current_profile_index = 0
        self.session_data = {
            "profiles_processed": 0,
            "likes_sent": 0,
            "comments_sent": 0,
            "errors_encountered": 0,
            "stuck_count": 0,
            "last_action": "",
            "current_screenshot": "",
            "profile_text": "",
            "profile_analysis": {},
            "decision_reason": ""
        }
    
    def get_tool_schema(self) -> str:
        """Return a detailed schema of all available tools for Gemini to understand."""
        return """
        Available Tools for Hinge Automation:
        
        1. capture_screenshot:
           - Purpose: Take a screenshot of the current screen
           - When to use: At the start of each profile analysis or after any action
           - Returns: Screenshot file path
        
        2. analyze_profile:
           - Purpose: Extract text and analyze profile quality from screenshot
           - When to use: After capturing screenshot of a profile
           - Returns: Profile text, quality score, interests, sentiment
        
        3. scroll_profile:
           - Purpose: Scroll down to see more profile content
           - When to use: When profile seems incomplete or truncated
           - Returns: Success status, additional content found
        
        4. detect_like_button:
           - Purpose: Find the like button location on screen
           - When to use: When decided to like a profile
           - Returns: Button coordinates, confidence level
        
        5. execute_like:
           - Purpose: Tap the like button
           - When to use: After successfully detecting like button
           - Returns: Success status, whether comment interface appeared
        
        6. generate_comment:
           - Purpose: Generate a personalized comment based on profile
           - When to use: When like action opens comment interface
           - Returns: Generated comment text
        
        7. handle_comment_interface:
           - Purpose: Input and send comment through the interface
           - When to use: After generating comment and comment interface is open
           - Returns: Success status of comment sending
        
        8. execute_dislike:
           - Purpose: Dislike/skip current profile
           - When to use: When profile doesn't meet criteria
           - Returns: Success status
        
        9. navigate_to_next:
           - Purpose: Move to the next profile
           - When to use: After processing current profile (like/dislike/comment)
           - Returns: Success status, whether new profile loaded
        
        10. recover_from_stuck:
            - Purpose: Attempt recovery when stuck on same screen using swipe patterns
            - When to use: When navigation fails or same content detected
            - Method: Uses multiple swipe patterns, NO back button or home navigation
            - Returns: Recovery attempt status
        
        11. verify_action:
            - Purpose: Check if previous action was successful
            - When to use: After important actions like like, comment, navigation
            - Returns: Verification results
        
        Tool Selection Guidelines:
        - Always start with capture_screenshot
        - Analyze profiles before making decisions
        - Only like profiles that meet quality criteria
        - Generate comments for liked profiles when interface appears
        - Use recovery tools when stuck
        - Verify important actions before proceeding
        """
    
    def ask_gemini_for_next_action(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask Gemini to analyze the current state and decide the next action.
        
        Returns:
            Dictionary with tool_name, parameters, and reasoning
        """
        
        # Prepare context for Gemini
        context = f"""
        Current Hinge Automation State:
        - Profile Index: {self.current_profile_index}/{self.max_profiles}
        - Profiles Processed: {self.session_data['profiles_processed']}
        - Last Action: {self.session_data['last_action']}
        - Current Screenshot: {self.session_data['current_screenshot']}
        - Profile Text: {self.session_data['profile_text'][:500]}...
        - Stuck Count: {self.session_data['stuck_count']}
        - Errors: {self.session_data['errors_encountered']}
        
        Profile Analysis:
        {json.dumps(self.session_data.get('profile_analysis', {}), indent=2)}
        
        Session Goal: Process {self.max_profiles} dating profiles, making intelligent like/dislike decisions
        and sending personalized comments when appropriate.
        
        {self.get_tool_schema()}
        """
        
        if self.session_data['current_screenshot']:
            # Include the current screenshot for visual analysis
            with open(self.session_data['current_screenshot'], 'rb') as f:
                image_bytes = f.read()
            
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/png'
            )
            
            prompt = f"""
            {context}
            
            Analyze the current screenshot and determine the best next action.
            
            Respond in JSON format:
            {{
                "tool_name": "name_of_tool_to_use",
                "parameters": {{"key": "value"}},
                "reasoning": "detailed explanation of why this tool was chosen",
                "confidence": 0.0-1.0,
                "expected_outcome": "what should happen after this action",
                "fallback_tool": "alternative tool if primary fails"
            }}
            
            Consider:
            - What type of screen is currently displayed?
            - What is the appropriate next step in the dating app workflow?
            - Are there any error conditions or stuck states?
            - Has the session goal been completed?
            
            Be strategic and efficient in tool selection.
            """
            
            config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            contents = [prompt, image_part]
        else:
            # No screenshot available, use text-only prompt
            prompt = f"""
            {context}
            
            Based on the current state (no screenshot available), determine the best next action.
            
            Respond in JSON format with tool selection and reasoning.
            If no screenshot is available, the first action should typically be "capture_screenshot".
            """
            
            config = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
            
            contents = [prompt]
        
        try:
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=config
            )
            
            decision = json.loads(response.text) if response.text else {}
            print(f"🤖 Gemini Decision: {decision.get('tool_name', 'unknown')} - {decision.get('reasoning', 'no reason')}")
            
            return decision
            
        except Exception as e:
            print(f"Error getting Gemini decision: {e}")
            # Fallback decision
            if not self.session_data['current_screenshot']:
                return {
                    "tool_name": "capture_screenshot",
                    "parameters": {},
                    "reasoning": "No screenshot available, need to capture current state",
                    "confidence": 1.0
                }
            else:
                return {
                    "tool_name": "navigate_to_next",
                    "parameters": {},
                    "reasoning": "Fallback action due to Gemini error",
                    "confidence": 0.5
                }
    
    # Tool implementations
    def capture_screenshot_tool(self, **kwargs) -> Dict[str, Any]:
        """Capture current screen screenshot"""
        screenshot_path = capture_screenshot(
            self.device,
            f"profile_{self.current_profile_index}_gemini"
        )
        
        self.session_data['current_screenshot'] = screenshot_path
        self.session_data['last_action'] = "capture_screenshot"
        
        return {
            "success": True,
            "screenshot_path": screenshot_path,
            "message": "Screenshot captured successfully"
        }
    
    def analyze_profile_tool(self, **kwargs) -> Dict[str, Any]:
        """Analyze profile from current screenshot"""
        if not self.session_data['current_screenshot']:
            return {"success": False, "message": "No screenshot available"}
        
        # Extract text and analyze profile
        profile_text = extract_text_from_image_gemini(
            self.session_data['current_screenshot'], GEMINI_API_KEY
        )
        
        ui_analysis = analyze_dating_ui_with_gemini(
            self.session_data['current_screenshot'], GEMINI_API_KEY
        )
        
        self.session_data['profile_text'] = profile_text
        self.session_data['profile_analysis'] = ui_analysis
        self.session_data['last_action'] = "analyze_profile"
        
        return {
            "success": True,
            "profile_text": profile_text,
            "analysis": ui_analysis,
            "should_like": ui_analysis.get('should_like', False),
            "quality_score": ui_analysis.get('profile_quality_score', 0),
            "message": f"Profile analyzed: quality {ui_analysis.get('profile_quality_score', 0)}/10"
        }
    
    def scroll_profile_tool(self, direction: str = "down", **kwargs) -> Dict[str, Any]:
        """Scroll to see more profile content"""
        scroll_analysis = analyze_profile_scroll_content(
            self.session_data['current_screenshot'], GEMINI_API_KEY
        )
        
        if not scroll_analysis.get('should_scroll_down'):
            return {
                "success": False,
                "message": "No additional content to scroll"
            }
        
        # Perform scroll
        scroll_x = int(scroll_analysis.get('scroll_area_center_x', 0.5) * self.width)
        scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * self.height)
        scroll_y_end = int(scroll_y_start * 0.3)
        
        swipe(self.device, scroll_x, scroll_y_start, scroll_x, scroll_y_end)
        time.sleep(2)
        
        # Capture new content
        new_screenshot = capture_screenshot(self.device, f"scrolled_content_{time.time()}")
        additional_text = extract_text_from_image_gemini(new_screenshot, GEMINI_API_KEY)
        
        # Update session data
        if additional_text and additional_text not in self.session_data['profile_text']:
            self.session_data['profile_text'] += "\n" + additional_text
        
        self.session_data['current_screenshot'] = new_screenshot
        self.session_data['last_action'] = "scroll_profile"
        
        return {
            "success": True,
            "additional_content": additional_text,
            "message": "Profile scrolled and content updated"
        }
    
    def detect_like_button_tool(self, **kwargs) -> Dict[str, Any]:
        """Detect like button location"""
        like_button_info = find_ui_elements_with_gemini(
            self.session_data['current_screenshot'], "like_button", GEMINI_API_KEY
        )
        
        if not like_button_info.get('element_found'):
            return {
                "success": False,
                "message": "Like button not detected"
            }
        
        confidence = like_button_info.get('confidence', 0)
        if confidence < self.config.min_button_confidence:
            return {
                "success": False,
                "confidence": confidence,
                "message": f"Like button confidence too low: {confidence}"
            }
        
        like_x = int(like_button_info['approximate_x_percent'] * self.width)
        like_y = int(like_button_info['approximate_y_percent'] * self.height)
        
        self.session_data['like_button_coords'] = (like_x, like_y)
        self.session_data['like_button_confidence'] = confidence
        self.session_data['last_action'] = "detect_like_button"
        
        return {
            "success": True,
            "coordinates": (like_x, like_y),
            "confidence": confidence,
            "message": f"Like button found at ({like_x}, {like_y}) with confidence {confidence}"
        }
    
    def execute_like_tool(self, **kwargs) -> Dict[str, Any]:
        """Execute like action"""
        if 'like_button_coords' not in self.session_data:
            return {
                "success": False,
                "message": "Like button coordinates not available"
            }
        
        like_x, like_y = self.session_data['like_button_coords']
        confidence = self.session_data.get('like_button_confidence', 0.8)
        
        # Tap the like button
        tap_with_confidence(self.device, like_x, like_y, confidence)
        time.sleep(2)
        
        # Verify like action
        verification_screenshot = capture_screenshot(self.device, "like_verification")
        like_verification = verify_action_success(verification_screenshot, "like_tap", GEMINI_API_KEY)
        
        self.session_data['current_screenshot'] = verification_screenshot
        self.session_data['last_action'] = "execute_like"
        self.session_data['likes_sent'] += 1
        
        return {
            "success": like_verification.get('like_successful', False),
            "comment_interface_appeared": like_verification.get('interface_state') == 'comment_modal',
            "verification": like_verification,
            "message": "Like executed" + (" - comment interface opened" if like_verification.get('interface_state') == 'comment_modal' else "")
        }
    
    def generate_comment_tool(self, style: str = "balanced", **kwargs) -> Dict[str, Any]:
        """Generate comment for current profile"""
        if not self.session_data['profile_text']:
            return {
                "success": False,
                "message": "No profile text available for comment generation"
            }
        
        comment = generate_comment_gemini(self.session_data['profile_text'], GEMINI_API_KEY)
        if not comment:
            comment = self.config.default_comment
        
        comment_id = str(uuid.uuid4())
        store_generated_comment(
            comment_id=comment_id,
            profile_text=self.session_data['profile_text'],
            generated_comment=comment,
            style_used=f"gemini_agent_{style}"
        )
        
        self.session_data['generated_comment'] = comment
        self.session_data['comment_id'] = comment_id
        self.session_data['last_action'] = "generate_comment"
        
        return {
            "success": True,
            "comment": comment,
            "comment_id": comment_id,
            "message": f"Comment generated: {comment[:50]}..."
        }
    
    def handle_comment_interface_tool(self, **kwargs) -> Dict[str, Any]:
        """Handle comment interface after like"""
        if 'generated_comment' not in self.session_data:
            return {
                "success": False,
                "message": "No comment generated to send"
            }
        
        comment = self.session_data['generated_comment']
        max_retries = self.config.max_retries_per_action
        
        for attempt in range(max_retries):
            try:
                time.sleep(2)
                
                # Detect comment UI elements
                comment_ui = detect_comment_ui_elements(
                    self.session_data['current_screenshot'], GEMINI_API_KEY
                )
                
                if not comment_ui.get('comment_field_found'):
                    if attempt < max_retries - 1:
                        continue
                    return {
                        "success": False,
                        "message": "Comment field not found after retries"
                    }
                
                # Tap comment field and input text
                comment_x = int(comment_ui['comment_field_x'] * self.width)
                comment_y = int(comment_ui['comment_field_y'] * self.height)
                tap_with_confidence(self.device, comment_x, comment_y,
                                  comment_ui.get('comment_field_confidence', 0.8))
                
                time.sleep(1.5)
                input_text(self.device, comment)
                time.sleep(1)
                
                # Try Enter key first
                self.device.shell("input keyevent KEYCODE_ENTER")
                time.sleep(2)
                
                # Check if Enter worked
                after_enter_screenshot = capture_screenshot(self.device, f"after_enter_{attempt}")
                post_enter_ui = detect_comment_ui_elements(after_enter_screenshot, GEMINI_API_KEY)
                
                if not post_enter_ui.get('comment_field_found'):
                    self.session_data['current_screenshot'] = after_enter_screenshot
                    self.session_data['comments_sent'] += 1
                    return {
                        "success": True,
                        "message": "Comment sent via Enter key"
                    }
                
                # Need to find and tap send button
                dismiss_keyboard(self.device, self.width, self.height)
                time.sleep(2)
                
                send_screenshot = capture_screenshot(self.device, f"send_interface_{attempt}")
                send_ui = detect_comment_ui_elements(send_screenshot, GEMINI_API_KEY)
                
                if send_ui.get('send_button_found'):
                    send_x = int(send_ui['send_button_x'] * self.width)
                    send_y = int(send_ui['send_button_y'] * self.height)
                else:
                    # Fallback coordinates
                    send_x = int(self.width * 0.75)
                    send_y = int(self.height * 0.82)
                
                tap_with_confidence(self.device, send_x, send_y, 0.8)
                time.sleep(3)
                
                # Verify comment sent
                final_screenshot = capture_screenshot(self.device, f"comment_verification_{attempt}")
                verification = verify_action_success(final_screenshot, "comment_sent", GEMINI_API_KEY)
                
                self.session_data['current_screenshot'] = final_screenshot
                self.session_data['last_action'] = "handle_comment_interface"
                self.session_data['comments_sent'] += 1
                
                return {
                    "success": verification.get('comment_sent', True),
                    "verification": verification,
                    "message": "Comment interface handled"
                }
                
            except Exception as e:
                print(f"Comment attempt {attempt + 1} failed: {e}")
                if attempt >= max_retries - 1:
                    self.session_data['errors_encountered'] += 1
                    return {
                        "success": False,
                        "message": f"Comment failed after {max_retries} attempts: {e}"
                    }
                continue
        
        return {
            "success": False,
            "message": "Comment interface handling failed"
        }
    
    def execute_dislike_tool(self, **kwargs) -> Dict[str, Any]:
        """Execute dislike action"""
        x_dislike = int(self.width * self.config.dislike_button_coords[0])
        y_dislike = int(self.height * self.config.dislike_button_coords[1])
        
        tap(self.device, x_dislike, y_dislike)
        time.sleep(2)
        
        self.session_data['last_action'] = "execute_dislike"
        
        return {
            "success": True,
            "message": f"Profile disliked: {self.session_data.get('decision_reason', 'criteria not met')}"
        }
    
    def navigate_to_next_tool(self, **kwargs) -> Dict[str, Any]:
        """Navigate to next profile"""
        x1_swipe = int(self.width * 0.15)
        y1_swipe = int(self.height * 0.5)
        x2_swipe = x1_swipe
        y2_swipe = int(y1_swipe * 0.75)
        
        swipe(self.device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        time.sleep(3)
        
        # Capture new state
        nav_screenshot = capture_screenshot(self.device, "navigation_result")
        verification = verify_action_success(nav_screenshot, "profile_change", GEMINI_API_KEY)
        
        self.session_data['current_screenshot'] = nav_screenshot
        self.session_data['last_action'] = "navigate_to_next"
        
        if verification.get('profile_changed', True):
            self.current_profile_index += 1
            self.session_data['profiles_processed'] += 1
            self.session_data['stuck_count'] = 0
            return {
                "success": True,
                "new_profile_loaded": True,
                "message": f"Navigated to profile {self.current_profile_index + 1}"
            }
        else:
            self.session_data['stuck_count'] += 1
            return {
                "success": False,
                "new_profile_loaded": False,
                "message": "Navigation failed - still on same profile"
            }
    
    def recover_from_stuck_tool(self, **kwargs) -> Dict[str, Any]:
        """Attempt recovery when stuck"""
        print("🔄 Attempting recovery from stuck state...")
        
        # Try multiple swipe patterns to get unstuck
        recovery_attempts = [
            # Try aggressive horizontal swipe (left to right)
            (int(self.width * 0.9), int(self.height * 0.5), int(self.width * 0.1), int(self.height * 0.5)),
            # Try vertical swipe down
            (int(self.width * 0.5), int(self.height * 0.3), int(self.width * 0.5), int(self.height * 0.7)),
            # Try diagonal swipe
            (int(self.width * 0.8), int(self.height * 0.3), int(self.width * 0.2), int(self.height * 0.7)),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(recovery_attempts):
            print(f"🔄 Recovery attempt {i + 1}: Swipe from ({x1}, {y1}) to ({x2}, {y2})")
            swipe(self.device, x1, y1, x2, y2, duration=800)
            time.sleep(2)
            
            # Check if we're still stuck after each attempt
            recovery_screenshot = capture_screenshot(self.device, f"recovery_attempt_{i}")
            current_text = extract_text_from_image_gemini(recovery_screenshot, GEMINI_API_KEY)
            
            # If we got different content, we might be unstuck
            if current_text != self.session_data.get('profile_text', ''):
                print(f"✅ Recovery successful on attempt {i + 1}")
                break
        
        # Capture final result
        final_screenshot = capture_screenshot(self.device, "recovery_result")
        self.session_data['current_screenshot'] = final_screenshot
        self.session_data['last_action'] = "recover_from_stuck"
        
        return {
            "success": True,
            "message": "Recovery attempt completed - used swipe patterns only"
        }
    
    def verify_action_tool(self, action_type: str = "general", **kwargs) -> Dict[str, Any]:
        """Verify if previous action was successful"""
        if not self.session_data['current_screenshot']:
            return {
                "success": False,
                "message": "No screenshot available for verification"
            }
        
        verification = verify_action_success(
            self.session_data['current_screenshot'], action_type, GEMINI_API_KEY
        )
        
        self.session_data['last_action'] = "verify_action"
        
        return {
            "success": True,
            "verification": verification,
            "action_successful": verification.get('action_successful', False),
            "message": f"Action verification completed for {action_type}"
        }
    
    def initialize_session(self) -> bool:
        """Initialize the automation session"""
        print("🚀 Initializing Gemini-controlled Hinge automation session...")
        
        self.device = connect_device(self.config.device_ip)
        if not self.device:
            print("❌ Failed to connect to device")
            return False
        
        self.width, self.height = get_screen_resolution(self.device)
        open_hinge(self.device)
        time.sleep(5)
        
        # Update template weights with current success rates
        success_rates = calculate_template_success_rates()
        update_template_weights(success_rates)
        
        print(f"✅ Session initialized - Device: {self.device.serial}, Resolution: {self.width}x{self.height}")
        return True
    
    def run_automation(self) -> Dict[str, Any]:
        """
        Run the complete Gemini-controlled automation workflow
        """
        print("🤖 Starting Gemini-controlled Hinge automation...")
        
        if not self.initialize_session():
            return {
                "success": False,
                "error": "Failed to initialize session",
                "profiles_processed": 0
            }
        
        # Main automation loop
        while self.current_profile_index < self.max_profiles:
            try:
                print(f"\n📱 Processing profile {self.current_profile_index + 1}/{self.max_profiles}")
                
                # Ask Gemini what to do next
                current_state = {
                    "profile_index": self.current_profile_index,
                    "session_data": self.session_data,
                    "device_info": {"width": self.width, "height": self.height}
                }
                
                gemini_decision = self.ask_gemini_for_next_action(current_state)
                tool_name = gemini_decision.get('tool_name')
                parameters = gemini_decision.get('parameters', {})
                
                print(f"🎯 Gemini chose tool: {tool_name}")
                print(f"💭 Reasoning: {gemini_decision.get('reasoning', 'N/A')}")
                
                # Execute the tool Gemini selected
                if tool_name in self.available_tools:
                    tool_function = self.available_tools[tool_name]
                    result = tool_function(**parameters)
                    
                    print(f"🔧 Tool result: {result.get('message', 'No message')}")
                    
                    # Handle special cases
                    if not result.get('success') and gemini_decision.get('fallback_tool'):
                        fallback_tool = gemini_decision['fallback_tool']
                        if fallback_tool in self.available_tools:
                            print(f"🔄 Using fallback tool: {fallback_tool}")
                            fallback_result = self.available_tools[fallback_tool]()
                            result = fallback_result
                    
                    # Check for stuck state
                    if self.session_data['stuck_count'] > self.config.max_stuck_count:
                        print("🚨 Too many stuck attempts, trying recovery...")
                        self.recover_from_stuck_tool()
                        self.session_data['stuck_count'] = 0
                    
                    # Check for too many errors
                    if self.session_data['errors_encountered'] > self.config.max_errors_before_abort:
                        print(f"🚨 Too many errors ({self.session_data['errors_encountered']}), aborting session")
                        break
                        
                else:
                    print(f"❌ Unknown tool: {tool_name}")
                    self.session_data['errors_encountered'] += 1
                    
                # Small delay between actions
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n⚠️ Automation interrupted by user")
                break
                
            except Exception as e:
                print(f"❌ Error in automation loop: {e}")
                self.session_data['errors_encountered'] += 1
                
                if self.session_data['errors_encountered'] > self.config.max_errors_before_abort:
                    print("🚨 Too many errors, stopping automation")
                    break
                
                # Try recovery
                self.recover_from_stuck_tool()
        
        # Update final success rates
        final_success_rates = calculate_template_success_rates()
        update_template_weights(final_success_rates)
        
        return {
            "success": True,
            "profiles_processed": self.session_data['profiles_processed'],
            "likes_sent": self.session_data['likes_sent'],
            "comments_sent": self.session_data['comments_sent'],
            "errors_encountered": self.session_data['errors_encountered'],
            "completion_reason": "Session completed",
            "final_success_rates": final_success_rates
        }
