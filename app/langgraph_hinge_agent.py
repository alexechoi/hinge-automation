# app/langgraph_hinge_agent.py

"""
LangGraph-powered Hinge automation agent that replaces GeminiAgentController.
Uses state-based workflow management for improved reliability and debugging.
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from helper_functions import (
    connect_device, get_screen_resolution, open_hinge,
    capture_screenshot, tap, tap_with_confidence, swipe,
    dismiss_keyboard, clear_screenshots_directory
)
from gemini_analyzer import (
    extract_text_from_image_gemini, analyze_dating_ui_with_gemini,
    find_ui_elements_with_gemini, analyze_profile_scroll_content,
    detect_comment_ui_elements, generate_comment_gemini
)
from data_store import store_generated_comment, calculate_template_success_rates
from prompt_engine import update_template_weights


class HingeAgentState(TypedDict):
    """State maintained throughout the dating app automation workflow"""
    
    # Device and session info
    device: Any
    width: int
    height: int
    max_profiles: int
    current_profile_index: int
    
    # Session metrics
    profiles_processed: int
    likes_sent: int
    comments_sent: int
    errors_encountered: int
    stuck_count: int
    
    # Current profile data
    current_screenshot: Optional[str]
    profile_text: str
    profile_analysis: Dict[str, Any]
    decision_reason: str
    
    # Profile change detection data
    previous_profile_text: str
    previous_profile_features: Dict[str, Any]
    
    # Action results
    last_action: str
    action_successful: bool
    retry_count: int
    
    # Generated content
    generated_comment: str
    comment_id: str
    
    # Button coordinates
    like_button_coords: Optional[tuple]
    like_button_confidence: float
    
    # Control flow
    should_continue: bool
    completion_reason: str
    
    # Gemini decision context
    gemini_reasoning: str
    next_tool_suggestion: str


class LangGraphHingeAgent:
    """
    LangGraph-powered Hinge automation agent with Gemini-controlled decision making.
    Replaces GeminiAgentController with improved workflow management.
    """
    
    def __init__(self, max_profiles: int = 10, config=None):
        from agent_config import DEFAULT_CONFIG
        
        self.max_profiles = max_profiles
        self.config = config or DEFAULT_CONFIG
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        self.graph = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with Gemini-controlled decision making"""
        
        workflow = StateGraph(HingeAgentState)
        
        # Add all workflow nodes
        workflow.add_node("initialize_session", self.initialize_session_node)
        workflow.add_node("gemini_decide_action", self.gemini_decide_action_node)
        workflow.add_node("capture_screenshot", self.capture_screenshot_node)
        workflow.add_node("analyze_profile", self.analyze_profile_node)
        workflow.add_node("scroll_profile", self.scroll_profile_node)
        workflow.add_node("make_like_decision", self.make_like_decision_node)
        workflow.add_node("detect_like_button", self.detect_like_button_node)
        workflow.add_node("execute_like", self.execute_like_node)
        workflow.add_node("generate_comment", self.generate_comment_node)
        workflow.add_node("handle_comment_interface", self.handle_comment_interface_node)
        workflow.add_node("execute_dislike", self.execute_dislike_node)
        workflow.add_node("navigate_to_next", self.navigate_to_next_node)
        workflow.add_node("verify_profile_change", self.verify_profile_change_node)
        workflow.add_node("recover_from_stuck", self.recover_from_stuck_node)
        workflow.add_node("finalize_session", self.finalize_session_node)
        
        # Set entry point
        workflow.set_entry_point("initialize_session")
        
        # Add edges with conditional routing
        workflow.add_conditional_edges(
            "initialize_session",
            self._route_initialization,
            {
                "success": "gemini_decide_action",
                "failure": "finalize_session"
            }
        )
        
        workflow.add_conditional_edges(
            "gemini_decide_action", 
            self._route_gemini_decision,
            {
                "capture_screenshot": "capture_screenshot",
                "analyze_profile": "analyze_profile",
                "scroll_profile": "scroll_profile",
                "make_like_decision": "make_like_decision",
                "detect_like_button": "detect_like_button",
                "execute_like": "execute_like",
                "generate_comment": "generate_comment",
                "handle_comment_interface": "handle_comment_interface",
                "execute_dislike": "execute_dislike",
                "navigate_to_next": "navigate_to_next",
                "verify_profile_change": "verify_profile_change",
                "recover_from_stuck": "recover_from_stuck",
                "finalize": "finalize_session"
            }
        )
        
        # Add edges back to Gemini decision node from all action nodes
        action_nodes = [
            "capture_screenshot", "analyze_profile", "scroll_profile", "make_like_decision",
            "detect_like_button", "execute_like", "generate_comment", "handle_comment_interface",
            "execute_dislike", "navigate_to_next", "verify_profile_change", "recover_from_stuck"
        ]
        
        for node in action_nodes:
            workflow.add_conditional_edges(
                node,
                self._route_action_result,
                {
                    "continue": "gemini_decide_action",
                    "finalize": "finalize_session"
                }
            )
        
        workflow.add_edge("finalize_session", END)
        
        return workflow.compile()
    
    # Routing functions
    def _route_initialization(self, state: HingeAgentState) -> str:
        return "success" if state.get("should_continue", False) else "failure"
    
    def _route_gemini_decision(self, state: HingeAgentState) -> str:
        return state.get("next_tool_suggestion", "finalize")
    
    def _route_action_result(self, state: HingeAgentState) -> str:
        # Check completion conditions
        if (state["current_profile_index"] >= state["max_profiles"] or
            state["errors_encountered"] > self.config.max_errors_before_abort or
            not state.get("should_continue", True)):
            return "finalize"
        return "continue"
    
    # Node implementations
    def initialize_session_node(self, state: HingeAgentState) -> HingeAgentState:
        """Initialize the automation session"""
        print("🚀 Initializing LangGraph Hinge automation session...")
        
        # Clear old screenshots to prevent confusion
        clear_screenshots_directory()
        
        device = connect_device(self.config.device_ip)
        if not device:
            return {
                **state,
                "should_continue": False,
                "completion_reason": "Failed to connect to device",
                "last_action": "initialize_session",
                "action_successful": False
            }
        
        width, height = get_screen_resolution(device)
        open_hinge(device)
        time.sleep(5)
        
        # Update template weights
        success_rates = calculate_template_success_rates()
        update_template_weights(success_rates)
        
        print(f"✅ Session initialized - Device: {device.serial}, Resolution: {width}x{height}")
        
        return {
            **state,
            "device": device,
            "width": width,
            "height": height,
            "max_profiles": self.max_profiles,
            "current_profile_index": 0,
            "profiles_processed": 0,
            "likes_sent": 0,
            "comments_sent": 0,
            "errors_encountered": 0,
            "stuck_count": 0,
            "profile_text": "",
            "profile_analysis": {},
            "decision_reason": "",
            "previous_profile_text": "",
            "previous_profile_features": {},
            "last_action": "initialize_session",
            "action_successful": True,
            "retry_count": 0,
            "generated_comment": "",
            "comment_id": "",
            "like_button_coords": None,
            "like_button_confidence": 0.0,
            "should_continue": True,
            "completion_reason": "",
            "gemini_reasoning": "",
            "next_tool_suggestion": "capture_screenshot",
            "current_screenshot": None
        }
    
    def gemini_decide_action_node(self, state: HingeAgentState) -> HingeAgentState:
        """Ask Gemini to analyze current state and decide next action"""
        print(f"🤖 Asking Gemini for next action (Profile {state['current_profile_index'] + 1}/{state['max_profiles']})")
        
        # Prepare context for Gemini
        context = f"""
        Current Hinge Automation State:
        - Profile Index: {state['current_profile_index']}/{state['max_profiles']}
        - Profiles Processed: {state['profiles_processed']}
        - Last Action: {state['last_action']}
        - Action Successful: {state['action_successful']}
        - Current Screenshot: {state['current_screenshot']}
        - Profile Text: {state['profile_text'][:300]}...
        - Stuck Count: {state['stuck_count']}
        - Errors: {state['errors_encountered']}
        
        Profile Analysis:
        {json.dumps(state.get('profile_analysis', {}), indent=2)[:500]}
        
        Available Actions:
        1. capture_screenshot - Take screenshot of current screen
        2. analyze_profile - Extract text and analyze profile quality
        3. scroll_profile - Scroll to see more profile content
        4. make_like_decision - Decide whether to like or dislike profile
        5. detect_like_button - Find like button coordinates
        6. execute_like - Tap the like button
        7. generate_comment - Create personalized comment
        8. handle_comment_interface - Send comment through interface
        9. execute_dislike - Dislike/skip current profile
        10. navigate_to_next - Move to next profile
        11. verify_profile_change - Check if we moved to new profile
        12. recover_from_stuck - Attempt recovery when stuck
        13. finalize - End the session
        
        Workflow Guidelines:
        - Always start with capture_screenshot if no current screenshot
        - Analyze profiles before making decisions
        - Only like profiles that meet quality criteria
        - Generate comments for liked profiles when interface appears
        - Use recovery when stuck count > 2
        - Finalize when max profiles reached or too many errors
        """
        
        try:
            if state['current_screenshot']:
                # Include screenshot for visual analysis
                with open(state['current_screenshot'], 'rb') as f:
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
                    "next_action": "action_name",
                    "reasoning": "detailed explanation of why this action was chosen",
                    "confidence": 0.0-1.0,
                    "expected_outcome": "what should happen after this action"
                }}
                
                Consider:
                - What type of screen is currently displayed?
                - What is the appropriate next step in the workflow?
                - Are there any error conditions or stuck states?
                - Has the session goal been completed?
                """
                
                config = types.GenerateContentConfig(response_mime_type="application/json")
                contents = [prompt, image_part]
            else:
                # No screenshot available
                prompt = f"""
                {context}
                
                No screenshot is available. Determine the best next action.
                Usually this should be "capture_screenshot" to see the current state.
                
                Respond in JSON format with next_action and reasoning.
                """
                
                config = types.GenerateContentConfig(response_mime_type="application/json")
                contents = [prompt]
            
            response = self.gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents,
                config=config
            )
            
            decision = json.loads(response.text) if response.text else {}
            next_action = decision.get('next_action', 'capture_screenshot')
            reasoning = decision.get('reasoning', 'Default action')
            
            print(f"🎯 Gemini chose: {next_action}")
            print(f"💭 Reasoning: {reasoning}")
            
            return {
                **state,
                "next_tool_suggestion": next_action,
                "gemini_reasoning": reasoning,
                "last_action": "gemini_decide_action",
                "action_successful": True
            }
            
        except Exception as e:
            print(f"❌ Gemini decision error: {e}")
            # Fallback decision
            fallback_action = "capture_screenshot" if not state['current_screenshot'] else "navigate_to_next"
            
            return {
                **state,
                "next_tool_suggestion": fallback_action,
                "gemini_reasoning": f"Fallback due to error: {e}",
                "last_action": "gemini_decide_action",
                "action_successful": False,
                "errors_encountered": state["errors_encountered"] + 1
            }
    
    def capture_screenshot_node(self, state: HingeAgentState) -> HingeAgentState:
        """Capture current screen screenshot"""
        print("📸 Capturing screenshot...")
        
        screenshot_path = capture_screenshot(
            state["device"],
            f"profile_{state['current_profile_index']}_langgraph"
        )
        
        return {
            **state,
            "current_screenshot": screenshot_path,
            "last_action": "capture_screenshot",
            "action_successful": True
        }
    
    def analyze_profile_node(self, state: HingeAgentState) -> HingeAgentState:
        """Analyze profile from current screenshot"""
        print("🔍 Analyzing profile...")
        
        if not state['current_screenshot']:
            return {
                **state,
                "last_action": "analyze_profile",
                "action_successful": False
            }
        
        # Extract text and analyze profile
        profile_text = extract_text_from_image_gemini(
            state['current_screenshot'], GEMINI_API_KEY
        )
        
        ui_analysis = analyze_dating_ui_with_gemini(
            state['current_screenshot'], GEMINI_API_KEY
        )
        
        quality_score = ui_analysis.get('profile_quality_score', 0)
        print(f"📊 Profile quality: {quality_score}/10")
        
        return {
            **state,
            "profile_text": profile_text,
            "profile_analysis": ui_analysis,
            "last_action": "analyze_profile",
            "action_successful": True
        }
    
    def scroll_profile_node(self, state: HingeAgentState) -> HingeAgentState:
        """Scroll to see more profile content"""
        print("📜 Scrolling profile...")
        
        scroll_analysis = analyze_profile_scroll_content(
            state['current_screenshot'], GEMINI_API_KEY
        )
        
        if not scroll_analysis.get('should_scroll_down'):
            return {
                **state,
                "last_action": "scroll_profile",
                "action_successful": False
            }
        
        # Perform scroll
        scroll_x = int(scroll_analysis.get('scroll_area_center_x', 0.5) * state["width"])
        scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * state["height"])
        scroll_y_end = int(scroll_y_start * 0.3)
        
        swipe(state["device"], scroll_x, scroll_y_start, scroll_x, scroll_y_end)
        time.sleep(2)
        
        # Capture new content
        new_screenshot = capture_screenshot(state["device"], f"scrolled_{time.time()}")
        additional_text = extract_text_from_image_gemini(new_screenshot, GEMINI_API_KEY)
        
        # Update profile text if new content found
        updated_text = state["profile_text"]
        if additional_text and additional_text not in updated_text:
            updated_text += "\n" + additional_text
        
        return {
            **state,
            "current_screenshot": new_screenshot,
            "profile_text": updated_text,
            "last_action": "scroll_profile",
            "action_successful": True
        }
    
    def make_like_decision_node(self, state: HingeAgentState) -> HingeAgentState:
        """Make like/dislike decision based on profile analysis"""
        print("🎯 Making like/dislike decision...")
        
        analysis = state.get("profile_analysis", {})
        quality = analysis.get('profile_quality_score', 0)
        potential = analysis.get('conversation_potential', 0)
        red_flags = analysis.get('red_flags', [])
        positive_indicators = analysis.get('positive_indicators', [])
        
        # Decision logic
        should_like = False
        reason = "Default: not meeting criteria"
        
        if red_flags:
            should_like = False
            reason = f"Red flags: {', '.join(red_flags[:2])}"
        elif quality >= self.config.quality_threshold_high and potential >= self.config.conversation_threshold_high:
            should_like = True
            reason = f"Excellent profile (quality: {quality}, potential: {potential})"
        elif quality >= self.config.quality_threshold_medium and len(positive_indicators) >= self.config.min_positive_indicators:
            should_like = True
            reason = f"Good profile with positives: {', '.join(positive_indicators[:2])}"
        elif len(state["profile_text"]) > self.config.min_text_length_detailed and quality >= self.config.min_quality_for_detailed:
            should_like = True
            reason = "Detailed profile with decent quality"
        
        print(f"🎯 DECISION: {'💖 LIKE' if should_like else '👎 DISLIKE'} - {reason}")
        
        return {
            **state,
            "decision_reason": reason,
            "last_action": "make_like_decision",
            "action_successful": True,
            "profile_analysis": {**analysis, "should_like": should_like}
        }
    
    def detect_like_button_node(self, state: HingeAgentState) -> HingeAgentState:
        """Detect like button location"""
        print("🎯 Detecting like button...")
        
        # Take fresh screenshot for button detection
        fresh_screenshot = capture_screenshot(
            state["device"],
            f"like_detection_{state['current_profile_index']}"
        )
        
        like_button_info = find_ui_elements_with_gemini(
            fresh_screenshot, "like_button", GEMINI_API_KEY
        )
        
        if not like_button_info.get('element_found'):
            print("❌ Like button not found during detection")
            return {
                **state,
                "current_screenshot": fresh_screenshot,
                "last_action": "detect_like_button",
                "action_successful": False
            }
        
        confidence = like_button_info.get('confidence', 0)
        if confidence < self.config.min_button_confidence:
            print(f"❌ Like button confidence too low during detection: {confidence} < {self.config.min_button_confidence}")
            return {
                **state,
                "current_screenshot": fresh_screenshot,
                "like_button_confidence": confidence,
                "last_action": "detect_like_button",
                "action_successful": False
            }
        
        like_x = int(like_button_info['approximate_x_percent'] * state["width"])
        like_y = int(like_button_info['approximate_y_percent'] * state["height"])
        
        print(f"✅ Like button detected successfully:")
        print(f"   📐 Percentages: ({like_button_info['approximate_x_percent']:.2f}, {like_button_info['approximate_y_percent']:.2f})")
        print(f"   📍 Coordinates: ({like_x}, {like_y})")
        print(f"   🎯 Confidence: {confidence:.2f}")
        
        return {
            **state,
            "current_screenshot": fresh_screenshot,
            "like_button_coords": (like_x, like_y),
            "like_button_confidence": confidence,
            "last_action": "detect_like_button",
            "action_successful": True
        }
    
    def execute_like_node(self, state: HingeAgentState) -> HingeAgentState:
        """Execute like action with profile change verification"""
        print("💖 Executing like action...")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Re-detect like button on current screen
        fresh_screenshot = capture_screenshot(state["device"], "fresh_like_detection")
        
        # Update state immediately with fresh screenshot
        updated_state["current_screenshot"] = fresh_screenshot
        
        like_button_info = find_ui_elements_with_gemini(
            fresh_screenshot, "like_button", GEMINI_API_KEY
        )
        
        if not like_button_info.get('element_found'):
            print("❌ Like button not found on fresh screenshot")
            return {
                **updated_state,
                "last_action": "execute_like",
                "action_successful": False
            }
        
        confidence = like_button_info.get('confidence', 0)
        if confidence < self.config.min_button_confidence:
            print(f"❌ Like button confidence too low: {confidence} < {self.config.min_button_confidence}")
            return {
                **updated_state,
                "like_button_confidence": confidence,
                "last_action": "execute_like",
                "action_successful": False
            }
        
        like_x = int(like_button_info['approximate_x_percent'] * state["width"])
        like_y = int(like_button_info['approximate_y_percent'] * state["height"])
        
        print(f"🎯 Like button detected:")
        print(f"   📐 Percentages: ({like_button_info['approximate_x_percent']:.2f}, {like_button_info['approximate_y_percent']:.2f})")
        print(f"   📱 Screen size: {state['width']}x{state['height']}")
        print(f"   📍 Coordinates: ({like_x}, {like_y})")
        print(f"   🎯 Confidence: {confidence:.2f}")
        
        # Execute the like tap
        tap_with_confidence(state["device"], like_x, like_y, confidence)
        time.sleep(3)
        
        # Check if comment interface appeared
        immediate_screenshot = capture_screenshot(state["device"], "post_like_immediate")
        comment_ui = detect_comment_ui_elements(immediate_screenshot, GEMINI_API_KEY)
        comment_interface_appeared = comment_ui.get('comment_field_found', False)
        
        if comment_interface_appeared:
            print("💬 Comment interface appeared - like successful!")
            return {
                **updated_state,
                "current_screenshot": immediate_screenshot,
                "likes_sent": state["likes_sent"] + 1,
                "last_action": "execute_like", 
                "action_successful": True
            }
        
        # Check if we moved to next profile using verification
        time.sleep(2)
        verification_screenshot = capture_screenshot(state["device"], "like_verification")
        
        # Use profile change verification
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": verification_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print(f"✅ Like successful - moved to new profile (confidence: {profile_verification.get('confidence', 0):.2f})")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "likes_sent": state["likes_sent"] + 1,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "execute_like",
                "action_successful": True
            }
        else:
            print("⚠️ Like may have failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "execute_like",
                "action_successful": False
            }
    
    def generate_comment_node(self, state: HingeAgentState) -> HingeAgentState:
        """Generate comment for current profile"""
        print("💬 Generating comment...")
        
        if not state['profile_text']:
            return {
                **state,
                "last_action": "generate_comment",
                "action_successful": False
            }
        
        comment = generate_comment_gemini(state['profile_text'], GEMINI_API_KEY)
        if not comment:
            comment = self.config.default_comment
        
        comment_id = str(uuid.uuid4())
        store_generated_comment(
            comment_id=comment_id,
            profile_text=state['profile_text'],
            generated_comment=comment,
            style_used="langgraph_gemini"
        )
        
        print(f"📝 Generated comment: {comment[:50]}...")
        
        return {
            **state,
            "generated_comment": comment,
            "comment_id": comment_id,
            "last_action": "generate_comment",
            "action_successful": True
        }
    
    def handle_comment_interface_node(self, state: HingeAgentState) -> HingeAgentState:
        """Handle comment interface after like"""
        print("💬 Handling comment interface...")
        
        if not state.get('generated_comment'):
            return {
                **state,
                "last_action": "handle_comment_interface",
                "action_successful": False
            }
        
        comment = state['generated_comment']
        print(f"💬 Sending comment: {comment}")
        
        try:
            # Fresh screenshot to see current interface
            fresh_screenshot = capture_screenshot(state["device"], "comment_interface_fresh")
            comment_ui = detect_comment_ui_elements(fresh_screenshot, GEMINI_API_KEY)
            
            if not comment_ui.get('comment_field_found'):
                return {
                    **state,
                    "current_screenshot": fresh_screenshot,
                    "last_action": "handle_comment_interface",
                    "action_successful": False
                }
            
            # Tap comment field
            comment_x = int(comment_ui['comment_field_x'] * state["width"])
            comment_y = int(comment_ui['comment_field_y'] * state["height"])
            tap_with_confidence(state["device"], comment_x, comment_y, 
                              comment_ui.get('comment_field_confidence', 0.8))
            time.sleep(2)
            
            # Input text using reliable method
            state["device"].shell("input keyevent KEYCODE_CTRL_A")
            time.sleep(0.5)
            
            escaped_comment = comment.replace('"', '\\"').replace("'", "\\'")
            state["device"].shell(f'input text "{escaped_comment}"')
            time.sleep(2)
            
            # Dismiss keyboard
            dismiss_keyboard(state["device"], state["width"], state["height"])
            time.sleep(2)
            
            # Find and tap Send Like button
            send_screenshot = capture_screenshot(state["device"], "send_button_detection")
            send_button_info = find_ui_elements_with_gemini(
                send_screenshot, "send_like_button", GEMINI_API_KEY
            )
            
            if send_button_info.get('element_found'):
                send_x = int(send_button_info['approximate_x_percent'] * state["width"])
                send_y = int(send_button_info['approximate_y_percent'] * state["height"])
            else:
                # Fallback coordinates
                send_x = int(state["width"] * 0.67)
                send_y = int(state["height"] * 0.75)
            
            tap_with_confidence(state["device"], send_x, send_y, 0.8)
            time.sleep(3)
            
            # Verify comment was sent
            verification_screenshot = capture_screenshot(state["device"], "comment_verification")
            
            # Use profile change verification
            profile_verification = self._verify_profile_change_internal({
                **state,
                "current_screenshot": verification_screenshot
            })
            
            if profile_verification.get('profile_changed', False):
                print("✅ Comment sent successfully - moved to new profile")
                return {
                    **state,
                    "current_screenshot": verification_screenshot,
                    "comments_sent": state["comments_sent"] + 1,
                    "current_profile_index": state["current_profile_index"] + 1,
                    "profiles_processed": state["profiles_processed"] + 1,
                    "stuck_count": 0,
                    "last_action": "handle_comment_interface",
                    "action_successful": True
                }
            else:
                # Check if comment interface is gone
                still_in_comment = detect_comment_ui_elements(verification_screenshot, GEMINI_API_KEY)
                
                if not still_in_comment.get('comment_field_found'):
                    print("✅ Comment sent (interface closed) - stayed on profile")
                    return {
                        **state,
                        "current_screenshot": verification_screenshot,
                        "comments_sent": state["comments_sent"] + 1,
                        "last_action": "handle_comment_interface",
                        "action_successful": True
                    }
                else:
                    print("❌ Comment sending failed - still in interface")
                    return {
                        **state,
                        "current_screenshot": verification_screenshot,
                        "last_action": "handle_comment_interface",
                        "action_successful": False
                    }
            
        except Exception as e:
            print(f"❌ Comment handling failed: {e}")
            return {
                **state,
                "errors_encountered": state["errors_encountered"] + 1,
                "last_action": "handle_comment_interface",
                "action_successful": False
            }
    
    def execute_dislike_node(self, state: HingeAgentState) -> HingeAgentState:
        """Execute dislike action with profile change verification"""
        print(f"👎 Executing dislike: {state.get('decision_reason', 'criteria not met')}")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Execute dislike tap
        x_dislike = int(state["width"] * self.config.dislike_button_coords[0])
        y_dislike = int(state["height"] * self.config.dislike_button_coords[1])
        
        tap(state["device"], x_dislike, y_dislike)
        time.sleep(3)
        
        # Verify dislike using profile change detection
        verification_screenshot = capture_screenshot(state["device"], "dislike_verification")
        
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": verification_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print("✅ Dislike successful - moved to new profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "execute_dislike",
                "action_successful": True
            }
        else:
            print("⚠️ Dislike may have failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": verification_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "execute_dislike",
                "action_successful": False
            }
    
    def navigate_to_next_node(self, state: HingeAgentState) -> HingeAgentState:
        """Navigate to next profile using swipe"""
        print("➡️ Navigating to next profile...")
        
        # Store previous profile data for verification
        updated_state = {
            **state,
            "previous_profile_text": state.get('profile_text', ''),
        }
        
        current_analysis = state.get('profile_analysis', {})
        updated_state["previous_profile_features"] = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        # Execute navigation swipe
        x1_swipe = int(state["width"] * 0.15)
        y1_swipe = int(state["height"] * 0.5)
        x2_swipe = x1_swipe
        y2_swipe = int(y1_swipe * 0.75)
        
        swipe(state["device"], x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        time.sleep(3)
        
        # Verify navigation
        nav_screenshot = capture_screenshot(state["device"], "navigation_verification")
        
        profile_verification = self._verify_profile_change_internal({
            **updated_state,
            "current_screenshot": nav_screenshot
        })
        
        if profile_verification.get('profile_changed', False):
            print(f"✅ Navigation successful - moved to profile {state['current_profile_index'] + 2}")
            return {
                **updated_state,
                "current_screenshot": nav_screenshot,
                "current_profile_index": state["current_profile_index"] + 1,
                "profiles_processed": state["profiles_processed"] + 1,
                "stuck_count": 0,
                "last_action": "navigate_to_next",
                "action_successful": True
            }
        else:
            print("⚠️ Navigation failed - still on same profile")
            return {
                **updated_state,
                "current_screenshot": nav_screenshot,
                "stuck_count": state["stuck_count"] + 1,
                "last_action": "navigate_to_next",
                "action_successful": False
            }
    
    def verify_profile_change_node(self, state: HingeAgentState) -> HingeAgentState:
        """Verify if we've moved to a new profile"""
        print("🔍 Verifying profile change...")
        
        verification_result = self._verify_profile_change_internal(state)
        profile_changed = verification_result.get('profile_changed', False)
        confidence = verification_result.get('confidence', 0)
        
        print(f"📊 Profile change verification: {profile_changed} (confidence: {confidence:.2f})")
        
        return {
            **state,
            "last_action": "verify_profile_change",
            "action_successful": profile_changed
        }
    
    def recover_from_stuck_node(self, state: HingeAgentState) -> HingeAgentState:
        """Attempt recovery when stuck using multiple swipe patterns"""
        print("🔄 Attempting recovery from stuck state...")
        
        # Multiple swipe patterns for recovery
        recovery_attempts = [
            # Aggressive horizontal swipe
            (int(state["width"] * 0.9), int(state["height"] * 0.5), 
             int(state["width"] * 0.1), int(state["height"] * 0.5)),
            # Vertical swipe down
            (int(state["width"] * 0.5), int(state["height"] * 0.3), 
             int(state["width"] * 0.5), int(state["height"] * 0.7)),
            # Diagonal swipe
            (int(state["width"] * 0.8), int(state["height"] * 0.3), 
             int(state["width"] * 0.2), int(state["height"] * 0.7)),
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(recovery_attempts):
            print(f"🔄 Recovery attempt {i + 1}: Swipe from ({x1}, {y1}) to ({x2}, {y2})")
            swipe(state["device"], x1, y1, x2, y2, duration=800)
            time.sleep(2)
            
            # Check if we're unstuck
            recovery_screenshot = capture_screenshot(state["device"], f"recovery_attempt_{i}")
            current_text = extract_text_from_image_gemini(recovery_screenshot, GEMINI_API_KEY)
            
            if current_text != state.get('profile_text', ''):
                print(f"✅ Recovery successful on attempt {i + 1}")
                break
        
        # Capture final result
        final_screenshot = capture_screenshot(state["device"], "recovery_result")
        
        return {
            **state,
            "current_screenshot": final_screenshot,
            "stuck_count": 0,  # Reset stuck count after recovery
            "last_action": "recover_from_stuck",
            "action_successful": True
        }
    
    def finalize_session_node(self, state: HingeAgentState) -> HingeAgentState:
        """Finalize the automation session"""
        print("🎉 Finalizing automation session...")
        
        # Update final success rates
        final_success_rates = calculate_template_success_rates()
        update_template_weights(final_success_rates)
        
        completion_reason = state.get("completion_reason", "Session completed")
        if state["current_profile_index"] >= state["max_profiles"]:
            completion_reason = "Max profiles reached"
        elif state["errors_encountered"] > self.config.max_errors_before_abort:
            completion_reason = "Too many errors"
        
        print(f"📊 Final stats: {state['profiles_processed']} processed, {state['likes_sent']} likes, {state['comments_sent']} comments")
        
        return {
            **state,
            "should_continue": False,
            "completion_reason": completion_reason,
            "last_action": "finalize_session",
            "action_successful": True
        }
    
    def _verify_profile_change_internal(self, state: HingeAgentState) -> Dict[str, Any]:
        """Internal helper for profile change verification"""
        if not state['current_screenshot']:
            return {
                "profile_changed": False,
                "confidence": 0.0,
                "message": "No screenshot available"
            }
        
        # Extract current profile info
        current_text = extract_text_from_image_gemini(
            state['current_screenshot'], GEMINI_API_KEY
        )
        
        current_analysis = analyze_dating_ui_with_gemini(
            state['current_screenshot'], GEMINI_API_KEY
        )
        
        # Get previous profile info
        previous_text = state.get('previous_profile_text', '')
        previous_features = state.get('previous_profile_features', {})
        
        # If first profile, consider it new
        if not previous_text and not previous_features:
            return {
                "profile_changed": True,
                "confidence": 1.0,
                "message": "First profile"
            }
        
        # Compare profiles to detect change
        profile_changed = False
        reasons = []
        
        # Text comparison
        if current_text and previous_text:
            current_words = set(current_text.lower().split())
            previous_words = set(previous_text.lower().split())
            
            if len(current_words) > 0 and len(previous_words) > 0:
                overlap = len(current_words.intersection(previous_words))
                similarity = overlap / max(len(current_words), len(previous_words))
                
                if similarity < 0.3:  # Less than 30% overlap = different profile
                    profile_changed = True
                    reasons.append(f"Text similarity low: {similarity:.2f}")
        
        # Feature comparison
        current_features = {
            'age': current_analysis.get('estimated_age', 0),
            'name': current_analysis.get('name', ''),
            'location': current_analysis.get('location', ''),
            'interests': current_analysis.get('interests', [])
        }
        
        if previous_features:
            # Compare key features
            if (current_features['name'] != previous_features.get('name', '') and 
                current_features['name'] and previous_features.get('name')):
                profile_changed = True
                reasons.append("Different name")
            
            if (abs(current_features['age'] - previous_features.get('age', 0)) > 5 and 
                current_features['age'] > 0 and previous_features.get('age', 0) > 0):
                profile_changed = True
                reasons.append("Age difference")
            
            # Interest overlap
            current_interests = set(current_features.get('interests', []))
            previous_interests = set(previous_features.get('interests', []))
            if current_interests and previous_interests:
                interest_overlap = len(current_interests.intersection(previous_interests))
                interest_similarity = interest_overlap / max(len(current_interests), len(previous_interests))
                if interest_similarity < 0.2:
                    profile_changed = True
                    reasons.append(f"Interest overlap low: {interest_similarity:.2f}")
        
        # Calculate confidence
        confidence = 0.8 if profile_changed else 0.3
        if len(reasons) > 1:
            confidence = min(0.95, confidence + 0.1 * (len(reasons) - 1))
        
        return {
            "profile_changed": profile_changed,
            "confidence": confidence,
            "reasons": reasons,
            "current_features": current_features,
            "message": f"Profile {'changed' if profile_changed else 'unchanged'}: {', '.join(reasons) if reasons else 'similar content'}"
        }
    
    def run_automation(self) -> Dict[str, Any]:
        """Run the complete LangGraph automation workflow"""
        print("🚀 Starting LangGraph-powered Hinge automation...")
        
        initial_state = HingeAgentState(
            device=None,
            width=0,
            height=0,
            max_profiles=self.max_profiles,
            current_profile_index=0,
            profiles_processed=0,
            likes_sent=0,
            comments_sent=0,
            errors_encountered=0,
            stuck_count=0,
            current_screenshot=None,
            profile_text="",
            profile_analysis={},
            decision_reason="",
            previous_profile_text="",
            previous_profile_features={},
            last_action="",
            action_successful=True,
            retry_count=0,
            generated_comment="",
            comment_id="",
            like_button_coords=None,
            like_button_confidence=0.0,
            should_continue=True,
            completion_reason="",
            gemini_reasoning="",
            next_tool_suggestion=""
        )
        
        # Execute the LangGraph workflow
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "success": True,
                "profiles_processed": final_state.get("profiles_processed", 0),
                "likes_sent": final_state.get("likes_sent", 0),
                "comments_sent": final_state.get("comments_sent", 0),
                "errors_encountered": final_state.get("errors_encountered", 0),
                "completion_reason": final_state.get("completion_reason", "Session completed"),
                "final_success_rates": calculate_template_success_rates()
            }
            
        except Exception as e:
            print(f"❌ LangGraph automation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "profiles_processed": 0,
                "likes_sent": 0,
                "comments_sent": 0,
                "errors_encountered": 1,
                "completion_reason": f"Failed with error: {e}"
            }


# Usage example for testing
if __name__ == "__main__":
    agent = LangGraphHingeAgent(max_profiles=5)
    result = agent.run_automation()
    print(f"🎯 Automation completed: {result}")