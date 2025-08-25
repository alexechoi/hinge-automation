# app/langgraph_agent.py

import time
import uuid
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
import operator

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


class HingeAgentState(TypedDict):
    """State maintained throughout the dating app automation"""
    # Device and screen info
    device: Any
    width: int
    height: int
    
    # Current session info
    profile_index: int
    max_profiles: int
    processed_profiles: List[str]
    
    # Current profile analysis
    current_screenshot: Optional[str]
    profile_text: str
    profile_quality: int
    conversation_potential: int
    like_decision: bool
    decision_reason: str
    
    # UI state
    ui_elements: Dict[str, Any]
    button_locations: Dict[str, Any]
    
    # Action results
    last_action: str
    action_successful: bool
    error_count: int
    retry_count: int
    
    # Generated content
    generated_comment: str
    comment_id: str
    
    # Navigation
    stuck_count: int
    previous_profile_text: str
    
    # Control flow
    should_continue: bool
    completion_reason: str


@dataclass
class ToolResult:
    success: bool
    data: Dict[str, Any]
    error_message: str = ""
    screenshot_path: str = ""
    
    
class HingeAutomationAgent:
    """LangGraph-based Hinge automation agent with individual tools"""
    
    def __init__(self, max_profiles: int = 10, config=None):
        from agent_config import DEFAULT_CONFIG
        
        self.max_profiles = max_profiles
        self.config = config or DEFAULT_CONFIG
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph with tools as nodes"""
        
        workflow = StateGraph(HingeAgentState)
        
        # Add nodes (tools)
        workflow.add_node("initialize", self.initialize_session)
        workflow.add_node("capture_screen", self.capture_screen_tool)
        workflow.add_node("analyze_screen", self.analyze_screen_tool)
        workflow.add_node("gather_profile_content", self.gather_profile_content_tool)
        workflow.add_node("analyze_profile_quality", self.analyze_profile_quality_tool)
        workflow.add_node("make_decision", self.make_decision_tool)
        workflow.add_node("detect_ui_elements", self.detect_ui_elements_tool)
        workflow.add_node("execute_like", self.execute_like_tool)
        workflow.add_node("handle_comment_interface", self.handle_comment_interface_tool)
        workflow.add_node("execute_dislike", self.execute_dislike_tool)
        workflow.add_node("navigate_next", self.navigate_next_tool)
        workflow.add_node("verify_navigation", self.verify_navigation_tool)
        workflow.add_node("handle_error", self.handle_error_tool)
        workflow.add_node("finalize", self.finalize_session)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add edges with conditional logic
        workflow.add_conditional_edges(
            "initialize",
            self._should_continue,
            {
                "continue": "capture_screen",
                "end": END
            }
        )
        
        workflow.add_edge("capture_screen", "analyze_screen")
        
        workflow.add_conditional_edges(
            "analyze_screen", 
            self._route_screen_analysis,
            {
                "gather_content": "gather_profile_content",
                "stuck": "handle_error",
                "navigate": "navigate_next"
            }
        )
        
        workflow.add_edge("gather_profile_content", "analyze_profile_quality")
        workflow.add_edge("analyze_profile_quality", "make_decision")
        
        workflow.add_conditional_edges(
            "make_decision",
            self._route_decision,
            {
                "like": "detect_ui_elements",
                "dislike": "execute_dislike"
            }
        )
        
        workflow.add_conditional_edges(
            "detect_ui_elements",
            self._route_ui_detection,
            {
                "like_found": "execute_like",
                "like_not_found": "execute_dislike",
                "retry": "detect_ui_elements"
            }
        )
        
        workflow.add_conditional_edges(
            "execute_like",
            self._route_like_result,
            {
                "comment_interface": "handle_comment_interface",
                "success": "navigate_next", 
                "retry": "execute_like",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_comment_interface",
            self._route_comment_result,
            {
                "success": "navigate_next",
                "retry": "handle_comment_interface", 
                "error": "navigate_next"  # Continue even if comment fails
            }
        )
        
        workflow.add_edge("execute_dislike", "navigate_next")
        
        workflow.add_conditional_edges(
            "navigate_next",
            self._route_navigation,
            {
                "verify": "verify_navigation",
                "complete": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "verify_navigation",
            self._route_verification,
            {
                "success": "capture_screen",
                "stuck": "handle_error",
                "complete": "finalize"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_error",
            self._route_error_handling,
            {
                "recover": "capture_screen",
                "retry_navigation": "navigate_next",
                "abort": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # Tool implementations
    def initialize_session(self, state: HingeAgentState) -> HingeAgentState:
        """Initialize the automation session"""
        print("🚀 Initializing Hinge automation session...")
        
        device = connect_device(self.config.device_ip)
        if not device:
            return {**state, "should_continue": False, "completion_reason": "Failed to connect to device"}
            
        width, height = get_screen_resolution(device)
        open_hinge(device)
        time.sleep(5)
        
        # Update template weights with current success rates
        success_rates = calculate_template_success_rates()
        update_template_weights(success_rates)
        
        return {
            **state,
            "device": device,
            "width": width, 
            "height": height,
            "profile_index": 0,
            "max_profiles": self.max_profiles,
            "processed_profiles": [],
            "error_count": 0,
            "stuck_count": 0,
            "should_continue": True,
            "previous_profile_text": ""
        }
    
    def capture_screen_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Capture current screen screenshot"""
        print(f"📱 Capturing screen for profile {state['profile_index'] + 1}")
        
        screenshot_path = capture_screenshot(
            state["device"], 
            f"profile_{state['profile_index']}_screen"
        )
        
        return {
            **state,
            "current_screenshot": screenshot_path,
            "last_action": "capture_screen",
            "action_successful": True
        }
    
    def analyze_screen_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Analyze current screen to determine state and next action"""
        print("🔍 Analyzing current screen state...")
        
        nav_strategy = get_profile_navigation_strategy(state["current_screenshot"], GEMINI_API_KEY)
        current_text = extract_text_from_image_gemini(state["current_screenshot"], GEMINI_API_KEY)
        
        # Check if stuck on same profile
        is_stuck = (state["previous_profile_text"] == current_text and 
                   current_text != "" and state["profile_index"] > 0)
        
        return {
            **state,
            "profile_text": current_text,
            "ui_elements": nav_strategy,
            "last_action": "analyze_screen",
            "action_successful": not is_stuck,
            "stuck_count": state["stuck_count"] + 1 if is_stuck else 0
        }
    
    def gather_profile_content_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Gather complete profile content by scrolling if needed"""
        print("📜 Gathering complete profile content...")
        
        complete_text = state["profile_text"]
        scroll_attempts = 0
        max_scrolls = self.config.max_scroll_attempts
        current_screenshot = state["current_screenshot"]
        
        while scroll_attempts < max_scrolls:
            scroll_analysis = analyze_profile_scroll_content(current_screenshot, GEMINI_API_KEY)
            
            if not scroll_analysis.get('should_scroll_down'):
                break
                
            # Perform scroll
            scroll_x = int(scroll_analysis.get('scroll_area_center_x', 0.5) * state["width"])
            scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * state["height"])
            scroll_y_end = int(scroll_y_start * 0.3)
            
            swipe(state["device"], scroll_x, scroll_y_start, scroll_x, scroll_y_end)
            time.sleep(2)
            
            # Capture new content
            current_screenshot = capture_screenshot(state["device"], f"scroll_content_{scroll_attempts}")
            additional_text = extract_text_from_image_gemini(current_screenshot, GEMINI_API_KEY)
            
            if additional_text and additional_text not in complete_text:
                complete_text += "\n" + additional_text
                
            scroll_attempts += 1
        
        # Return to original position for button detection
        if scroll_attempts > 0:
            scroll_back_x = int(state["width"] * 0.5)
            scroll_back_start = int(state["height"] * 0.3)  
            scroll_back_end = int(state["height"] * 0.7)
            swipe(state["device"], scroll_back_x, scroll_back_start, scroll_back_x, scroll_back_end)
            time.sleep(2)
            
            # Final screenshot at original position
            final_screenshot = capture_screenshot(state["device"], "back_to_original")
            current_screenshot = final_screenshot
        
        return {
            **state,
            "profile_text": complete_text,
            "current_screenshot": current_screenshot,
            "last_action": "gather_content",
            "action_successful": True
        }
    
    def analyze_profile_quality_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Analyze profile quality and potential"""
        print("🤖 Analyzing profile quality...")
        
        ui_analysis = analyze_dating_ui_with_gemini(state["current_screenshot"], GEMINI_API_KEY)
        
        return {
            **state,
            "profile_quality": ui_analysis.get('profile_quality_score', 0),
            "conversation_potential": ui_analysis.get('conversation_potential', 0),
            "ui_elements": {**state.get("ui_elements", {}), **ui_analysis},
            "last_action": "analyze_quality",
            "action_successful": True
        }
    
    def make_decision_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Make like/dislike decision based on analysis"""
        print("🎯 Making like/dislike decision...")
        
        quality = state["profile_quality"]
        potential = state["conversation_potential"]
        text_length = len(state["profile_text"])
        red_flags = state.get("ui_elements", {}).get('red_flags', [])
        positive_indicators = state.get("ui_elements", {}).get('positive_indicators', [])
        
        # Decision logic
        like_decision = False
        reason = "Default: not meeting criteria"
        
        if red_flags:
            like_decision = False
            reason = f"Red flags: {', '.join(red_flags[:2])}"
        elif quality >= self.config.quality_threshold_high and potential >= self.config.conversation_threshold_high:
            like_decision = True
            reason = f"Excellent profile (quality: {quality}, potential: {potential})"
        elif quality >= self.config.quality_threshold_medium and len(positive_indicators) >= self.config.min_positive_indicators:
            like_decision = True
            reason = f"Good profile with positives: {', '.join(positive_indicators[:2])}"
        elif text_length > self.config.min_text_length_detailed and quality >= self.config.min_quality_for_detailed:
            like_decision = True
            reason = "Detailed profile with decent quality"
            
        print(f"🎯 DECISION: {'💖 LIKE' if like_decision else '👎 DISLIKE'} - {reason}")
        
        return {
            **state,
            "like_decision": like_decision,
            "decision_reason": reason,
            "last_action": "make_decision",
            "action_successful": True
        }
    
    def detect_ui_elements_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Detect UI elements needed for actions"""
        print("🎯 Detecting UI elements...")
        
        like_button_info = find_ui_elements_with_gemini(
            state["current_screenshot"], "like_button", GEMINI_API_KEY
        )
        
        button_data = {
            'like_button_found': False,
            'like_x': None,
            'like_y': None,
            'confidence': 0.0,
            'tap_area_size': 'medium'
        }
        
        if like_button_info.get('element_found') and like_button_info.get('confidence', 0) > self.config.min_button_confidence:
            button_data.update({
                'like_button_found': True,
                'like_x': int(like_button_info['approximate_x_percent'] * state["width"]),
                'like_y': int(like_button_info['approximate_y_percent'] * state["height"]),
                'confidence': like_button_info.get('confidence', 0.8),
                'tap_area_size': like_button_info.get('tap_area_size', 'medium')
            })
        
        return {
            **state,
            "button_locations": button_data,
            "last_action": "detect_ui",
            "action_successful": button_data['like_button_found']
        }
    
    def execute_like_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Execute like action with generated comment"""
        print("💖 Executing like action...")
        
        # Generate comment
        comment = generate_comment_gemini(state["profile_text"], GEMINI_API_KEY)
        if not comment:
            comment = self.config.default_comment
            
        comment_id = str(uuid.uuid4())
        store_generated_comment(
            comment_id=comment_id,
            profile_text=state["profile_text"],
            generated_comment=comment,
            style_used="gemini_agent"
        )
        
        # Execute like tap
        button_data = state["button_locations"]
        tap_with_confidence(
            state["device"],
            button_data['like_x'],
            button_data['like_y'], 
            button_data['confidence'],
            button_data['tap_area_size']
        )
        
        time.sleep(2)
        
        # Verify like action
        verification_screenshot = capture_screenshot(state["device"], "like_verification")
        like_verification = verify_action_success(verification_screenshot, "like_tap", GEMINI_API_KEY)
        
        return {
            **state,
            "generated_comment": comment,
            "comment_id": comment_id,
            "current_screenshot": verification_screenshot,
            "last_action": "execute_like",
            "action_successful": like_verification.get('like_successful', False),
            "ui_elements": like_verification
        }
    
    def handle_comment_interface_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Handle comment interface after like"""
        print("💬 Handling comment interface...")
        
        max_retries = self.config.max_retries_per_action
        for attempt in range(max_retries):
            try:
                time.sleep(2)
                
                comment_screenshot = capture_screenshot(state["device"], f"comment_interface_{attempt}")
                comment_ui = detect_comment_ui_elements(comment_screenshot, GEMINI_API_KEY)
                
                if not comment_ui.get('comment_field_found'):
                    if attempt < max_retries - 1:
                        continue
                    return {
                        **state,
                        "last_action": "handle_comment",
                        "action_successful": False,
                        "retry_count": state.get("retry_count", 0) + 1
                    }
                
                # Tap comment field and input text
                comment_x = int(comment_ui['comment_field_x'] * state["width"])
                comment_y = int(comment_ui['comment_field_y'] * state["height"])
                tap_with_confidence(state["device"], comment_x, comment_y, 
                                  comment_ui.get('comment_field_confidence', 0.8))
                
                time.sleep(1.5)
                input_text(state["device"], state["generated_comment"])
                time.sleep(1)
                
                # Try Enter key first (might send directly)
                state["device"].shell("input keyevent KEYCODE_ENTER")
                time.sleep(2)
                
                # Check if Enter worked
                after_enter_screenshot = capture_screenshot(state["device"], f"after_enter_{attempt}")
                post_enter_ui = detect_comment_ui_elements(after_enter_screenshot, GEMINI_API_KEY)
                
                if not post_enter_ui.get('comment_field_found'):
                    return {
                        **state,
                        "current_screenshot": after_enter_screenshot,
                        "last_action": "handle_comment", 
                        "action_successful": True
                    }
                
                # Need to dismiss keyboard and find send button
                dismiss_keyboard(state["device"], state["width"], state["height"])
                time.sleep(2)
                
                send_screenshot = capture_screenshot(state["device"], f"send_interface_{attempt}")
                send_ui = detect_comment_ui_elements(send_screenshot, GEMINI_API_KEY)
                
                if send_ui.get('send_button_found'):
                    send_x = int(send_ui['send_button_x'] * state["width"])
                    send_y = int(send_ui['send_button_y'] * state["height"])
                else:
                    # Fallback coordinates
                    send_x = int(state["width"] * 0.75)
                    send_y = int(state["height"] * 0.82)
                
                tap_with_confidence(state["device"], send_x, send_y, 0.8)
                time.sleep(3)
                
                # Verify comment sent
                final_screenshot = capture_screenshot(state["device"], f"comment_verification_{attempt}")
                verification = verify_action_success(final_screenshot, "comment_sent", GEMINI_API_KEY)
                
                return {
                    **state,
                    "current_screenshot": final_screenshot,
                    "last_action": "handle_comment",
                    "action_successful": verification.get('comment_sent', True)  # Optimistic
                }
                
            except Exception as e:
                print(f"Comment attempt {attempt + 1} failed: {e}")
                if attempt >= max_retries - 1:
                    return {
                        **state,
                        "last_action": "handle_comment",
                        "action_successful": False,
                        "error_count": state.get("error_count", 0) + 1
                    }
                continue
        
        return {
            **state, 
            "last_action": "handle_comment",
            "action_successful": False
        }
    
    def execute_dislike_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Execute dislike action"""
        print(f"👎 Executing dislike: {state['decision_reason']}")
        
        x_dislike = int(state["width"] * self.config.dislike_button_coords[0])
        y_dislike = int(state["height"] * self.config.dislike_button_coords[1])
        
        tap(state["device"], x_dislike, y_dislike)
        time.sleep(2)
        
        return {
            **state,
            "last_action": "execute_dislike",
            "action_successful": True
        }
    
    def navigate_next_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Navigate to next profile"""
        print("➡️ Navigating to next profile...")
        
        x1_swipe = int(state["width"] * 0.15)
        y1_swipe = int(state["height"] * 0.5)
        x2_swipe = x1_swipe
        y2_swipe = int(y1_swipe * 0.75)
        
        swipe(state["device"], x1_swipe, y1_swipe, x2_swipe, y2_swipe)
        time.sleep(3)
        
        return {
            **state,
            "previous_profile_text": state["profile_text"],
            "profile_index": state["profile_index"] + 1,
            "last_action": "navigate_next", 
            "action_successful": True,
            "retry_count": 0  # Reset retry count on navigation
        }
    
    def verify_navigation_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Verify we successfully moved to next profile"""
        print("🔍 Verifying navigation...")
        
        verification_screenshot = capture_screenshot(state["device"], "navigation_verification")
        verification = verify_action_success(verification_screenshot, "profile_change", GEMINI_API_KEY)
        
        is_stuck = not verification.get('profile_changed', True)
        
        return {
            **state,
            "current_screenshot": verification_screenshot,
            "last_action": "verify_navigation",
            "action_successful": not is_stuck,
            "stuck_count": state["stuck_count"] + 1 if is_stuck else 0,
            "ui_elements": verification
        }
    
    def handle_error_tool(self, state: HingeAgentState) -> HingeAgentState:
        """Handle errors and attempt recovery"""
        print("🔄 Handling error and attempting recovery...")
        
        error_count = state.get("error_count", 0) + 1
        stuck_count = state.get("stuck_count", 0)
        
        if error_count > self.config.max_errors_before_abort or stuck_count > self.config.max_stuck_count:
            return {
                **state,
                "should_continue": False,
                "completion_reason": f"Too many errors (errors: {error_count}, stuck: {stuck_count})",
                "last_action": "handle_error",
                "action_successful": False
            }
        
        # Try different recovery strategies
        if stuck_count > 1:
            # Aggressive navigation for unstuck
            print("🔄 Trying aggressive navigation to get unstuck...")
            swipe(state["device"], 
                  int(state["width"] * 0.9), int(state["height"] * 0.5),
                  int(state["width"] * 0.1), int(state["height"] * 0.3))
            time.sleep(3)
            
            # Try back button
            try:
                state["device"].shell("input keyevent KEYCODE_BACK")
                time.sleep(1)
            except:
                pass
        
        return {
            **state,
            "error_count": error_count,
            "last_action": "handle_error",
            "action_successful": True  # Recovery attempt made
        }
    
    def finalize_session(self, state: HingeAgentState) -> HingeAgentState:
        """Finalize the automation session"""
        print("🎉 Finalizing session...")
        
        final_success_rates = calculate_template_success_rates()
        update_template_weights(final_success_rates)
        
        print(f"Session completed: {state['profile_index']} profiles processed")
        print(f"Final success rates: {final_success_rates}")
        
        return {
            **state,
            "should_continue": False,
            "completion_reason": "Session completed successfully",
            "last_action": "finalize",
            "action_successful": True
        }
    
    # Routing functions for conditional edges
    def _should_continue(self, state: HingeAgentState) -> str:
        return "continue" if state.get("should_continue", True) else "end"
    
    def _route_screen_analysis(self, state: HingeAgentState) -> str:
        if state["stuck_count"] > 2:
            return "stuck"
        elif state.get("ui_elements", {}).get("screen_type") == "profile":
            return "gather_content"
        else:
            return "navigate"
    
    def _route_decision(self, state: HingeAgentState) -> str:
        return "like" if state["like_decision"] else "dislike"
    
    def _route_ui_detection(self, state: HingeAgentState) -> str:
        if not state["action_successful"] and state.get("retry_count", 0) < 2:
            return "retry"
        elif state["button_locations"]["like_button_found"]:
            return "like_found"
        else:
            return "like_not_found"
    
    def _route_like_result(self, state: HingeAgentState) -> str:
        if not state["action_successful"]:
            if state.get("retry_count", 0) < 2:
                return "retry"
            else:
                return "error"
        elif state.get("ui_elements", {}).get("interface_state") == "comment_modal":
            return "comment_interface"
        else:
            return "success"
    
    def _route_comment_result(self, state: HingeAgentState) -> str:
        if not state["action_successful"]:
            if state.get("retry_count", 0) < 2:
                return "retry" 
            else:
                return "error"
        else:
            return "success"
    
    def _route_navigation(self, state: HingeAgentState) -> str:
        if state["profile_index"] >= state["max_profiles"]:
            return "complete"
        else:
            return "verify"
    
    def _route_verification(self, state: HingeAgentState) -> str:
        if state["profile_index"] >= state["max_profiles"]:
            return "complete"
        elif not state["action_successful"]:
            return "stuck"
        else:
            return "success"
    
    def _route_error_handling(self, state: HingeAgentState) -> str:
        error_count = state.get("error_count", 0)
        stuck_count = state.get("stuck_count", 0)
        
        if error_count > self.config.max_errors_before_abort or stuck_count > self.config.max_stuck_count:
            return "abort"
        elif stuck_count > 0:
            return "retry_navigation"
        else:
            return "recover"
    
    # Main execution method
    def run_automation(self) -> Dict[str, Any]:
        """Run the complete automation workflow"""
        print("🚀 Starting LangGraph-based Hinge automation...")
        
        initial_state = HingeAgentState(
            device=None,
            width=0,
            height=0,
            profile_index=0,
            max_profiles=self.max_profiles,
            processed_profiles=[],
            current_screenshot=None,
            profile_text="",
            profile_quality=0,
            conversation_potential=0,
            like_decision=False,
            decision_reason="",
            ui_elements={},
            button_locations={},
            last_action="",
            action_successful=True,
            error_count=0,
            retry_count=0,
            generated_comment="",
            comment_id="",
            stuck_count=0,
            previous_profile_text="",
            should_continue=True,
            completion_reason=""
        )
        
        # Execute the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "profiles_processed": result.get("profile_index", 0),
            "completion_reason": result.get("completion_reason", "Unknown"),
            "final_state": result
        }


# Usage example
if __name__ == "__main__":
    agent = HingeAutomationAgent(max_profiles=10)
    result = agent.run_automation()
    print(f"Automation completed: {result}")