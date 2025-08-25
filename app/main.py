# app/main.py

import asyncio
import time
import uuid
from dotenv import load_dotenv
from multiprocessing import Process, freeze_support, set_start_method
from ppadb.client import Client as AdbClient

# Import your prompt engine weight updater
from prompt_engine import update_template_weights
from config import GEMINI_API_KEY

# Import your existing helper functions
from helper_functions import (
    connect_device,
    connect_device_remote,
    get_screen_resolution,
    open_hinge,
    swipe,
    capture_screenshot,
    generate_comment,
    tap,
    tap_with_confidence,
    scroll_profile,
    input_text,
)

# Import Gemini-based image analyzer
from gemini_analyzer import (
    extract_text_from_image_gemini,
    analyze_dating_ui_with_gemini,
    find_ui_elements_with_gemini,
    analyze_profile_scroll_content,
    get_profile_navigation_strategy,
    detect_comment_ui_elements
)

# Import data store logic for success-rate tracking
from data_store import (
    store_generated_comment,
    calculate_template_success_rates,
)

# Using Gemini API instead of OpenAI


def scroll_and_gather_profile_content(device, screenshot_path, width, height, gemini_api_key):
    """
    Scroll through profile to gather complete content, then return to original position
    
    Returns:
        complete_profile_text: All text content found
        final_screenshot_path: Screenshot after returning to original position
    """
    print("📜 Phase 1: Gathering complete profile content...")
    
    # Start with initial content
    initial_text = extract_text_from_image_gemini(screenshot_path, gemini_api_key).strip()
    complete_profile_text = initial_text
    
    scroll_attempts = 0
    max_scrolls = 3
    scroll_positions = []  # Track how much we scrolled
    
    current_screenshot = screenshot_path
    
    while scroll_attempts < max_scrolls:
        scroll_analysis = analyze_profile_scroll_content(current_screenshot, gemini_api_key)
        
        if scroll_analysis.get('should_scroll_down'):
            print(f"  ⬇️  Scrolling down to find more content (attempt {scroll_attempts + 1}/{max_scrolls})")
            
            # Calculate scroll parameters
            scroll_x = int(scroll_analysis.get('scroll_area_center_x', 0.5) * width)
            scroll_y_start = int(scroll_analysis.get('scroll_area_center_y', 0.6) * height)
            scroll_y_end = int(scroll_y_start * 0.3)  # Scroll up significantly
            
            # Perform scroll and track it
            swipe(device, scroll_x, scroll_y_start, scroll_x, scroll_y_end)
            scroll_distance = scroll_y_start - scroll_y_end
            scroll_positions.append(scroll_distance)
            
            time.sleep(2)  # Wait for scroll to complete
            
            # Take new screenshot and extract additional text
            current_screenshot = capture_screenshot(device, f"screen_scroll_content_{scroll_attempts}")
            additional_text = extract_text_from_image_gemini(current_screenshot, gemini_api_key).strip()
            
            if additional_text and additional_text not in complete_profile_text:
                complete_profile_text += "\n" + additional_text
                print(f"  ✅ Found additional content: {additional_text[:100]}...")
            else:
                print("  ℹ️  No new content found")
            
            scroll_attempts += 1
        else:
            print("  ✅ No more content to scroll")
            break
    
    print(f"📋 Content gathering complete. Total text: {len(complete_profile_text)} characters")
    
    # CRITICAL: Scroll back to original position for button detection
    if scroll_positions:
        print("🔄 Phase 2: Returning to original position for button detection...")
        
        # Scroll back up by the total distance we scrolled down
        total_scroll_back = sum(scroll_positions)
        scroll_back_x = int(width * 0.5)
        scroll_back_start = int(height * 0.3)
        scroll_back_end = scroll_back_start + int(total_scroll_back)
        
        # Make sure we don't scroll beyond screen
        scroll_back_end = min(scroll_back_end, int(height * 0.8))
        
        print(f"  ⬆️  Scrolling back to original position...")
        swipe(device, scroll_back_x, scroll_back_start, scroll_back_x, scroll_back_end)
        time.sleep(2)
        
        # Take final screenshot at original position
        final_screenshot_path = capture_screenshot(device, "screen_back_to_original")
        print("  ✅ Returned to original position")
    else:
        final_screenshot_path = current_screenshot
        print("  ℹ️  No scrolling performed, using current position")
    
    return complete_profile_text, final_screenshot_path


def detect_buttons_for_action(screenshot_path, width, height, gemini_api_key):
    """
    Detect button positions from the current screenshot for immediate action
    
    Returns:
        dict with button information
    """
    print("🎯 Phase 3: Detecting buttons for immediate action...")
    
    # Detect like button from CURRENT screenshot
    like_button_info = find_ui_elements_with_gemini(screenshot_path, "like_button", gemini_api_key)
    
    button_data = {
        'like_button_found': False,
        'like_x': None,
        'like_y': None,
        'confidence': 0.0,
        'tap_area_size': 'medium'
    }
    
    if like_button_info.get('element_found') and like_button_info.get('confidence', 0) > 0.5:
        button_data['like_button_found'] = True
        button_data['like_x'] = int(like_button_info['approximate_x_percent'] * width)
        button_data['like_y'] = int(like_button_info['approximate_y_percent'] * height)
        button_data['confidence'] = like_button_info.get('confidence', 0.8)
        button_data['tap_area_size'] = like_button_info.get('tap_area_size', 'medium')
        
        print(f"  ✅ Like button detected: ({button_data['like_x']}, {button_data['like_y']}) confidence: {button_data['confidence']:.2f}")
    else:
        print("  ❌ Like button not reliably detected")
    
    return button_data


def handle_comment_interface(device, comment_text, width, height, gemini_api_key, max_retries=3):
    """
    Handle the comment interface after tapping like button.
    
    Returns:
        bool: True if comment was successfully sent, False otherwise
    """
    print("💬 Phase 4: Handling comment interface...")
    
    for attempt in range(max_retries):
        try:
            print(f"  📝 Comment attempt {attempt + 1}/{max_retries}")
            
            # Wait for comment interface to load
            time.sleep(2)
            
            # Take screenshot of comment interface
            comment_screenshot = capture_screenshot(device, f"comment_interface_{attempt}")
            
            # Detect comment UI elements using Gemini
            comment_ui = detect_comment_ui_elements(comment_screenshot, gemini_api_key)
            
            print(f"  📊 Interface analysis: {comment_ui.get('description', 'No description')}")
            
            # Check if comment field is found
            if not comment_ui.get('comment_field_found'):
                print(f"  ❌ Comment field not found (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    # Try tapping in approximate comment area
                    approximate_comment_y = int(height * 0.75)  # Based on your screenshot
                    tap(device, int(width * 0.5), approximate_comment_y)
                    continue
                else:
                    return False
            
            # Tap on comment field
            comment_x = int(comment_ui['comment_field_x'] * width)
            comment_y = int(comment_ui['comment_field_y'] * height)
            comment_confidence = comment_ui.get('comment_field_confidence', 0.8)
            
            print(f"  👆 Tapping comment field at ({comment_x}, {comment_y}) confidence: {comment_confidence:.2f}")
            tap_with_confidence(device, comment_x, comment_y, comment_confidence)
            
            # Wait for text field to become active
            time.sleep(1.5)
            
            # Input the comment text
            print(f"  ⌨️  Typing comment: {comment_text}")
            input_text(device, comment_text)
            
            # Wait for text to be entered
            time.sleep(2)
            
            # Take another screenshot to verify text was entered and find send button
            after_text_screenshot = capture_screenshot(device, f"after_text_{attempt}")
            send_ui = detect_comment_ui_elements(after_text_screenshot, gemini_api_key)
            
            # Check for send button
            if not send_ui.get('send_button_found'):
                print(f"  ⚠️  Send button not found, trying fallback coordinates")
                # Based on your screenshot, "Send Like" button is around this area
                send_x = int(width * 0.75)  # Right side
                send_y = int(height * 0.82)  # Near bottom
            else:
                send_x = int(send_ui['send_button_x'] * width)
                send_y = int(send_ui['send_button_y'] * height)
                send_confidence = send_ui.get('send_button_confidence', 0.8)
                print(f"  ✅ Send button detected at ({send_x}, {send_y}) confidence: {send_confidence:.2f}")
            
            # Tap send button
            print(f"  📤 Tapping Send button at ({send_x}, {send_y})")
            tap_with_confidence(device, send_x, send_y, send_ui.get('send_button_confidence', 0.8))
            
            # Wait for send action to complete
            time.sleep(3)
            
            # Take final screenshot to verify success
            final_screenshot = capture_screenshot(device, f"after_send_{attempt}")
            
            # Check if we're back to the main interface (comment interface should be gone)
            # You could add verification logic here if needed
            
            print("  ✅ Comment sent successfully!")
            return True
            
        except Exception as e:
            print(f"  ❌ Error in comment attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"  🔄 Retrying comment submission...")
                # Take screenshot for debugging
                capture_screenshot(device, f"comment_error_{attempt}")
                time.sleep(2)
                continue
            else:
                print("  ❌ All comment attempts failed")
                
                # Try to cancel/escape the comment interface
                try:
                    print("  🚫 Attempting to cancel comment interface...")
                    # Look for cancel button or try back gesture
                    cancel_ui = detect_comment_ui_elements(final_screenshot, gemini_api_key)
                    if cancel_ui.get('cancel_button_found'):
                        cancel_x = int(cancel_ui['cancel_button_x'] * width)
                        cancel_y = int(cancel_ui['cancel_button_y'] * height)
                        tap(device, cancel_x, cancel_y)
                    else:
                        # Try back gesture or escape
                        device.shell("input keyevent KEYCODE_BACK")
                    time.sleep(2)
                except:
                    pass
                
                return False
    
    return False


def main():
    device = connect_device("127.0.0.1")
    if not device:
        return

    width, height = get_screen_resolution(device)

    # Approximate coordinates for dislike button (always at bottom)
    x_dislike_button_approx = int(width * 0.15)
    y_dislike_button_approx = int(height * 0.85)

    # Navigation coordinates
    x1_swipe = int(width * 0.15)
    x2_swipe = x1_swipe
    y1_swipe = int(height * 0.5)
    y2_swipe = int(y1_swipe * 0.75)

    print("🚀 Using Fixed Gemini AI with Proper Button Detection")

    open_hinge(device=device)
    time.sleep(5)

    previous_profile_text = ""
    success_rates = calculate_template_success_rates()
    update_template_weights(success_rates)

    for profile_index in range(10):
        print(f"\n🔄 === Processing Profile {profile_index + 1}/10 ===")
        
        # STEP 1: Initial screenshot for navigation check
        initial_screenshot = capture_screenshot(device, f"initial_profile_{profile_index}")
        
        # Check navigation strategy
        nav_strategy = get_profile_navigation_strategy(initial_screenshot, GEMINI_API_KEY)
        print(f"📱 Screen type: {nav_strategy.get('screen_type', 'unknown')}")
        
        # Check if stuck on same profile
        current_profile_text = extract_text_from_image_gemini(initial_screenshot, GEMINI_API_KEY).strip()
        
        if (previous_profile_text == current_profile_text and current_profile_text != "" and profile_index > 0):
            print("⚠️  STUCK: Same profile detected! Using aggressive navigation...")
            # Use aggressive swipe to get unstuck
            swipe(device, int(width * 0.9), int(height * 0.5), int(width * 0.1), int(height * 0.3))
            time.sleep(3)
            initial_screenshot = capture_screenshot(device, f"after_unstuck_{profile_index}")
            current_profile_text = extract_text_from_image_gemini(initial_screenshot, GEMINI_API_KEY).strip()
        
        # STEP 2: Gather complete profile content (with return to original position)
        complete_profile_text, final_screenshot = scroll_and_gather_profile_content(
            device, initial_screenshot, width, height, GEMINI_API_KEY
        )
        
        # STEP 3: Analyze profile quality using complete content
        print("🤖 Analyzing complete profile...")
        ui_analysis = analyze_dating_ui_with_gemini(final_screenshot, GEMINI_API_KEY)
        
        profile_quality = ui_analysis.get('profile_quality_score', 0)
        conversation_potential = ui_analysis.get('conversation_potential', 0)
        positive_indicators = ui_analysis.get('positive_indicators', [])
        red_flags = ui_analysis.get('red_flags', [])
        
        print(f"📊 Analysis: {ui_analysis.get('reason', 'No reason')}")
        print(f"⭐ Profile Quality: {profile_quality}/10")
        print(f"💬 Conversation Potential: {conversation_potential}/10")
        
        # STEP 4: Make decision based on analysis
        like_decision = False
        reason = "Default: not meeting criteria"
        
        if red_flags:
            like_decision = False
            reason = f"Red flags: {', '.join(red_flags[:2])}"
        elif profile_quality >= 8 and conversation_potential >= 7:
            like_decision = True
            reason = f"Excellent profile (quality: {profile_quality}, conversation: {conversation_potential})"
        elif profile_quality >= 6 and len(positive_indicators) >= 2:
            like_decision = True
            reason = f"Good profile with positives: {', '.join(positive_indicators[:2])}"
        elif len(complete_profile_text) > 200 and profile_quality >= 5:
            like_decision = True
            reason = "Detailed profile with decent quality"
        
        print(f"🎯 DECISION: {'💖 LIKE' if like_decision else '👎 DISLIKE'} - {reason}")
        
        # STEP 5: Detect buttons at the EXACT moment before action
        button_data = detect_buttons_for_action(final_screenshot, width, height, GEMINI_API_KEY)
        
        # STEP 6: Execute action with fresh button coordinates
        if like_decision and button_data['like_button_found']:
            # Generate comment
            comment = generate_comment(complete_profile_text) or "Hey, I'd love to meet up!"
            print(f"💬 Generated Comment: {comment}")

            # Store for analytics
            comment_id = str(uuid.uuid4())
            store_generated_comment(
                comment_id=comment_id,
                profile_text=complete_profile_text,
                generated_comment=comment,
                style_used="gemini_fixed",
            )

            # Tap like button with current coordinates
            print(f"💖 Tapping LIKE at current position: ({button_data['like_x']}, {button_data['like_y']})")
            tap_with_confidence(
                device, 
                button_data['like_x'], 
                button_data['like_y'], 
                button_data['confidence'], 
                button_data['tap_area_size']
            )
            
            # Handle the comment interface that appears after liking
            comment_success = handle_comment_interface(
                device, comment, width, height, GEMINI_API_KEY, max_retries=3
            )
            
            if comment_success:
                print("  🎉 Like with comment sent successfully!")
            else:
                print("  ⚠️  Like sent but comment may have failed")
            
            time.sleep(3)  # Wait for interface to settle
            
        else:
            # Execute dislike
            if not button_data['like_button_found'] and like_decision:
                reason = "Like button not found"
            
            print(f"👎 Tapping DISLIKE: {reason}")
            tap(device, x_dislike_button_approx, y_dislike_button_approx)

        # STEP 7: Navigate to next profile
        time.sleep(3)
        print("➡️  Navigating to next profile...")
        swipe(device, x1_swipe, y1_swipe, x2_swipe, y2_swipe)

        previous_profile_text = current_profile_text
        time.sleep(3)

    print("\n🎉 Processing complete!")
    final_success_rates = calculate_template_success_rates()
    update_template_weights(final_success_rates)
    print("Final success rates:", final_success_rates)


if __name__ == "__main__":
    main()