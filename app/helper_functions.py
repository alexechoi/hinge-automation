from ppadb.client import Client as AdbClient
import time
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import glob

load_dotenv()


def clear_screenshots_directory():
    """
    Clear all old screenshots from the images directory to prevent confusion
    """
    try:
        if os.path.exists("images"):
            # Remove all PNG files in the images directory
            old_screenshots = glob.glob("images/*.png")
            count = len(old_screenshots)
            
            if count > 0:
                print(f"🗑️  Clearing {count} old screenshots from images directory...")
                for screenshot in old_screenshots:
                    os.remove(screenshot)
                print("✅ Screenshots directory cleared")
            else:
                print("📁 Images directory already clean")
        else:
            print("📁 Images directory doesn't exist - will be created when needed")
            
    except Exception as e:
        print(f"⚠️  Warning: Could not clear screenshots directory: {e}")


# Use to connect directly
def connect_device(user_ip_address="127.0.0.1"):
    adb = AdbClient(host=user_ip_address, port=5037)
    devices = adb.devices()

    print("Devices connected: ", devices)

    if len(devices) == 0:
        print("No devices connected")
        return None
    device = devices[0]
    print(f"Connected to {device.serial}")
    return device


def capture_screenshot(device, filename):
    """
    Capture screenshot with timestamp to prevent confusion between screenshots
    """
    timestamp = int(time.time() * 1000)  # millisecond timestamp
    
    result = device.screencap()
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    
    # Add timestamp to filename for uniqueness
    timestamped_filename = f"{timestamp}_{filename}.png"
    filepath = f"images/{timestamped_filename}"
    
    with open(filepath, "wb") as fp:
        fp.write(result)
    
    print(f"📸 Screenshot saved: {filepath}")
    return filepath


def tap(device, x, y):
    """Basic tap function"""
    device.shell(f"input tap {x} {y}")


def tap_with_confidence(device, x, y, confidence=1.0, tap_area_size="medium"):
    """
    Enhanced tap function with accuracy adjustments based on confidence and area size
    """
    # Adjust tap position based on confidence and area size
    if confidence < 0.7:
        # If low confidence, tap slightly offset to increase hit chance
        offset = 20 if tap_area_size == "small" else 10
        device.shell(f"input tap {x - offset} {y}")
        time.sleep(0.2)
        device.shell(f"input tap {x + offset} {y}")
    elif tap_area_size == "large":
        # For large areas, tap the center
        device.shell(f"input tap {x} {y}")
    else:
        # Standard tap
        device.shell(f"input tap {x} {y}")
    
    print(f"Tapped at ({x}, {y}) with confidence {confidence:.2f}")


def dismiss_keyboard(device, width=None, height=None):
    """
    Try multiple methods to dismiss/hide the on-screen keyboard
    
    Returns:
        bool: True if likely successful, False otherwise
    """
    methods_tried = []
    
    try:
        # Method 1: Press Enter (might send message in some apps)
        print("  📥 Trying ENTER key to close keyboard...")
        device.shell("input keyevent KEYCODE_ENTER")
        methods_tried.append("ENTER")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  ENTER key failed: {e}")
    
    try:
        # Method 2: Back key to hide keyboard
        print("  ⬅️  Trying BACK key to hide keyboard...")
        device.shell("input keyevent KEYCODE_BACK")
        methods_tried.append("BACK")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  BACK key failed: {e}")
    
    try:
        # Method 3: Hide keyboard ADB command
        print("  📱 Trying hide keyboard command...")
        device.shell("ime disable com.android.inputmethod.latin/.LatinIME")
        time.sleep(0.5)
        device.shell("ime enable com.android.inputmethod.latin/.LatinIME")
        methods_tried.append("IME_TOGGLE")
        time.sleep(1)
        
    except Exception as e:
        print(f"  ⚠️  IME toggle failed: {e}")
    
    try:
        # Method 4: Tap outside keyboard area
        if width and height:
            print("  👆 Trying tap outside keyboard area...")
            # Tap in upper third of screen where keyboard shouldn't be
            tap(device, int(width * 0.5), int(height * 0.25))
            methods_tried.append("TAP_OUTSIDE")
            time.sleep(1)
            
    except Exception as e:
        print(f"  ⚠️  Tap outside failed: {e}")
    
    print(f"  📝 Keyboard dismissal methods tried: {', '.join(methods_tried)}")
    return len(methods_tried) > 0


def input_text(device, text):
    # Escape spaces in the text
    text = text.replace(" ", "%s")
    print("text to be written: ", text)
    device.shell(f'input text "{text}"')


def swipe(device, x1, y1, x2, y2, duration=500):
    device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def generate_comment(profile_text):
    """Legacy function - now uses Gemini via gemini_analyzer"""
    from gemini_analyzer import generate_comment_gemini
    from config import GEMINI_API_KEY
    return generate_comment_gemini(profile_text, GEMINI_API_KEY)


def get_screen_resolution(device):
    output = device.shell("wm size")
    print("screen size: ", output)
    resolution = output.strip().split(":")[1].strip()
    width, height = map(int, resolution.split("x"))
    return width, height


def detect_like_button_cv(screenshot_path):
    """
    Detect like button using OpenCV template matching
    
    Returns:
        dict: {
            'found': bool,
            'x': int, 
            'y': int,
            'confidence': float,
            'width': int,
            'height': int
        }
    """
    try:
        # Load template image
        template_path = "assets/like_button.png"
        if not os.path.exists(template_path):
            print(f"❌ Like button template not found: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Load screenshot and template
        screenshot = cv2.imread(screenshot_path)
        template = cv2.imread(template_path)
        
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return {'found': False, 'confidence': 0.0}
            
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            return {'found': False, 'confidence': 0.0}
        
        # Get template dimensions
        template_height, template_width = template.shape[:2]
        
        # Convert to grayscale for better matching
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # max_val is the confidence score (0-1)
        confidence = float(max_val)
        
        # Calculate center coordinates
        top_left = max_loc
        center_x = top_left[0] + template_width // 2
        center_y = top_left[1] + template_height // 2
        
        # Consider it found if confidence is above threshold
        confidence_threshold = 0.7
        found = confidence >= confidence_threshold
        
        print(f"🎯 CV Like Button Detection:")
        print(f"   📍 Center: ({center_x}, {center_y})")
        print(f"   📐 Template size: {template_width}x{template_height}")
        print(f"   🎯 Confidence: {confidence:.3f}")
        print(f"   ✅ Found: {found} (threshold: {confidence_threshold})")
        
        return {
            'found': found,
            'x': center_x,
            'y': center_y, 
            'confidence': confidence,
            'width': template_width,
            'height': template_height,
            'top_left_x': top_left[0],
            'top_left_y': top_left[1]
        }
        
    except Exception as e:
        print(f"❌ CV like button detection failed: {e}")
        return {'found': False, 'confidence': 0.0}


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)