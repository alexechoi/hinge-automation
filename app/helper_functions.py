from ppadb.client import Client as AdbClient
import time
import cv2
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()


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
    result = device.screencap()
    # Ensure images directory exists
    os.makedirs("images", exist_ok=True)
    with open("images/" + str(filename) + ".png", "wb") as fp:
        fp.write(result)
    return "images/" + str(filename) + ".png"


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


def open_hinge(device):
    package_name = "co.match.android.matchhinge"
    device.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")
    time.sleep(5)