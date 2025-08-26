#!/usr/bin/env python3
# test_cv_send_button.py

"""
Test script to verify OpenCV-based send button detection
"""

import os
import glob
from helper_functions import detect_send_button_cv

def test_cv_send_button_detection():
    """Test CV-based send button detection on existing screenshots"""
    
    print("🧪 Testing OpenCV Send Button Detection...")
    
    # Check if template exists
    template_path = "assets/send_button.png"
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        print("💡 Make sure the send button template is in the assets directory")
        return False
    
    print(f"✅ Template found: {template_path}")
    
    # Find screenshot files to test with
    screenshot_patterns = [
        "images/*send_button*.png",
        "images/*comment*.png", 
        "images/*interface*.png"
    ]
    
    test_screenshots = []
    for pattern in screenshot_patterns:
        test_screenshots.extend(glob.glob(pattern))
    
    if not test_screenshots:
        print("⚠️  No test screenshots found in images directory")
        print("💡 Run the agent once to generate screenshots with comment interfaces")
        return False
    
    # Test each screenshot
    print(f"\n🎯 Testing {len(test_screenshots)} screenshots...")
    
    successful_detections = 0
    
    for i, screenshot_path in enumerate(test_screenshots[:5], 1):  # Test max 5 screenshots
        print(f"\n📸 Test {i}: {os.path.basename(screenshot_path)}")
        
        # Run CV detection
        result = detect_send_button_cv(screenshot_path)
        
        if result.get('found'):
            successful_detections += 1
            print(f"   ✅ DETECTED - Coords: ({result['x']}, {result['y']}) | Confidence: {result['confidence']:.3f}")
        else:
            print(f"   ❌ NOT FOUND - Confidence: {result.get('confidence', 0):.3f}")
    
    # Summary
    print(f"\n🎯 Detection Summary:")
    print(f"   📊 Screenshots tested: {min(len(test_screenshots), 5)}")
    print(f"   ✅ Successful detections: {successful_detections}")
    if len(test_screenshots) > 0:
        print(f"   📈 Success rate: {successful_detections/min(len(test_screenshots), 5)*100:.1f}%")
    
    if successful_detections > 0:
        print("\n✅ CV send button detection is working!")
        print("💡 The new comment workflow will use precise CV coordinates")
        return True
    else:
        print("\n❌ No successful detections found")
        print("💡 Check:")
        print("   - Template image matches actual Send Like button style")
        print("   - Screenshots contain visible send buttons")
        print("   - Consider adjusting confidence threshold in detect_send_button_cv()")
        return False

def test_combined_cv_detection():
    """Test both like and send button detection together"""
    
    print("\n🧪 Testing Combined CV Detection (Like + Send Buttons)...")
    
    from helper_functions import detect_like_button_cv
    
    # Find any screenshot to test both detections
    screenshots = glob.glob("images/*.png")
    if not screenshots:
        print("❌ No screenshots found")
        return False
    
    test_screenshot = screenshots[0]
    print(f"📸 Testing on: {os.path.basename(test_screenshot)}")
    
    # Test like button detection
    like_result = detect_like_button_cv(test_screenshot)
    print(f"💖 Like Button: {'✅ FOUND' if like_result.get('found') else '❌ NOT FOUND'}")
    
    # Test send button detection  
    send_result = detect_send_button_cv(test_screenshot)
    print(f"📤 Send Button: {'✅ FOUND' if send_result.get('found') else '❌ NOT FOUND'}")
    
    both_found = like_result.get('found', False) and send_result.get('found', False)
    print(f"\n🎯 Both buttons detected: {'✅ YES' if both_found else '❌ NO'}")
    
    return True

if __name__ == "__main__":
    success1 = test_cv_send_button_detection()
    success2 = test_combined_cv_detection()
    
    exit(0 if (success1 or success2) else 1)