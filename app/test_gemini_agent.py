#!/usr/bin/env python3
# app/test_gemini_agent.py

"""
Test script for Hinge automation system using Gemini AI.
This script tests various components to ensure proper setup and functionality.
"""

import os
import time
from config import GEMINI_API_KEY
from helper_functions import connect_device, get_screen_resolution, capture_screenshot
from gemini_analyzer import extract_text_from_image_gemini, analyze_dating_ui_with_gemini


def test_gemini_connection():
    """Test if Gemini API connection works properly"""
    print("🧪 Testing Gemini API connection...")
    
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("💡 Make sure to set your API key in the .env file:")
        print("   echo 'GEMINI_API_KEY=your-key-here' > .env")
        return False
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Simple text generation test
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=['Say "Hello, Hinge automation test successful!" if you can read this.']
        )
        
        if response.text and "successful" in response.text.lower():
            print("✅ Gemini API connection working properly")
            print(f"📝 Response: {response.text.strip()}")
            return True
        else:
            print("⚠️ Gemini API responded but response seems unexpected")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API connection failed: {e}")
        return False


def test_device_connection():
    """Test ADB device connection"""
    print("\n🧪 Testing device connection...")
    
    try:
        device = connect_device()
        if not device:
            print("❌ No device connected")
            print("💡 Make sure to:")
            print("   1. Enable USB debugging on your Android device")
            print("   2. Connect device via USB")
            print("   3. Authorize computer on device when prompted")
            print("   4. Run 'adb devices' to verify connection")
            return False, None
        
        print(f"✅ Device connected: {device.serial}")
        
        # Test screen resolution
        width, height = get_screen_resolution(device)
        print(f"📱 Screen resolution: {width}x{height}")
        
        return True, device
        
    except Exception as e:
        print(f"❌ Device connection failed: {e}")
        return False, None


def test_screenshot_and_analysis(device):
    """Test screenshot capture and Gemini analysis"""
    print("\n🧪 Testing screenshot and analysis...")
    
    if not device:
        print("⚠️ Skipping screenshot test - no device connected")
        return False
    
    try:
        # Capture test screenshot
        print("📸 Capturing test screenshot...")
        screenshot_path = capture_screenshot(device, "gemini_test")
        
        if not os.path.exists(screenshot_path):
            print(f"❌ Screenshot not saved to {screenshot_path}")
            return False
        
        print(f"✅ Screenshot saved: {screenshot_path}")
        
        # Test text extraction
        print("🔍 Testing text extraction with Gemini...")
        extracted_text = extract_text_from_image_gemini(screenshot_path, GEMINI_API_KEY)
        
        if extracted_text:
            print(f"✅ Text extraction successful")
            print(f"📝 Extracted text preview: {extracted_text[:100]}...")
        else:
            print("⚠️ No text extracted (may be normal if screen has no text)")
        
        # Test UI analysis
        print("🎯 Testing UI analysis with Gemini...")
        ui_analysis = analyze_dating_ui_with_gemini(screenshot_path, GEMINI_API_KEY)
        
        if ui_analysis:
            print("✅ UI analysis successful")
            print(f"📊 Analysis keys: {list(ui_analysis.keys())}")
            if 'profile_quality_score' in ui_analysis:
                print(f"🎯 Profile quality score: {ui_analysis.get('profile_quality_score', 'N/A')}")
        else:
            print("⚠️ UI analysis returned empty (may indicate issue)")
        
        return True
        
    except Exception as e:
        print(f"❌ Screenshot and analysis test failed: {e}")
        return False


def test_agent_config():
    """Test agent configuration loading"""
    print("\n🧪 Testing agent configuration...")
    
    try:
        from agent_config import DEFAULT_CONFIG, FAST_CONFIG, CONSERVATIVE_CONFIG
        
        configs = {
            "default": DEFAULT_CONFIG,
            "fast": FAST_CONFIG,
            "conservative": CONSERVATIVE_CONFIG
        }
        
        for name, config in configs.items():
            print(f"✅ {name.capitalize()} config loaded:")
            print(f"   📊 Quality threshold: {config.quality_threshold_medium}")
            print(f"   ⏱️  Max retries: {config.max_retries_per_action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Hinge Automation System Tests")
    print("=" * 50)
    
    results = {
        "gemini": test_gemini_connection(),
        "device": test_device_connection(),
        "config": test_agent_config()
    }
    
    device_connected, device = results["device"]
    if device_connected:
        results["screenshot"] = test_screenshot_and_analysis(device)
    else:
        results["screenshot"] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Results Summary:")
    print("=" * 50)
    
    for test_name, passed in results.items():
        if test_name == "device":
            passed = device_connected
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.capitalize():12} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result is True)
    if not device_connected and results["device"] is not False:
        passed_tests += 1  # Account for device test returning tuple
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! System is ready for automation.")
        print("💡 You can now run: uv run python main_agent.py")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        
        if not results["gemini"]:
            print("🔧 Fix Gemini API setup first")
        if not device_connected:
            print("🔧 Fix device connection next")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)