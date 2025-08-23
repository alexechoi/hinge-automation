#!/usr/bin/env python3
"""
Test script for Gemini API integration
"""

import os
from dotenv import load_dotenv
from gemini_analyzer import extract_text_from_image_gemini, analyze_profile_with_gemini

load_dotenv()

def test_gemini_integration():
    """Test if Gemini API integration works"""
    
    # Check if API key is available
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please set GEMINI_API_KEY in your .env file")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Test with a sample image (if it exists)
    test_image_paths = [
        "images/screen.png",
        "../images/screen.png",
        "test_image.png"
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("⚠️  No test image found. Create a sample image to test the integration.")
        print("Expected paths: images/screen.png")
        return False
    
    print(f"📷 Using test image: {test_image}")
    
    try:
        # Test basic text extraction
        print("🧪 Testing basic text extraction...")
        text = extract_text_from_image_gemini(test_image, gemini_api_key)
        print(f"✅ Text extraction successful. Length: {len(text)} characters")
        if text:
            print(f"Preview: {text[:100]}...")
        else:
            print("⚠️  No text extracted (image might be empty or contain no text)")
        
        # Test comprehensive analysis
        print("\n🧪 Testing comprehensive profile analysis...")
        analysis = analyze_profile_with_gemini(test_image, gemini_api_key)
        print(f"✅ Profile analysis successful. Keys: {list(analysis.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini integration: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Gemini API Integration\n")
    success = test_gemini_integration()
    
    if success:
        print("\n✅ All tests passed! Gemini integration is ready to use.")
    else:
        print("\n❌ Tests failed. Check your API key and network connection.")