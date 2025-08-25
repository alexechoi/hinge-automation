# app/test_gemini_agent.py

"""
Test script for the Gemini-controlled agent to verify integration.
This script tests the agent without actually running device automation.
"""

import os
import sys
from unittest.mock import Mock, patch
import json

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_agent_controller import GeminiAgentController
from agent_config import DEFAULT_CONFIG


def test_gemini_agent_initialization():
    """Test that the Gemini agent initializes correctly"""
    print("🧪 Testing Gemini Agent Initialization...")
    
    try:
        agent = GeminiAgentController(max_profiles=5, config=DEFAULT_CONFIG)
        
        # Check if agent was created properly
        assert agent.max_profiles == 5
        assert agent.config == DEFAULT_CONFIG
        assert agent.gemini_client is not None
        assert len(agent.available_tools) > 0
        
        print("✅ Agent initialization successful")
        print(f"📊 Available tools: {len(agent.available_tools)}")
        print(f"🔧 Tools: {list(agent.available_tools.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False


def test_tool_schema():
    """Test that the tool schema is properly defined"""
    print("\n🧪 Testing Tool Schema...")
    
    try:
        agent = GeminiAgentController(max_profiles=5)
        schema = agent.get_tool_schema()
        
        assert len(schema) > 0
        assert "Available Tools" in schema
        assert "capture_screenshot" in schema
        assert "analyze_profile" in schema
        assert "execute_like" in schema
        
        print("✅ Tool schema test passed")
        print(f"📋 Schema length: {len(schema)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool schema test failed: {e}")
        return False


def test_mock_decision_making():
    """Test Gemini decision making with a mock response"""
    print("\n🧪 Testing Mock Decision Making...")
    
    try:
        agent = GeminiAgentController(max_profiles=5)
        
        # Mock the session state
        agent.session_data = {
            'current_screenshot': '',
            'last_action': 'initialize',
            'profile_text': '',
            'stuck_count': 0,
            'errors_encountered': 0,
            'profiles_processed': 0
        }
        
        # Mock Gemini response
        mock_decision = {
            "tool_name": "capture_screenshot",
            "parameters": {},
            "reasoning": "Need to capture the current state of the screen to begin analysis",
            "confidence": 1.0,
            "expected_outcome": "Screenshot will be saved and available for analysis",
            "fallback_tool": "navigate_to_next"
        }
        
        # Test decision structure
        assert "tool_name" in mock_decision
        assert "reasoning" in mock_decision
        assert "confidence" in mock_decision
        assert mock_decision["tool_name"] in agent.available_tools
        
        print("✅ Mock decision making test passed")
        print(f"🎯 Mock decision: {mock_decision['tool_name']}")
        print(f"💭 Reasoning: {mock_decision['reasoning']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock decision making test failed: {e}")
        return False


def test_tool_execution_mock():
    """Test tool execution with mocked device"""
    print("\n🧪 Testing Tool Execution (Mocked)...")
    
    try:
        agent = GeminiAgentController(max_profiles=5)
        
        # Mock device and screen dimensions
        agent.device = Mock()
        agent.width = 1080
        agent.height = 1920
        
        # Mock session data
        agent.session_data = {
            'current_screenshot': '',
            'last_action': '',
            'profile_text': '',
            'stuck_count': 0,
            'errors_encountered': 0,
            'profiles_processed': 0
        }
        
        # Test capture_screenshot_tool (which doesn't need real device for this test)
        with patch('gemini_agent_controller.capture_screenshot') as mock_screenshot:
            mock_screenshot.return_value = 'test_screenshot.png'
            
            result = agent.capture_screenshot_tool()
            
            assert result['success'] == True
            assert 'screenshot_path' in result
            assert agent.session_data['last_action'] == 'capture_screenshot'
        
        # Test execute_dislike_tool
        with patch('gemini_agent_controller.tap') as mock_tap:
            result = agent.execute_dislike_tool()
            
            assert result['success'] == True
            assert 'message' in result
            assert agent.session_data['last_action'] == 'execute_dislike'
        
        # Test recover_from_stuck_tool (should not use back button)
        with patch('gemini_agent_controller.swipe') as mock_swipe, \
             patch('gemini_agent_controller.capture_screenshot') as mock_screenshot, \
             patch('gemini_agent_controller.extract_text_from_image_gemini') as mock_text:
            
            mock_screenshot.return_value = 'recovery_test.png'
            mock_text.return_value = 'different_content'
            
            result = agent.recover_from_stuck_tool()
            
            assert result['success'] == True
            assert 'swipe patterns only' in result['message']
            assert agent.session_data['last_action'] == 'recover_from_stuck'
            # Verify swipe was called (recovery method)
            assert mock_swipe.called
        
        print("✅ Tool execution test passed")
        print(f"🔧 Tools tested: capture_screenshot, execute_dislike, recover_from_stuck")
        
        return True
        
    except Exception as e:
        print(f"❌ Tool execution test failed: {e}")
        return False


def test_configuration_integration():
    """Test that configuration is properly integrated"""
    print("\n🧪 Testing Configuration Integration...")
    
    try:
        # Test with different configurations
        from agent_config import FAST_CONFIG, CONSERVATIVE_CONFIG
        
        # Test default config
        agent1 = GeminiAgentController(max_profiles=10, config=DEFAULT_CONFIG)
        assert agent1.config == DEFAULT_CONFIG
        
        # Test fast config
        agent2 = GeminiAgentController(max_profiles=15, config=FAST_CONFIG)
        assert agent2.config == FAST_CONFIG
        assert agent2.max_profiles == 15
        
        # Test conservative config
        agent3 = GeminiAgentController(max_profiles=5, config=CONSERVATIVE_CONFIG)
        assert agent3.config == CONSERVATIVE_CONFIG
        
        print("✅ Configuration integration test passed")
        print(f"🔧 Tested configs: DEFAULT, FAST, CONSERVATIVE")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and return overall result"""
    print("🚀 Starting Gemini Agent Integration Tests")
    print("=" * 50)
    
    tests = [
        test_gemini_agent_initialization,
        test_tool_schema,
        test_mock_decision_making,
        test_tool_execution_mock,
        test_configuration_integration
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Gemini agent is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
