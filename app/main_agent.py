# app/main_agent.py

"""
Main entry point for the Gemini-controlled Hinge automation agent.
This uses Gemini AI to intelligently select and execute tools for dating app automation.
"""

import asyncio
import argparse
from typing import Dict, Any, Optional

from gemini_agent_controller import GeminiAgentController
from agent_config import AgentConfig, DEFAULT_CONFIG, FAST_CONFIG, CONSERVATIVE_CONFIG


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hinge Automation Agent")
    
    parser.add_argument(
        "--profiles", "-p",
        type=int,
        default=10,
        help="Maximum number of profiles to process (default: 10)"
    )
    
    parser.add_argument(
        "--config", "-c",
        choices=["default", "fast", "conservative"],
        default="default",
        help="Configuration preset to use (default: default)"
    )
    
    parser.add_argument(
        "--device-ip",
        type=str,
        default="127.0.0.1", 
        help="Device IP address (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Disable screenshot saving"
    )
    
    return parser.parse_args()


def get_config(config_name: str, args) -> AgentConfig:
    """Get configuration based on name and override with args"""
    
    configs = {
        "default": DEFAULT_CONFIG,
        "fast": FAST_CONFIG,
        "conservative": CONSERVATIVE_CONFIG
    }
    
    config = configs[config_name]
    
    # Override with command line arguments
    config.max_profiles = args.profiles
    config.device_ip = args.device_ip
    config.verbose_logging = args.verbose
    config.save_screenshots = not args.no_screenshots
    
    return config


def print_session_summary(result: Dict[str, Any]):
    """Print a summary of the automation session"""
    print("\n" + "="*60)
    print("🤖 GEMINI-CONTROLLED HINGE AUTOMATION SUMMARY")
    print("="*60)
    print(f"📊 Profiles Processed: {result.get('profiles_processed', 0)}")
    print(f"💖 Likes Sent: {result.get('likes_sent', 0)}")
    print(f"💬 Comments Sent: {result.get('comments_sent', 0)}")
    print(f"❌ Errors Encountered: {result.get('errors_encountered', 0)}")
    print(f"🏁 Completion Reason: {result.get('completion_reason', 'Unknown')}")
    
    if result.get('final_success_rates'):
        print(f"📈 Final Success Rates: {result['final_success_rates']}")
    
    success = result.get('success', False)
    if success:
        print("✅ Session: Completed Successfully")
    else:
        print("⚠️  Session: Completed with Issues")
    
    print("="*60)


async def main():
    """Main entry point for the Gemini-controlled agent"""
    print("🤖 Starting Gemini-Controlled Hinge Automation Agent")
    print("="*55)
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Get configuration  
        config = get_config(args.config, args)
        
        print(f"📋 Configuration: {args.config}")
        print(f"📱 Device IP: {config.device_ip}")
        print(f"🎯 Max Profiles: {config.max_profiles}")
        print(f"🔊 Verbose Logging: {config.verbose_logging}")
        print(f"📸 Save Screenshots: {config.save_screenshots}")
        print(f"🤖 AI Controller: Google Gemini")
        print()
        
        # Create and run Gemini-controlled agent
        agent = GeminiAgentController(
            max_profiles=config.max_profiles,
            config=config
        )
        
        # Run automation
        print("🎬 Starting Gemini-powered automation workflow...")
        print("🧠 Gemini will analyze screenshots and intelligently select tools...")
        result = agent.run_automation()
        
        # Print summary
        print_session_summary(result)
        
        # Return success
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Automation interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Automation failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        if args.verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        
        return 1


def run_sync():
    """Synchronous wrapper for the main function"""
    try:
        return asyncio.run(main())
    except Exception as e:
        print(f"Failed to run Gemini-controlled automation: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_sync()
    exit(exit_code)