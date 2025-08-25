# app/main_agent.py

"""
Main entry point for the LangGraph-based Hinge automation agent.
This replaces the original main.py with a more robust, agent-based approach.
"""

import asyncio
import argparse
from typing import Dict, Any, Optional

from langgraph_agent import HingeAutomationAgent
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
    print("🎉 HINGE AUTOMATION SESSION SUMMARY")
    print("="*60)
    print(f"📊 Profiles Processed: {result.get('profiles_processed', 0)}")
    print(f"🏁 Completion Reason: {result.get('completion_reason', 'Unknown')}")
    
    final_state = result.get('final_state', {})
    if final_state:
        error_count = final_state.get('error_count', 0)
        stuck_count = final_state.get('stuck_count', 0)
        
        print(f"❌ Total Errors: {error_count}")
        print(f"🔒 Times Stuck: {stuck_count}")
        print(f"🔄 Last Action: {final_state.get('last_action', 'Unknown')}")
        
        if final_state.get('action_successful', False):
            print("✅ Final Action: Successful")
        else:
            print("⚠️  Final Action: Failed")
    
    print("="*60)


async def main():
    """Main entry point for the agent"""
    print("🚀 Starting Hinge Automation Agent (LangGraph)")
    print("="*50)
    
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
        print()
        
        # Create and run agent
        agent = HingeAutomationAgent(
            max_profiles=config.max_profiles,
            config=config
        )
        
        # Run automation
        print("🎬 Starting automation workflow...")
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
        print(f"Failed to run async main: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_sync()
    exit(exit_code)