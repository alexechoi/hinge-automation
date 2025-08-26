# app/agent_config.py

from dataclasses import dataclass
from typing import Dict, Any


@dataclass 
class AgentConfig:
    """Configuration settings for the Hinge automation agent"""
    
    # Session settings
    max_profiles: int = 10
    max_retries_per_action: int = 3
    max_errors_before_abort: int = 5
    max_stuck_count: int = 3
    
    # Timing settings (in seconds)
    screenshot_delay: float = 1.0
    action_delay: float = 2.0
    navigation_delay: float = 3.0
    text_input_delay: float = 1.5
    keyboard_dismiss_delay: float = 2.0
    
    # Profile analysis thresholds
    quality_threshold_high: int = 8
    quality_threshold_medium: int = 6
    conversation_threshold_high: int = 7
    min_positive_indicators: int = 2
    min_text_length_detailed: int = 200
    min_quality_for_detailed: int = 5
    
    # UI detection confidence thresholds
    min_button_confidence: float = 0.5
    min_ui_confidence: float = 0.7
    retry_confidence_threshold: float = 0.7
    
    # Swipe and tap coordinates (as percentages)
    dislike_button_coords: tuple = (0.15, 0.85)
    navigation_swipe_coords: tuple = (0.15, 0.5, 0.15, 0.375)  # x1, y1, x2, y2
    aggressive_swipe_coords: tuple = (0.9, 0.5, 0.1, 0.3)  # for unstuck
    scroll_area_coords: tuple = (0.5, 0.6)  # center x, y for scrolling
    keyboard_tap_area: tuple = (0.5, 0.25)  # tap outside keyboard
    
    # Fallback coordinates for comment interface
    fallback_comment_coords: tuple = (0.5, 0.75)
    fallback_send_coords: tuple = (0.75, 0.82)
    
    # Scroll settings
    max_scroll_attempts: int = 3
    scroll_distance_factor: float = 0.3  # how far to scroll
    
    # Recovery strategies
    enable_aggressive_navigation: bool = True
    enable_back_button_recovery: bool = True
    enable_keyboard_recovery: bool = True
    
    # Comment generation
    default_comment: str = "Hey, I'd love to meet up!"
    comment_style: str = "balanced"  # comedic, flirty, straightforward, balanced
    
    # Debug settings
    save_screenshots: bool = True
    screenshot_dir: str = "images"
    verbose_logging: bool = True
    
    # Device settings
    device_ip: str = "127.0.0.1"
    adb_port: int = 5037
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# Default configuration
DEFAULT_CONFIG = AgentConfig()

# High-performance configuration (faster, less thorough)
FAST_CONFIG = AgentConfig(
    max_profiles=20,
    max_retries_per_action=2,
    screenshot_delay=0.5,
    action_delay=1.0,
    navigation_delay=2.0,
    max_scroll_attempts=2,
    quality_threshold_high=7,
    quality_threshold_medium=5,
    conversation_threshold_high=6
)

# Conservative configuration (slower, more thorough)
CONSERVATIVE_CONFIG = AgentConfig(
    max_profiles=5,
    max_retries_per_action=5,
    max_errors_before_abort=8,
    screenshot_delay=2.0,
    action_delay=3.0,
    navigation_delay=4.0,
    max_scroll_attempts=4,
    quality_threshold_high=9,
    quality_threshold_medium=7,
    conversation_threshold_high=8,
    min_positive_indicators=3
)