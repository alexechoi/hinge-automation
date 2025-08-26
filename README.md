# 🤖 Hinge Automation - AI-Powered Dating App Bot

An intelligent dating app automation system that uses **Google Gemini AI**, **Computer Vision**, and **Android Debug Bridge (ADB)** to automatically analyze profiles, make smart decisions, and send personalized comments on Hinge.

## 🌟 Features

- **🧠 AI-Powered Decision Making**: Multiple agent architectures (Gemini, LangGraph) for intelligent profile analysis
- **👀 Advanced Computer Vision**: Detects UI elements, analyzes profile images, and handles dynamic screens
- **💬 Personalized Comment Generation**: Creates contextual, human-like messages based on profile content
- **🔄 Adaptive Learning**: Tracks success rates and continuously improves comment strategies
- **📱 Full Device Automation**: Handles complex interactions like scrolling, tapping, text input, and verification
- **🎯 Smart Verification**: Uses profile change detection to verify actions succeeded
- **⚡ Multiple Execution Modes**: Choose between different AI agent controllers

## 🛠️ Tech Stack

- **Python 3.13+** with modern dependency management via [uv](https://github.com/astral-sh/uv)
- **Google Gemini 2.5 Flash** for multimodal AI analysis and text generation
- **OpenCV** for computer vision and UI element detection
- **LangGraph** for advanced agent workflows (optional)
- **ADB (Android Debug Bridge)** for device automation
- **Docker** for containerized deployment

## 📋 Requirements

### Hardware & Software
- **Android device** with USB debugging enabled
- **ADB**: Install [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools)
- **Python 3.13+**
- **uv** package manager (recommended) or pip

### API Keys
- **Google Gemini API Key**: Get your free key from [Google AI Studio](https://aistudio.google.com/) - however please note the free key has a very low rate limit that may not be enough for this agent

### Device Setup
1. Enable **Developer Options** on your Android device
2. Enable **USB Debugging** 
3. Authorize your computer when prompted
4. Install and open the **Hinge app**
5. Ensure Hinge is logged in and open on the main stack screen

It is also recommended to turn off auto screen lock for your device so that your device does not lock as the agent works. Also avoid placing the device face down as this can also cause the screen to lock.

## 🚀 Quick Start

### Method 1: Using uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/alexechoi/hinge-automation.git
cd hinge-automation

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies and create virtual environment
cd app/
uv sync

# 4. Configure your Gemini API key
echo "GEMINI_API_KEY=your-gemini-api-key-here" > .env

# 5. Verify device connection
adb devices  # Should show your connected device

# 6. Test the setup
uv run python test_gemini_agent.py

# 7. Run the automation (choose your preferred agent)
uv run python main_agent.py           # Gemini-controlled agent
# or
uv run python langgraph_agent.py      # LangGraph-based agent
```

## 📁 Project Structure

```
hinge-automation/
├── app/
│   ├── main_agent.py              # 🎯 Main Gemini-controlled agent (recommended)
│   ├── langgraph_agent.py         # 🔄 Advanced LangGraph workflow agent
│   ├── gemini_agent_controller.py # 🤖 Core Gemini agent implementation
│   ├── gemini_analyzer.py         # 🧠 AI analysis functions (OCR, decision making)
│   ├── helper_functions.py        # 📱 ADB automation & computer vision utilities
│   ├── agent_config.py            # ⚙️  Agent configuration presets
│   ├── config.py                  # 🔧 API keys and settings management
│   ├── data_store.py              # 💾 Comment storage and success tracking
│   ├── prompt_engine.py           # 📝 Comment generation and template management
│   ├── server.py                  # 🌐 Optional web server interface
│   ├── ocr_extractor.py           # 👁️  Legacy OCR functions functionality
│   ├── pyproject.toml             # 📦 uv/Python project configuration
│   ├── uv.lock                    # 🔒 Dependency lock file
│   ├── generated_comments.json    # 💬 Stored comment history and analytics
│   └── images/                    # 📸 Screenshot storage for debugging
├── docker/
│   └── Dockerfile                 # 🐳 Docker container configuration
├── README.md                      # 📖 This file
└── GEMINI_SETUP.md               # 🔧 Gemini migration guide (legacy)
```

## 🎮 Usage & Configuration

### Command Line Options

```bash
# Basic usage with default settings
uv run python main_agent.py

# Process 20 profiles with verbose logging
uv run python main_agent.py --profiles 20 --verbose

# Use fast configuration preset
uv run python main_agent.py --config fast --profiles 5
```

### Agent Types

1. **Gemini Agent** (`main_agent.py`): 
   - Single AI agent that makes all decisions
   - Simpler architecture, faster execution
   - Best for most users

2. **LangGraph Agent** (`langgraph_agent.py`):
   - Complex workflow with multiple specialized nodes
   - More sophisticated decision making
   - Better for advanced users and experimentation

## 🧠 How It Works

### 1. Intelligent Screenshot Analysis
- Captures device screenshots using ADB
- Uses Gemini's multimodal AI to extract profile text and analyze images
- Detects UI elements (buttons, text fields) with computer vision

### 2. Smart Decision Making  
- Analyzes profile quality, interests, and compatibility signals
- Makes like/dislike decisions based on configurable criteria
- Handles edge cases and error conditions gracefully

### 3. Profile Change Verification
- **Key Innovation**: Verifies actions by detecting profile changes rather than UI elements
- Compares profile text, names, ages, and interests to determine if navigation succeeded
- Much more reliable than traditional UI-based verification

### 4. Personalized Comment Generation
- Generates contextual comments based on profile content
- Adapts style and tone based on success rate analytics
- Stores comments and tracks performance for continuous improvement

### 5. Robust Error Handling
- Automatic recovery from stuck states using swipe patterns
- Multiple retry mechanisms for failed actions
- Comprehensive logging for debugging

## 🔧 Troubleshooting

### Common Issues

**Device Connection**
```bash
# Check if device is connected
adb devices

# Restart ADB server if needed
adb kill-server && adb start-server
```
- Also ensure that the Hinge app is already open
- Do not put your phone face down this sometimes enables device lock on some devices
- Ensure you have granted your computer access on the device
- Remove screen timeout as the agent will stop working if the device locks

**API Key Issues**
```bash
# Verify your .env file
cat .env
# Should show: GEMINI_API_KEY=your-actual-key-here

# Test Gemini connection
uv run python test_gemini_agent.py
```

**Dependency Issues**
```bash
# Reinstall dependencies with uv
uv sync --reinstall

# Or with pip
pip install -r requirements.txt --force-reinstall
```

**Image Directory Missing**
- The system automatically creates `images/` directory for screenshots
- If you see file path errors, ensure write permissions in the app directory

### Debug Mode

Enable verbose logging to see detailed execution steps:
```bash
uv run python main_agent.py --verbose
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes following the existing code patterns
4. **Test** your changes: `uv run python test_gemini_agent.py`
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to the branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Development Setup

```bash
# Clone for development
git clone https://github.com/alexechoi/hinge-automation.git
cd hinge-automation/app/

# Install dev dependencies
uv sync --dev
```

## Recommend device config 

- I recommend setting your Android device to have no screen timeout
- Make sure you open Hinge before starting the Agent
- Ensure ADB is enabled in the developer settings
- Turn on do not disturb to reduce the likelihood of other notifications disturbing the agent

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

The authors are not responsible for any misuse of this software or violations of platform terms of service.

## 🎯 Performance & Analytics

The system tracks and displays:
- **Profiles processed**: Total number of profiles analyzed
- **Success rates**: Like/comment success percentages  
- **Error handling**: Automatic recovery from failed states
- **Comment analytics**: Performance of different comment styles

---

**Built with ❤️ and 🤖 AI**