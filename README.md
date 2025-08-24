# Pitch Perfect

This project demonstrates how to automate interactions with Hinge (a dating app) using a combination of the following tools and techniques:

- **ADB (Android Debug Bridge)**: Automate device actions such as taps, swipes, and text input.
- **Computer Vision (OpenCV)**: Detect and locate UI elements on the screen using feature-based and template matching methods.
- **AI Image Understanding (Google Gemini)**: Extract and analyze text from screenshots using advanced multimodal AI capabilities.
- **LLM (Google Gemini)**: Generate personalized, human-like comments based on extracted text content.

By integrating these components, the script can make automated decisions (like or dislike profiles) and even respond with a custom-generated pickup line or comment.

> **🆕 Latest Update**: This project has been fully migrated from OpenAI + Tesseract OCR to Google's Gemini API for both image analysis and text generation. See [GEMINI_SETUP.md](GEMINI_SETUP.md) for migration details.

## Demo

[![PitchPerfect Demo: ](https://img.youtube.com/vi/VgES1_QHrR8/maxresdefault.jpg)](https://youtube.com/shorts/VgES1_QHrR8)

## Features

- **🤖 Intelligent Profile Analysis**: Uses Gemini AI to understand profile content, interests, and personality traits
- **💬 Smart Comment Generation**: Creates personalized, contextual messages with different styles (comedic, flirty, straightforward)
- **👀 Advanced Computer Vision**: Detects UI elements and analyzes profile images with high accuracy
- **📱 Full Device Automation**: Handles taps, swipes, text input, and navigation via ADB
- **🔄 Adaptive Learning**: Tracks success rates and adjusts comment generation strategies
- **🐳 Docker Support**: Easy deployment with containerized environment

## Requirements

- **Python 3.x**
- **ADB**:  
  Install the [Android SDK Platform Tools](https://developer.android.com/studio/releases/platform-tools) and ensure `adb` is accessible from your PATH.
- **Device Setup**:
  - Enable Developer Options and USB Debugging on your Android device.
  - Authorize your computer for USB debugging when prompted.
- **Google Gemini API Key**:
  - Get your free API key from [Google AI Studio](https://aistudio.google.com/)
  - No credit card required for basic usage

### Dependencies Installation

Install all dependencies at once:
```bash
cd app/
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pure-python-adb opencv-python pillow python-dotenv google-genai spacy textblob vaderSentiment requests
```

**Note**: Tesseract OCR and OpenAI dependencies are no longer required thanks to the Gemini migration.

## Quick Start

### Option 1: Local Setup (Recommended)

1. **Clone and Navigate**:
   ```bash
   git clone <repository-url>
   cd hinge-automation/app
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API Key**:
   Create a `.env` file in the app directory:
   ```env
   GEMINI_API_KEY=your-gemini-api-key-here
   ```
   Get your free API key from [Google AI Studio](https://aistudio.google.com/)

4. **Connect Your Android Device**:
   ```bash
   # Enable USB debugging on your phone, then:
   adb devices  # Should show your device
   ```

5. **Test the Setup**:
   ```bash
   python test_gemini.py  # Test Gemini integration
   ```

6. **Run the Bot**:
   ```bash
   python main.py
   ```

### Option 2: Docker Setup

1. **Build the Docker container**:
   ```bash
   docker build -t my-ocr-bot -f docker/Dockerfile .
   ```

2. **Run the container**:
   ```bash
   docker run my-ocr-bot
   ```

## Troubleshooting

### Docker Debugging
In case of weird behavior, open the container and check what's up:
```bash
docker run -it --entrypoint /bin/bash my-ocr-bot
```

### Wireless ADB Connection
To connect wirelessly from shell:
```bash
adb tcpip 5555
adb connect 192.168.X.Y:5555  # Replace with your phone's IP
```

### Common Issues

- **"No devices connected"**: Make sure USB debugging is enabled and device is authorized
- **"GEMINI_API_KEY not set"**: Check your `.env` file is in the correct directory with the right key
- **Import errors**: Run `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`
- **Docker build fails**: Use `docker/Dockerfile` path (forward slashes, not backslashes)

## Architecture Overview

The bot works in these steps:
1. **Screenshot Capture**: Takes screenshots of the dating app
2. **AI Image Analysis**: Uses Gemini to extract profile text and analyze content  
3. **Decision Making**: Analyzes profile compatibility using computer vision and AI
4. **Comment Generation**: Creates personalized comments using Gemini's language capabilities
5. **Automated Actions**: Executes swipes, likes, and message sending via ADB

## Migration from OpenAI/Tesseract

This project has been fully migrated to Google's Gemini API for better performance and cost-effectiveness:

✅ **Better OCR**: Gemini's multimodal understanding vs. Tesseract  
✅ **Smarter Comments**: Context-aware generation vs. basic GPT prompts  
✅ **Cost Effective**: Single API for both image and text processing  
✅ **No Local Dependencies**: No need to install Tesseract OCR  

See [GEMINI_SETUP.md](GEMINI_SETUP.md) for detailed migration information.

## File Structure

```
hinge-automation/
├── app/
│   ├── main.py              # Main automation script
│   ├── gemini_analyzer.py   # Gemini AI integration
│   ├── helper_functions.py  # ADB and computer vision utilities
│   ├── prompt_engine.py     # Comment generation logic
│   ├── config.py           # Configuration management
│   ├── requirements.txt    # Python dependencies
│   └── test_gemini.py      # Test script for Gemini integration
├── docker/
│   └── Dockerfile          # Docker container configuration
├── README.md               # This file
└── GEMINI_SETUP.md         # Gemini migration guide
```

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing patterns
- New features include appropriate tests
- Documentation is updated for any API changes

## Disclaimer

This project is for educational purposes. Use responsibly and in accordance with the terms of service of any applications you interact with.
