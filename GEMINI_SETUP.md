# Gemini API Setup Guide

The application has been updated to use Google's Gemini API instead of OpenCV + Tesseract for OCR and image analysis.

## Setup Steps

1. **Install Dependencies**
   ```bash
   cd app/
   pip install -r requirements.txt
   ```

2. **Get Gemini API Key**
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Generate an API key for Gemini
   - Keep your API key secure

3. **Configure Environment Variables**
   Add to your `.env` file:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Test the Integration**
   ```bash
   python test_gemini.py
   ```

## Changes Made

- **Added**: `gemini_analyzer.py` - New module for Gemini-based image analysis
- **Updated**: `main.py` - Now uses Gemini API instead of OpenCV OCR
- **Updated**: `requirements.txt` - Added `google-genai` dependency
- **Updated**: `config.py` - Added `GEMINI_API_KEY` configuration
- **Removed**: Tesseract OCR dependency from `helper_functions.py`

## Benefits

- **Better Text Recognition**: Gemini's multimodal capabilities provide more accurate text extraction
- **Context Understanding**: Can understand image context, not just extract text
- **Profile Analysis**: Can analyze profile content, interests, and sentiment
- **Future Extensibility**: Can easily add more sophisticated image understanding features

## API Usage Limits

- Gemini 2.5 Flash: Cost-effective for high-volume usage
- Request size limit: 20MB for inline images
- For larger files, use the File API (already supported in the code)