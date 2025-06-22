# VLM Quick Start Guide 🚀

Your VLM Real-Time Object Detection system is ready! Here's how to get started:

## ✅ Setup Status
- ✅ Virtual environment created
- ✅ All Python packages installed  
- ✅ Dependencies verified
- ⚠️ Camera permission needed (see step 2 below)

## 🚀 Quick Start (3 Steps)

### Step 1: Get Your Google API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API key" 
3. Create a new project or use existing
4. Copy your API key

### Step 2: Grant Camera Permission
**macOS Users:** You need to grant camera access to Terminal:
1. Open **System Settings** → **Privacy & Security** → **Camera** 
2. Find and enable **Terminal** (or **iTerm** if using iTerm)
3. Restart your terminal application

### Step 3: Launch the Application
```bash
# Always activate the virtual environment first
source venv/bin/activate

# Set your API key (replace with your actual key)
export GOOGLE_API_KEY='your_actual_api_key_here'

# Launch the interactive menu
python launch.py
```

## 🎯 Application Options

From the launch menu, you can choose:

### Option 1: Streamlit Web Interface (Recommended)
- Beautiful web interface with real-time video feed
- Performance metrics and controls
- Perfect for experimentation and demos
- Access at `http://localhost:8501`

### Option 2: Command Line Interface
- Lightweight terminal-based interface
- Good for testing and automation
- Multiple modes: realtime, test, preview

## 📋 Usage Tips

1. **First Time Setup**: Use the web interface - it's more user-friendly
2. **Camera Issues**: Make sure your iPhone/webcam is connected and permissions are granted
3. **Performance**: Adjust frame skip ratio in settings for better performance
4. **API Costs**: Monitor your usage - each frame processed costs ~$0.001-0.002

## 🔧 Troubleshooting

### Camera Not Working?
- Check System Preferences → Privacy & Security → Camera
- Restart terminal after granting permissions
- Try different camera indices (0, 1, 2) in settings

### API Errors?
- Verify your API key is correct
- Check your Google Cloud billing is enabled
- Ensure you have Gemini API access

### Performance Issues?
- Increase frame skip ratio (process every 3rd or 4th frame)
- Reduce detection frequency
- Use "objects" prompt instead of "detailed" for faster processing

## 🆘 Need Help?

If you encounter issues:
1. Run `python test_setup.py` to diagnose problems
2. Check the full README.md for detailed documentation
3. Ensure all environment variables are set correctly

## 💡 Pro Tips

- Start with the web interface to get familiar with the system
- Use the "objects" detection mode for real-time performance  
- Adjust the confidence threshold to filter detections
- Monitor API usage in the Google Cloud Console

---

**Ready to start?** Run `python launch.py` and have fun with real-time object detection! 🎉 