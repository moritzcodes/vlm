#!/bin/bash

# VLM Setup Script for macOS
# This script creates a virtual environment and installs all dependencies

echo "🚀 Setting up VLM Real-Time Object Detection..."
echo "================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    echo "   You can install it with: brew install python3"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements.txt

# Check if installation was successful
echo "✅ Testing imports..."
python3 -c "import cv2; print('OpenCV:', cv2.__version__)" 2>/dev/null && echo "✅ OpenCV installed" || echo "❌ OpenCV failed"
python3 -c "import google.generativeai; print('✅ Google AI installed')" 2>/dev/null || echo "❌ Google AI failed"
python3 -c "import streamlit; print('✅ Streamlit installed')" 2>/dev/null || echo "❌ Streamlit failed"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set your Google API key:"
echo "   export GOOGLE_API_KEY='your_api_key_here'"
echo ""
echo "3. Run the application:"
echo "   python launch.py"
echo ""
echo "4. Or test your setup:"
echo "   python test_setup.py"
echo ""
echo "💡 Remember to activate the virtual environment each time:"
echo "   source venv/bin/activate" 