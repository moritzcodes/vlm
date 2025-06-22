# VLM Real-Time Object Detection 🚀

Real-time object detection using **Google's Vertex AI Gemini 2.5 Pro** with iPhone camera support. This application processes live camera feeds to detect and describe objects in real-time using advanced Vision Language Models.

## ✨ Features

- **🤖 Google Vertex AI Integration**: Uses Gemini 2.5 Pro via Vertex AI for superior performance
- **📱 iPhone Camera Support**: Connect your iPhone camera via USB for high-quality video input
- **⚡ Real-time Processing**: Optimized for low-latency object detection (<200ms)
- **🎯 Advanced Detection**: Multiple detection modes (realtime, detailed, structured)
- **📊 Performance Analytics**: Real-time performance monitoring and statistics
- **🖥️ Dual Interface**: Both web (Streamlit) and command-line interfaces
- **🔧 Flexible Configuration**: Comprehensive settings for optimization
- **📈 Visualization**: Bounding boxes, confidence scores, and detection history

## 🛠️ Quick Setup

### 1. Clone and Install
```bash
git clone <repository-url>
cd vlm
chmod +x setup.sh
./setup.sh
```

### 2. Configure Vertex AI Authentication

**Option A: Application Default Credentials (Recommended)**
```bash
# Install Google Cloud CLI
brew install google-cloud-sdk

# Authenticate
gcloud auth application-default login

# Set up environment
./setup_env.sh
```

**Option B: API Key (Alternative)**
```bash
# Get API key from https://aistudio.google.com/
./setup_env.sh
```

### 3. Enable Required APIs
```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com --project=YOUR_PROJECT_ID
```

### 4. Test Your Setup
```bash
source venv/bin/activate
python test_setup.py
```

### 5. Launch the Application
```bash
python launch.py
```

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **OS**: macOS, Linux, or Windows
- **Camera**: USB camera or iPhone (via USB)
- **Memory**: 4GB RAM minimum, 8GB recommended

### Google Cloud Requirements
- **Google Cloud Project** with billing enabled
- **Vertex AI API** enabled
- **Authentication** configured (gcloud or API key)

## 🔧 Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL=gemini-2.0-flash-exp

# Alternative: Google AI Studio (fallback)
GOOGLE_API_KEY=your-api-key

# Camera Settings
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
DETECTION_THRESHOLD=0.5

# Performance Settings
MAX_CONCURRENT_REQUESTS=3
REQUEST_TIMEOUT=30
FRAME_SKIP_RATIO=2
```

### iPhone Camera Setup (macOS)

1. **Connect iPhone**: Use USB cable
2. **Trust Computer**: Accept trust dialog on iPhone
3. **Grant Permissions**: System Settings → Privacy & Security → Camera → Enable Terminal
4. **Test Camera**: Run `python test_setup.py`

## 🚀 Usage

### Web Interface (Streamlit)
```bash
# Launch web interface
streamlit run streamlit_app.py

# Or use the launcher
python launch.py
# Choose option 1: Web Interface
```

### Command Line Interface
```bash
# Real-time detection
python main.py --mode realtime

# Preview mode (no saving)
python main.py --mode preview

# Test mode (single frame)
python main.py --mode test
```

### Programmatic Usage
```python
from vlm_processor import VertexAIVLMProcessor
from camera_utils import CameraHandler
from config import config

# Initialize components
processor = VertexAIVLMProcessor(config)
camera = CameraHandler()

# Process frame
frame = camera.get_latest_frame()
result = processor.process_frame(frame, prompt_type="realtime")

print(f"Detected objects: {result['detection_result'].objects}")
```

## 🎯 Detection Modes

### 1. Real-time Mode
- **Purpose**: Fast object detection for live feeds
- **Speed**: ~100-200ms per frame
- **Output**: Basic object list with descriptions

### 2. Detailed Mode
- **Purpose**: Comprehensive scene analysis
- **Speed**: ~300-500ms per frame
- **Output**: Detailed descriptions, spatial relationships

### 3. Structured Mode
- **Purpose**: JSON-formatted detection data
- **Speed**: ~200-400ms per frame
- **Output**: Structured data with confidence scores

### 4. Objects Mode
- **Purpose**: Simple object enumeration
- **Speed**: ~150-250ms per frame
- **Output**: Comma-separated object list

## 📊 Performance Optimization

### Speed Optimization
```python
# Adjust frame skip ratio
FRAME_SKIP_RATIO=3  # Process every 3rd frame

# Reduce image resolution
CAMERA_WIDTH=640
CAMERA_HEIGHT=480

# Use faster model
VERTEX_AI_MODEL=gemini-2.0-flash-exp
```

### Quality Optimization
```python
# Higher detection threshold
DETECTION_THRESHOLD=0.7

# More detailed prompts
# Use "detailed" or "structured" modes

# Higher resolution
CAMERA_WIDTH=1920
CAMERA_HEIGHT=1080
```

## 🔍 Troubleshooting

### Authentication Issues
```bash
# Check authentication
gcloud auth list

# Re-authenticate
gcloud auth application-default login

# Check project
gcloud config get-value project
```

### Camera Issues
```bash
# Check camera permissions (macOS)
# System Settings → Privacy & Security → Camera

# Test camera access
python -c "import cv2; print(cv2.VideoCapture(0).read())"

# List available cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
    cap.release()
"
```

### Performance Issues
```bash
# Check system resources
python -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
"

# Monitor API quotas
gcloud logging read "resource.type=vertex_ai_endpoint" --limit=10
```

## 🏗️ Architecture

### Core Components

1. **VLM Processor** (`vlm_processor.py`)
   - Vertex AI integration
   - Async processing
   - Response parsing

2. **Camera Handler** (`camera_utils.py`)
   - Multi-backend camera support
   - Frame optimization
   - iPhone connectivity

3. **Configuration** (`config.py`)
   - Environment management
   - Validation
   - Performance tuning

4. **Streamlit App** (`streamlit_app.py`)
   - Web interface
   - Real-time display
   - Performance metrics

5. **CLI Interface** (`main.py`)
   - Command-line processing
   - Batch operations
   - Performance analysis

### Data Flow
```
Camera → Frame Capture → Preprocessing → Vertex AI → Response Processing → Display
```

## 💰 Cost Estimation

### Vertex AI Pricing (Approximate)
- **Gemini 2.0 Flash**: ~$0.25 per 1K requests
- **Real-time usage**: ~$0.15-0.30 per hour
- **Monthly estimate**: $20-50 for regular use

### Optimization Tips
- Use frame skipping (`FRAME_SKIP_RATIO`)
- Batch processing for offline analysis
- Choose appropriate model (Flash vs Pro)
- Monitor usage in Google Cloud Console

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Getting Help
- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub discussions
- **Documentation**: Check the code comments

### Useful Links
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Reference](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)
- [Google Cloud Console](https://console.cloud.google.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 🎉 Acknowledgments

- Google Cloud Vertex AI team for the amazing Gemini models
- OpenCV community for computer vision tools
- Streamlit team for the fantastic web framework

---

**Built with ❤️ using Google Vertex AI Gemini 2.5 Pro** 