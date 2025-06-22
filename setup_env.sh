#!/bin/bash

# VLM Environment Setup Helper
echo "🔧 VLM Environment Setup"
echo "========================"
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "📄 Found existing .env file"
    echo "Do you want to update it? (y/n)"
    read -r update_env
    if [ "$update_env" != "y" ]; then
        echo "Using existing .env file..."
        source .env
        echo "✅ Environment loaded from .env"
        exit 0
    fi
fi

echo "🚀 Setting up Vertex AI for Google Gemini 2.5 Pro..."
echo ""

# Get Google Cloud Project ID (required for Vertex AI)
echo "☁️  Google Cloud Project Setup (Required for Vertex AI)..."
echo "Please enter your Google Cloud Project ID:"
echo "(Get it from: https://console.cloud.google.com/)"
read -r project_id

if [ -z "$project_id" ]; then
    echo "❌ Google Cloud Project ID is required for Vertex AI!"
    echo "Please create a project at https://console.cloud.google.com/"
    exit 1
fi

# Get Vertex AI location (optional)
echo ""
echo "🌍 Vertex AI Location (optional)..."
echo "Please enter your preferred Vertex AI location (default: us-central1):"
echo "Common options: us-central1, us-east1, europe-west1, asia-southeast1"
echo "Press Enter for default:"
read -r location
if [ -z "$location" ]; then
    location="us-central1"
fi

# Ask about authentication method
echo ""
echo "🔐 Choose your authentication method:"
echo "1. Application Default Credentials (Recommended for local development)"
echo "2. Google API Key (Alternative method)"
echo "Enter your choice (1 or 2):"
read -r auth_choice

api_key=""
if [ "$auth_choice" = "2" ]; then
    echo ""
    echo "🔑 Setting up Google API Key..."
    echo "Please enter your Google API Key:"
    echo "(Get it from: https://aistudio.google.com/)"
    read -r api_key
    
    if [ -z "$api_key" ]; then
        echo "❌ API key is required for this authentication method!"
        exit 1
    fi
elif [ "$auth_choice" = "1" ]; then
    echo ""
    echo "🔐 Setting up Application Default Credentials..."
    echo "You'll need to authenticate with Google Cloud."
    echo "Run the following command after this setup:"
    echo "  gcloud auth application-default login"
    echo ""
    echo "If you don't have gcloud CLI installed:"
    echo "  brew install google-cloud-sdk"
    echo ""
else
    echo "❌ Invalid choice. Please run the script again."
    exit 1
fi

# Create .env file
echo "💾 Creating .env file..."
cat > .env << EOF
# VLM Environment Configuration
# Generated on $(date)

# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=$project_id
VERTEX_AI_LOCATION=$location
VERTEX_AI_MODEL=gemini-2.5-pro
EOF

if [ -n "$api_key" ]; then
    echo "GOOGLE_API_KEY=$api_key" >> .env
fi

# Add optional settings
cat >> .env << EOF

# Camera settings
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
DETECTION_THRESHOLD=0.5
FRAME_SKIP_RATIO=2

# Performance settings
MAX_CONCURRENT_REQUESTS=3
REQUEST_TIMEOUT=30
LOG_LEVEL=INFO
EOF

echo "✅ Environment file created successfully!"
echo ""
echo "📋 Next steps:"

if [ "$auth_choice" = "1" ]; then
    echo "1. Authenticate with Google Cloud:"
    echo "   gcloud auth application-default login"
    echo ""
    echo "2. Enable required APIs:"
    echo "   gcloud services enable aiplatform.googleapis.com --project=$project_id"
    echo ""
else
    echo "1. Load the environment:"
    echo "   source .env"
    echo ""
fi

echo "3. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Test your setup:"
echo "   python test_setup.py"
echo ""
echo "5. Launch the application:"
echo "   python launch.py"
echo ""

echo "💡 Important Notes:"
echo "- Vertex AI provides better performance than Google AI Studio"
echo "- You'll need billing enabled on your Google Cloud project"
echo "- Gemini 2.0 models are optimized for real-time applications"
echo ""

echo "🆘 Need help?"
echo "- Vertex AI Setup: https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform"
echo "- Billing Setup: https://cloud.google.com/billing/docs/how-to/create-billing-account"
echo "- Project Setup: https://cloud.google.com/resource-manager/docs/creating-managing-projects" 