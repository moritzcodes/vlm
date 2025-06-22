#!/usr/bin/env python3
"""
Simple iPhone Camera + Gemini 2.5 Pro Analysis
Just the basics: USB iPhone camera -> Gemini analysis
"""

import cv2
import time
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check API key configuration
google_api_key = os.getenv('GOOGLE_API_KEY')
google_project = os.getenv('GOOGLE_CLOUD_PROJECT')

if google_api_key:
    # Use Google AI Studio API
    print("🔑 Using Google AI Studio API")
    import google.generativeai as genai
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-2.5-pro')
    
elif google_project:
    # Use Vertex AI
    print("🔑 Using Vertex AI")
    import vertexai
    from vertexai.generative_models import GenerativeModel
    vertexai.init(project=google_project, location="us-central1")
    model = GenerativeModel('gemini-2.5-pro')
    
else:
    print("❌ Neither GOOGLE_API_KEY nor GOOGLE_CLOUD_PROJECT found in .env file")
    print("For Google AI Studio: GOOGLE_API_KEY=your_key_here")  
    print("For Vertex AI: GOOGLE_CLOUD_PROJECT=your_project_id")
    exit(1)

def main():
    """Main function - connect iPhone camera and analyze with Gemini"""
    print("🚀 Simple iPhone Camera + Gemini Analysis")
    print("Press 'q' to quit, 'a' to analyze current frame")
    
    # Connect to iPhone camera (index 0)
    print("🔍 Connecting to iPhone camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to connect to camera 0")
        print("Make sure your iPhone is connected via USB and unlocked")
        return
    
    # Test frame capture
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera connected but can't capture frames")
        cap.release()
        return
    
    height, width = frame.shape[:2]
    print(f"✅ iPhone camera connected: {width}x{height}")
    
    try:
        while True:
            # Get frame from camera
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            # Show live feed
            cv2.imshow('iPhone Camera - Press A to Analyze', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('a'):
                print("🔍 Analyzing with Gemini...")
                
                try:
                    # Convert frame for Gemini
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # Analyze with Gemini
                    response = model.generate_content([
                        "What objects do you see in this image? List them clearly.",
                        pil_image
                    ])
                    
                    print("🎯 Gemini Analysis:")
                    print("-" * 50)
                    print(response.text)
                    print("-" * 50)
                    
                    # Save analyzed frame
                    timestamp = int(time.time())
                    filename = f"analyzed_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved: {filename}")
                    
                except Exception as e:
                    print(f"❌ Analysis error: {e}")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Done")

if __name__ == "__main__":
    main() 