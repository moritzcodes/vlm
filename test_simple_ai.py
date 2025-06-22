#!/usr/bin/env python3
"""
Simple test to debug Vertex AI responses
"""

import cv2
import os
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
if not google_project:
    print("❌ GOOGLE_CLOUD_PROJECT not found")
    exit(1)

print(f"🔑 Using project: {google_project}")

import vertexai
from vertexai.generative_models import GenerativeModel, Part

print("🤖 Loading Vertex AI...")
try:
    vertexai.init(project=google_project, location="us-central1")
    model = GenerativeModel('gemini-1.5-pro')
    print("✅ Vertex AI loaded")
except Exception as e:
    print(f"❌ Vertex AI failed: {e}")
    exit(1)

def test_camera_and_ai():
    print("📱 Testing camera...")
    cap = cv2.VideoCapture(0)  # iPhone camera
    
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame")
        cap.release()
        return
    
    print(f"✅ Frame captured: {frame.shape}")
    
    try:
        # Convert frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()
        print(f"✅ Image converted to bytes: {len(img_bytes)} bytes")
        
        # Create Vertex AI Part
        image_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
        print("✅ Vertex AI Part created")
        
        # Test simple prompt
        print("🔍 Testing simple prompt...")
        response = model.generate_content(["What do you see?", image_part])
        
        print("🔍 RAW RESPONSE:")
        print("=" * 60)
        print(f"Text: '{response.text}'")
        print(f"Length: {len(response.text) if response.text else 0}")
        print(f"Type: {type(response.text)}")
        print("=" * 60)
        
        if hasattr(response, 'candidates'):
            print(f"Candidates: {len(response.candidates)}")
            for i, candidate in enumerate(response.candidates):
                print(f"  Candidate {i}: {candidate}")
        
        # Test structured prompt
        print("\n🔍 Testing structured prompt...")
        structured_response = model.generate_content([
            "Describe what you see. Format: SCENE: [description]", 
            image_part
        ])
        
        print("🔍 STRUCTURED RESPONSE:")
        print("=" * 60)
        print(f"Text: '{structured_response.text}'")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ AI Error: {e}")
        import traceback
        traceback.print_exc()
    
    cap.release()
    print("✅ Test complete")

if __name__ == "__main__":
    test_camera_and_ai() 