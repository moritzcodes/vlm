#!/usr/bin/env python3
"""
Ultra Simple iPhone Camera + Gemini
Absolutely minimal - just connect and analyze
"""

import cv2
import os
from PIL import Image
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Simple API key check
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    exit(1)

print("🔑 Found API key")

# Import and configure Gemini
import google.generativeai as genai
genai.configure(api_key=api_key)

# Use basic model name
model = genai.GenerativeModel('gemini-1.5-pro')
print("🤖 Gemini model loaded")

def main():
    print("🚀 Ultra Simple iPhone + Gemini")
    print("Press 'q' to quit, 'a' to analyze")
    
    # Connect iPhone camera
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ Camera failed")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No frames")
        return
    
    print(f"✅ iPhone camera: {frame.shape[1]}x{frame.shape[0]}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow('iPhone Camera', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            print("🔍 Analyzing...")
            try:
                # Convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                
                # Send to Gemini
                response = model.generate_content(["What do you see?", pil_img])
                print("🎯 Result:", response.text)
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done")

if __name__ == "__main__":
    main() 