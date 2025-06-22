#!/usr/bin/env python3
"""
🎯 FINAL Real-time Thumbs Up Detector with Gemini 2.5 Pro
- Auto-detects iPhone camera
- Real-time scene description  
- Thumbs up gesture detection
- Works with Vertex AI
"""

import cv2
import os
import time
import threading
import queue
import io
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def find_iphone_camera():
    """Auto-detect iPhone camera index"""
    print("🔍 Searching for iPhone camera...")
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                # iPhone cameras are typically high resolution
                if width >= 1920 and height >= 1080:
                    cap.release()
                    print(f"📱 iPhone camera found at index {i}: {width}x{height}")
                    return i
            cap.release()
    
    print("❌ No iPhone camera found")
    return None

def setup_ai_model():
    """Setup Vertex AI Gemini"""
    google_project = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    if not google_project:
        print("❌ GOOGLE_CLOUD_PROJECT not found in .env file")
        print("Set up Vertex AI by adding: GOOGLE_CLOUD_PROJECT=your_project_id")
        return None
    
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        vertexai.init(project=google_project, location="us-central1")
        model = GenerativeModel('gemini-1.5-pro')
        print("🤖 Vertex AI Gemini 1.5 Pro loaded")
        return model
        
    except Exception as e:
        print(f"❌ Vertex AI setup failed: {e}")
        return None

class ThumbsUpDetector:
    def __init__(self):
        self.cap = None
        self.model = None
        self.running = False
        self.analysis_queue = queue.Queue(maxsize=1)
        self.last_analysis_time = 0
        self.analysis_interval = 2.0  # Analyze every 2 seconds
        self.thumbs_up_detected = False
        self.last_description = "Starting up..."
        self.camera_index = None
        
    def setup(self):
        """Setup detector components"""
        # Find iPhone camera
        self.camera_index = find_iphone_camera()
        if self.camera_index is None:
            return False
        
        # Setup AI model
        self.model = setup_ai_model()
        if self.model is None:
            return False
        
        return True
        
    def connect_camera(self):
        """Connect to iPhone camera"""
        print(f"📱 Connecting to iPhone camera at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("❌ Camera connection failed")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Cannot capture frames")
            return False
        
        print(f"✅ Connected: {frame.shape[1]}x{frame.shape[0]}")
        return True
    
    def analyze_frame_with_gemini(self, frame):
        """Send frame to Gemini for analysis"""
        try:
            # Convert frame to Vertex AI format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Convert to bytes for Vertex AI
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=85)
            img_bytes = img_buffer.getvalue()
            
            # Create Vertex AI Part
            from vertexai.generative_models import Part
            image_part = Part.from_data(data=img_bytes, mime_type="image/jpeg")
            
            # Clear, simple prompt
            prompt = """Analyze this image:

1. Describe what you see (people, objects, actions)
2. Check for thumbs up gestures (YES/NO)

Respond exactly like this:
SCENE: [brief description]
THUMBS_UP: [YES or NO]"""

            response = self.model.generate_content([prompt, image_part])
            return response.text
            
        except Exception as e:
            return f"Error: {str(e)[:50]}"
    
    def parse_response(self, response_text):
        """Parse Gemini response"""
        scene = "Analyzing..."
        thumbs_up = False
        
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('SCENE:'):
                    scene = line.replace('SCENE:', '').strip()
                elif line.startswith('THUMBS_UP:'):
                    thumbs_status = line.replace('THUMBS_UP:', '').strip().upper()
                    thumbs_up = thumbs_status == "YES"
            
            # If no structured response, use full text as scene
            if scene == "Analyzing..." and not response_text.startswith("Error"):
                scene = response_text[:100] + "..." if len(response_text) > 100 else response_text
                
        except Exception as e:
            scene = f"Parse error: {e}"
            
        return scene, thumbs_up
    
    def analysis_worker(self):
        """Background thread for AI analysis"""
        while self.running:
            try:
                if not self.analysis_queue.empty():
                    frame = self.analysis_queue.get(timeout=1)
                    
                    print("🧠 Analyzing with Gemini...")
                    response = self.analyze_frame_with_gemini(frame)
                    
                    # Parse response
                    scene, is_thumbs_up = self.parse_response(response)
                    
                    # Update state
                    self.last_description = scene
                    
                    # Handle thumbs up detection
                    if is_thumbs_up and not self.thumbs_up_detected:
                        print("\n" + "🎉" * 20)
                        print("👍 THUMBS UP DETECTED! 👍")
                        print("🎉" * 20 + "\n")
                        self.thumbs_up_detected = True
                    elif not is_thumbs_up:
                        self.thumbs_up_detected = False
                    
                    # Show results
                    print(f"📝 Scene: {scene}")
                    status = "👍 THUMBS UP!" if is_thumbs_up else "👀 Monitoring..."
                    print(f"🎯 Status: {status}")
                    print("-" * 50)
                    
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                time.sleep(1)
    
    def run(self):
        """Main detection loop"""
        if not self.setup():
            print("❌ Setup failed")
            return
            
        if not self.connect_camera():
            print("❌ Camera setup failed")
            return
        
        print("\n🚀 THUMBS UP DETECTOR READY!")
        print("👍 Hold up your thumb to trigger detection")
        print("📱 Real-time scene analysis every 2 seconds")
        print("Press 'q' to quit, 's' for immediate analysis")
        print("=" * 60)
        
        self.running = True
        
        # Start AI analysis thread
        ai_thread = threading.Thread(target=self.analysis_worker, daemon=True)
        ai_thread.start()
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera disconnected")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Add overlay
                self.add_overlay(frame, frame_count)
                
                # Show live feed
                cv2.imshow('🎯 Thumbs Up Detector - iPhone Camera', frame)
                
                # Queue frame for analysis
                if (current_time - self.last_analysis_time) >= self.analysis_interval:
                    try:
                        # Clear queue and add fresh frame
                        while not self.analysis_queue.empty():
                            try:
                                self.analysis_queue.get_nowait()
                            except queue.Empty:
                                break
                        
                        self.analysis_queue.put_nowait(frame.copy())
                        self.last_analysis_time = current_time
                        
                    except queue.Full:
                        pass
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Force immediate analysis
                    try:
                        self.analysis_queue.put_nowait(frame.copy())
                        print("🔍 Immediate analysis requested...")
                    except queue.Full:
                        print("⚠️ Analysis busy, try again...")
        
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
        
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Thumbs Up Detector stopped")
    
    def add_overlay(self, frame, frame_count):
        """Add status overlay to video"""
        height, width = frame.shape[:2]
        
        # Status colors
        if self.thumbs_up_detected:
            color = (0, 255, 0)  # Green for thumbs up
            status = "👍 THUMBS UP!"
        else:
            color = (255, 255, 255)  # White for monitoring
            status = "👀 Monitoring..."
        
        # Main status box
        cv2.rectangle(frame, (10, 10), (600, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (600, 130), color, 3)
        
        # Title
        cv2.putText(frame, "🎯 Thumbs Up Detector", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status
        cv2.putText(frame, status, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_count} | Gemini 1.5 Pro", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, "Press 'q' to quit, 's' to analyze now", (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Scene description at bottom
        if self.last_description:
            desc = self.last_description[:70] + "..." if len(self.last_description) > 70 else self.last_description
            cv2.rectangle(frame, (10, height - 50), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, f"📝 {desc}", (20, height - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    print("🎯 Starting Thumbs Up Detector with Gemini 2.5 Pro")
    detector = ThumbsUpDetector()
    detector.run()

if __name__ == "__main__":
    main() 