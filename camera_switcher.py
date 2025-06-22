#!/usr/bin/env python3
"""
Camera Switcher Utility
Easy switching between desktop and iPhone cameras for VLM object detection
"""

import cv2
import sys
import os
import json
from typing import Dict, List, Optional, Tuple
from config import config

class CameraSwitcher:
    """Utility to detect, list, and switch between available cameras"""
    
    def __init__(self):
        self.cameras = {}
        self.current_camera = None
        self.config_file = "camera_config.json"
        
    def detect_cameras(self) -> Dict[int, Dict]:
        """Detect all available cameras and categorize them"""
        print("🔍 Detecting available cameras...")
        cameras = {}
        
        # Check only first 5 camera indices to avoid excessive warnings
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Give camera time to initialize
                import time
                time.sleep(0.1)
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend = cap.getBackendName()
                    
                    # Determine camera type based on resolution and characteristics
                    camera_type = self._identify_camera_type(width, height, i)
                    
                    cameras[i] = {
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps,
                        'backend': backend,
                        'type': camera_type,
                        'resolution': f"{width}x{height}",
                        'available': True
                    }
                    
                    print(f"📷 Camera {i}: {width}x{height} @ {fps:.1f}fps - {camera_type}")
                else:
                    # Camera opened but no frame - mark as unavailable
                    cameras[i] = {
                        'index': i,
                        'available': False
                    }
                cap.release()
            else:
                cameras[i] = {
                    'index': i,
                    'available': False
                }
        
        self.cameras = cameras
        return cameras
    
    def _identify_camera_type(self, width: int, height: int, index: int) -> str:
        """Identify camera type based on resolution and index"""
        # iPhone cameras typically have high resolution
        if width >= 1920 and height >= 1080:
            if index == 0:
                return "📱 iPhone Camera (Primary)"
            else:
                return "📱 iPhone Camera (Secondary)"
        
        # MacBook built-in cameras are typically lower resolution
        elif width == 1280 and height == 720:
            return "💻 MacBook FaceTime HD Camera"
        
        # Other common resolutions
        elif width == 640 and height == 480:
            return "💻 Built-in Camera (VGA)"
        
        else:
            return f"📹 External Camera ({width}x{height})"
    
    def list_cameras(self) -> None:
        """List all available cameras with details"""
        if not self.cameras:
            self.detect_cameras()
        
        print("\n" + "="*60)
        print("📷 AVAILABLE CAMERAS")
        print("="*60)
        
        available_cameras = {k: v for k, v in self.cameras.items() if v.get('available', False)}
        
        if not available_cameras:
            print("❌ No cameras detected!")
            return
        
        for index, camera in available_cameras.items():
            print(f"\n🎯 Camera {index}:")
            print(f"   Type: {camera['type']}")
            print(f"   Resolution: {camera['resolution']}")
            print(f"   FPS: {camera['fps']:.1f}")
            print(f"   Backend: {camera['backend']}")
        
        print("\n" + "="*60)
    
    def get_iphone_cameras(self) -> List[int]:
        """Get list of iPhone camera indices"""
        if not self.cameras:
            self.detect_cameras()
        
        iphone_cameras = []
        for index, camera in self.cameras.items():
            if camera.get('available') and "iPhone" in camera.get('type', ''):
                iphone_cameras.append(index)
        
        return iphone_cameras
    
    def get_desktop_cameras(self) -> List[int]:
        """Get list of desktop/built-in camera indices"""
        if not self.cameras:
            self.detect_cameras()
        
        desktop_cameras = []
        for index, camera in self.cameras.items():
            if camera.get('available') and ("MacBook" in camera.get('type', '') or "Built-in" in camera.get('type', '')):
                desktop_cameras.append(index)
        
        return desktop_cameras
    
    def switch_to_iphone(self) -> Optional[int]:
        """Switch to iPhone camera (returns camera index)"""
        iphone_cameras = self.get_iphone_cameras()
        
        if not iphone_cameras:
            print("❌ No iPhone cameras detected!")
            return None
        
        # Use the first iPhone camera (usually index 0)
        camera_index = iphone_cameras[0]
        self.current_camera = camera_index
        
        print(f"📱 Switched to iPhone Camera (Index: {camera_index})")
        self._update_config(camera_index, "iphone")
        
        return camera_index
    
    def switch_to_desktop(self) -> Optional[int]:
        """Switch to desktop/built-in camera (returns camera index)"""
        desktop_cameras = self.get_desktop_cameras()
        
        if not desktop_cameras:
            print("❌ No desktop cameras detected!")
            return None
        
        # Use the first desktop camera
        camera_index = desktop_cameras[0]
        self.current_camera = camera_index
        
        print(f"💻 Switched to Desktop Camera (Index: {camera_index})")
        self._update_config(camera_index, "desktop")
        
        return camera_index
    
    def switch_to_camera(self, camera_index: int) -> bool:
        """Switch to specific camera by index"""
        if not self.cameras:
            self.detect_cameras()
        
        if camera_index not in self.cameras or not self.cameras[camera_index].get('available'):
            print(f"❌ Camera {camera_index} is not available!")
            return False
        
        self.current_camera = camera_index
        camera_info = self.cameras[camera_index]
        
        print(f"🎯 Switched to Camera {camera_index}: {camera_info['type']}")
        self._update_config(camera_index, "custom")
        
        return True
    
    def _update_config(self, camera_index: int, camera_type: str) -> None:
        """Update configuration with selected camera"""
        camera_config = {
            'current_camera': camera_index,
            'camera_type': camera_type,
            'timestamp': str(os.popen('date').read().strip())
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(camera_config, f, indent=2)
            print(f"✅ Camera configuration saved to {self.config_file}")
        except Exception as e:
            print(f"⚠️ Could not save camera configuration: {e}")
    
    def load_saved_camera(self) -> Optional[int]:
        """Load previously saved camera configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    camera_config = json.load(f)
                return camera_config.get('current_camera')
        except Exception as e:
            print(f"⚠️ Could not load camera configuration: {e}")
        return None
    
    def test_camera(self, camera_index: int, duration: int = 5) -> bool:
        """Test camera by capturing frames for specified duration"""
        # Ensure cameras are detected first
        if not self.cameras:
            self.detect_cameras()
            
        if camera_index not in self.cameras or not self.cameras[camera_index].get('available'):
            print(f"❌ Camera {camera_index} is not available!")
            return False
        
        print(f"🧪 Testing Camera {camera_index} for {duration} seconds...")
        print("Press 'q' to quit early")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cameras[camera_index]['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cameras[camera_index]['height'])
        
        frame_count = 0
        import time
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    cv2.imshow(f'Camera {camera_index} Test', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("❌ Failed to capture frame")
                    break
            
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            
            print(f"✅ Camera test completed:")
            print(f"   Frames captured: {frame_count}")
            print(f"   Duration: {elapsed_time:.2f}s")
            print(f"   Average FPS: {fps:.2f}")
            
            return True
            
        except KeyboardInterrupt:
            print("\n🛑 Camera test interrupted by user")
            return True
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Main function for command-line usage"""
    switcher = CameraSwitcher()
    
    if len(sys.argv) < 2:
        print("📱 Camera Switcher - Easy camera switching for VLM")
        print("\nUsage:")
        print("  python camera_switcher.py list          - List all cameras")
        print("  python camera_switcher.py iphone        - Switch to iPhone camera")
        print("  python camera_switcher.py desktop       - Switch to desktop camera")
        print("  python camera_switcher.py switch <index> - Switch to specific camera")
        print("  python camera_switcher.py test <index>   - Test specific camera")
        print("  python camera_switcher.py detect        - Detect all cameras")
        return
    
    command = sys.argv[1].lower()
    
    if command == "list":
        switcher.list_cameras()
    
    elif command == "iphone":
        camera_index = switcher.switch_to_iphone()
        if camera_index is not None:
            print(f"\n🚀 Ready to use iPhone camera with VLM:")
            print(f"   python main.py --mode realtime --camera-index {camera_index}")
    
    elif command == "desktop":
        camera_index = switcher.switch_to_desktop()
        if camera_index is not None:
            print(f"\n🚀 Ready to use desktop camera with VLM:")
            print(f"   python main.py --mode realtime --camera-index {camera_index}")
    
    elif command == "switch" and len(sys.argv) > 2:
        try:
            camera_index = int(sys.argv[2])
            if switcher.switch_to_camera(camera_index):
                print(f"\n🚀 Ready to use camera {camera_index} with VLM:")
                print(f"   python main.py --mode realtime --camera-index {camera_index}")
        except ValueError:
            print("❌ Invalid camera index. Please provide a number.")
    
    elif command == "test" and len(sys.argv) > 2:
        try:
            camera_index = int(sys.argv[2])
            switcher.test_camera(camera_index)
        except ValueError:
            print("❌ Invalid camera index. Please provide a number.")
    
    elif command == "detect":
        switcher.detect_cameras()
        switcher.list_cameras()
    
    else:
        print("❌ Invalid command. Use 'python camera_switcher.py' for help.")

if __name__ == "__main__":
    main() 