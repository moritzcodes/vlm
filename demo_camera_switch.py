#!/usr/bin/env python3
"""
Demo: Camera Switching for VLM Object Detection
Shows how to easily switch between iPhone and desktop cameras
"""

import os
import json
from camera_switcher import CameraSwitcher

def main():
    print("🎯 VLM Camera Switching Demo")
    print("=" * 50)
    
    # Initialize camera switcher
    switcher = CameraSwitcher()
    
    # Show available cameras
    print("\n1. 📷 Available Cameras:")
    switcher.list_cameras()
    
    # Switch to iPhone camera
    print("\n2. 📱 Switching to iPhone Camera:")
    iphone_camera = switcher.switch_to_iphone()
    
    if iphone_camera is not None:
        print(f"✅ Successfully switched to iPhone Camera {iphone_camera}")
        
        # Show saved configuration
        if os.path.exists('camera_config.json'):
            with open('camera_config.json', 'r') as f:
                config = json.load(f)
            print(f"💾 Saved configuration: {config}")
        
        print(f"\n🚀 Ready to use iPhone camera with VLM:")
        print(f"   python main.py --mode test --camera-index {iphone_camera}")
        print(f"   python streamlit_app.py  # Web interface")
        print(f"   python quick_launch.py   # Interactive launcher")
    
    # Show desktop camera option
    print(f"\n3. 💻 Desktop Camera Alternative:")
    desktop_cameras = switcher.get_desktop_cameras()
    if desktop_cameras:
        print(f"   Available desktop cameras: {desktop_cameras}")
        print(f"   To switch: python camera_switcher.py desktop")
    else:
        print("   No desktop cameras detected")
    
    print(f"\n4. 🎮 Quick Commands:")
    print(f"   📱 Switch to iPhone:  python camera_switcher.py iphone")
    print(f"   💻 Switch to Desktop: python camera_switcher.py desktop")
    print(f"   🔍 List cameras:      python camera_switcher.py list")
    print(f"   🧪 Test camera:       python camera_switcher.py test 0")
    print(f"   🚀 Quick launcher:    python quick_launch.py")
    
    print(f"\n✅ Camera switching demo completed!")
    print(f"Your iPhone camera is ready for VLM object detection! 📱🤖")

if __name__ == "__main__":
    main() 