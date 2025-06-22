#!/usr/bin/env python3
"""
Quick Launch Script for VLM Object Detection
Easy camera switching and app launching
"""

import sys
import os
import subprocess
from camera_switcher import CameraSwitcher

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🎯 VLM OBJECT DETECTION - QUICK LAUNCHER")
    print("📱 iPhone Camera + 🤖 Gemini 2.5 Pro")
    print("=" * 60)

def print_menu():
    """Print main menu"""
    print("\n📋 QUICK ACTIONS:")
    print("1. 📱 Use iPhone Camera")
    print("2. 💻 Use Desktop Camera") 
    print("3. 🔍 List All Cameras")
    print("4. 🧪 Test Camera")
    print("5. 🎥 Launch Web Interface")
    print("6. ⚡ Launch Real-time Detection")
    print("7. 📸 Single Frame Test")
    print("8. ⚙️  Advanced Options")
    print("9. ❌ Exit")
    print("-" * 60)

def launch_with_camera(camera_index: int, mode: str = "realtime"):
    """Launch VLM with specific camera"""
    print(f"🚀 Launching VLM with Camera {camera_index} in {mode} mode...")
    
    if mode == "web":
        # Launch Streamlit app
        cmd = ["streamlit", "run", "streamlit_app.py", "--server.port", "8501"]
        os.environ['CAMERA_INDEX'] = str(camera_index)
    elif mode == "realtime":
        cmd = ["python", "main.py", "--mode", "realtime", "--camera-index", str(camera_index)]
    elif mode == "test":
        cmd = ["python", "main.py", "--mode", "test", "--camera-index", str(camera_index)]
    else:
        print(f"❌ Unknown mode: {mode}")
        return
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
    except FileNotFoundError:
        print(f"❌ Command not found. Make sure all dependencies are installed.")

def advanced_menu():
    """Show advanced options menu"""
    print("\n⚙️  ADVANCED OPTIONS:")
    print("1. 🔧 Configure Camera Settings")
    print("2. 📊 View System Status")
    print("3. 🧹 Clear Camera Config")
    print("4. 🔄 Refresh Camera Detection")
    print("5. 📋 Export Camera Info")
    print("6. ⬅️  Back to Main Menu")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    switcher = CameraSwitcher()
    
    if choice == "1":
        configure_camera_settings(switcher)
    elif choice == "2":
        show_system_status(switcher)
    elif choice == "3":
        clear_camera_config()
    elif choice == "4":
        switcher.detect_cameras()
        switcher.list_cameras()
    elif choice == "5":
        export_camera_info(switcher)
    elif choice == "6":
        return
    else:
        print("❌ Invalid option")

def configure_camera_settings(switcher: CameraSwitcher):
    """Configure camera settings"""
    print("\n🔧 Camera Configuration:")
    switcher.list_cameras()
    
    try:
        camera_index = int(input("\nEnter camera index to configure: "))
        if switcher.switch_to_camera(camera_index):
            print("✅ Camera configuration updated!")
        else:
            print("❌ Failed to configure camera")
    except ValueError:
        print("❌ Invalid camera index")

def show_system_status(switcher: CameraSwitcher):
    """Show system status"""
    print("\n📊 SYSTEM STATUS:")
    print("-" * 40)
    
    # Check saved camera config
    saved_camera = switcher.load_saved_camera()
    if saved_camera is not None:
        print(f"💾 Saved Camera: {saved_camera}")
    else:
        print("💾 No saved camera configuration")
    
    # Check available cameras
    cameras = switcher.detect_cameras()
    available_count = len([c for c in cameras.values() if c.get('available', False)])
    print(f"📷 Available Cameras: {available_count}")
    
    # Check iPhone cameras
    iphone_cameras = switcher.get_iphone_cameras()
    print(f"📱 iPhone Cameras: {len(iphone_cameras)} ({iphone_cameras})")
    
    # Check desktop cameras  
    desktop_cameras = switcher.get_desktop_cameras()
    print(f"💻 Desktop Cameras: {len(desktop_cameras)} ({desktop_cameras})")
    
    print("-" * 40)

def clear_camera_config():
    """Clear saved camera configuration"""
    config_file = "camera_config.json"
    if os.path.exists(config_file):
        os.remove(config_file)
        print("✅ Camera configuration cleared!")
    else:
        print("ℹ️  No camera configuration to clear")

def export_camera_info(switcher: CameraSwitcher):
    """Export camera information to file"""
    cameras = switcher.detect_cameras()
    
    with open("camera_info.txt", "w") as f:
        f.write("VLM Camera Information\n")
        f.write("=" * 30 + "\n\n")
        
        for index, camera in cameras.items():
            if camera.get('available', False):
                f.write(f"Camera {index}:\n")
                f.write(f"  Type: {camera['type']}\n")
                f.write(f"  Resolution: {camera['resolution']}\n")
                f.write(f"  FPS: {camera['fps']:.1f}\n")
                f.write(f"  Backend: {camera['backend']}\n\n")
    
    print("✅ Camera information exported to camera_info.txt")

def main():
    """Main launcher function"""
    switcher = CameraSwitcher()
    
    print_banner()
    
    while True:
        print_menu()
        choice = input("Select option (1-9): ").strip()
        
        if choice == "1":
            # Use iPhone Camera
            camera_index = switcher.switch_to_iphone()
            if camera_index is not None:
                mode = input("Choose mode (realtime/web/test) [realtime]: ").strip() or "realtime"
                launch_with_camera(camera_index, mode)
        
        elif choice == "2":
            # Use Desktop Camera
            camera_index = switcher.switch_to_desktop()
            if camera_index is not None:
                mode = input("Choose mode (realtime/web/test) [realtime]: ").strip() or "realtime"
                launch_with_camera(camera_index, mode)
        
        elif choice == "3":
            # List All Cameras
            switcher.list_cameras()
        
        elif choice == "4":
            # Test Camera
            switcher.list_cameras()
            try:
                camera_index = int(input("\nEnter camera index to test: "))
                duration = input("Test duration in seconds [5]: ").strip() or "5"
                switcher.test_camera(camera_index, int(duration))
            except ValueError:
                print("❌ Invalid input")
        
        elif choice == "5":
            # Launch Web Interface
            saved_camera = switcher.load_saved_camera()
            if saved_camera is None:
                print("No saved camera. Please select a camera first.")
                continue
            launch_with_camera(saved_camera, "web")
        
        elif choice == "6":
            # Launch Real-time Detection
            saved_camera = switcher.load_saved_camera()
            if saved_camera is None:
                print("No saved camera. Please select a camera first.")
                continue
            launch_with_camera(saved_camera, "realtime")
        
        elif choice == "7":
            # Single Frame Test
            saved_camera = switcher.load_saved_camera()
            if saved_camera is None:
                print("No saved camera. Please select a camera first.")
                continue
            launch_with_camera(saved_camera, "test")
        
        elif choice == "8":
            # Advanced Options
            advanced_menu()
        
        elif choice == "9":
            # Exit
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option. Please try again.")
        
        # Wait for user input before showing menu again
        if choice not in ["9"]:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 