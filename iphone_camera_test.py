#!/usr/bin/env python3
"""
iPhone Camera Test Script
Test and configure iPhone camera for VLM object detection
"""

import cv2
import sys
import time
import subprocess
from config import config
from camera_utils import CameraHandler

def list_available_cameras():
    """List all available cameras with details"""
    print("🔍 Scanning for available cameras...")
    print("=" * 50)
    
    cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                camera_info = {
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'available': True
                }
                cameras.append(camera_info)
                
                # Detect camera type based on resolution
                camera_type = "Unknown"
                if width == 1920 and height == 1080:
                    camera_type = "📱 Likely iPhone Camera"
                elif width == 1280 and height == 720:
                    camera_type = "💻 Likely Built-in Camera"
                
                print(f"Camera {i}: {width}x{height} @ {fps:.1f}fps ({backend}) - {camera_type}")
            else:
                print(f"Camera {i}: Connected but no frame available")
            cap.release()
        else:
            break
    
    print("=" * 50)
    return cameras

def get_system_cameras():
    """Get system camera information using system_profiler"""
    try:
        print("\n🔍 System Camera Information:")
        print("=" * 50)
        result = subprocess.run(['system_profiler', 'SPCameraDataType'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Camera:' in line or 'Model ID:' in line or 'Unique ID:' in line:
                    print(line.strip())
        print("=" * 50)
    except Exception as e:
        print(f"Could not get system camera info: {e}")

def test_camera_performance(camera_index: int, duration: int = 10):
    """Test camera performance and latency"""
    print(f"\n🧪 Testing Camera {camera_index} Performance...")
    print("=" * 50)
    
    handler = CameraHandler(camera_index=camera_index)
    if not handler.initialize_camera():
        print(f"❌ Failed to initialize camera {camera_index}")
        return False
    
    handler.start_capture()
    
    frame_count = 0
    start_time = time.time()
    
    try:
        print(f"Testing for {duration} seconds... Press Ctrl+C to stop early")
        while time.time() - start_time < duration:
            frame = handler.get_latest_frame()
            if frame is not None:
                frame_count += 1
                
                # Show progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"  Frames captured: {frame_count}, FPS: {fps:.2f}")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        handler.stop_capture()
        
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time
    
    print(f"\n📊 Test Results:")
    print(f"  Duration: {elapsed_time:.2f} seconds")
    print(f"  Frames captured: {frame_count}")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Camera latency: ~{1000/avg_fps:.1f}ms per frame")
    
    return True

def preview_camera(camera_index: int):
    """Show live preview of camera feed"""
    print(f"\n📹 Starting Camera {camera_index} Preview...")
    print("Press 'q' to quit, 's' to save snapshot, 'i' for info")
    
    handler = CameraHandler(camera_index=camera_index)
    if not handler.initialize_camera():
        print(f"❌ Failed to initialize camera {camera_index}")
        return False
    
    handler.start_capture()
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = handler.get_latest_frame()
            if frame is not None:
                frame_count += 1
                
                # Add overlay information
                height, width = frame.shape[:2]
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Add text overlay
                cv2.putText(frame, f"Camera {camera_index}: {width}x{height}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame: {frame_count}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press 'q' to quit, 's' to save, 'i' for info", 
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(f'iPhone Camera {camera_index} Preview', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"iphone_camera_{camera_index}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Snapshot saved: {filename}")
                elif key == ord('i'):
                    print(f"\n📊 Camera Info:")
                    print(f"  Resolution: {width}x{height}")
                    print(f"  Current FPS: {fps:.2f}")
                    print(f"  Total frames: {frame_count}")
                    print(f"  Runtime: {elapsed:.1f}s")
            
    except KeyboardInterrupt:
        print("\nPreview stopped by user")
    finally:
        cv2.destroyAllWindows()
        handler.stop_capture()
    
    return True

def main():
    """Main function"""
    print("📱 iPhone Camera Configuration Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'list':
            list_available_cameras()
            get_system_cameras()
            
        elif command == 'test':
            camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            test_camera_performance(camera_index, duration)
            
        elif command == 'preview':
            camera_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            preview_camera(camera_index)
            
        else:
            print("Unknown command. Use: list, test, or preview")
            
    else:
        # Interactive mode
        print("Current configuration:")
        print(f"  Camera Index: {config.CAMERA_INDEX}")
        print(f"  Resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        print(f"  iPhone Optimization: {config.IPHONE_CAMERA_OPTIMIZATION}")
        print()
        
        while True:
            print("Options:")
            print("1. List available cameras")
            print("2. Test camera performance")
            print("3. Preview camera")
            print("4. Update configuration")
            print("5. Exit")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == '1':
                list_available_cameras()
                get_system_cameras()
                
            elif choice == '2':
                camera_index = input(f"Camera index to test (default: {config.CAMERA_INDEX}): ").strip()
                camera_index = int(camera_index) if camera_index else config.CAMERA_INDEX
                duration = input("Test duration in seconds (default: 10): ").strip()
                duration = int(duration) if duration else 10
                test_camera_performance(camera_index, duration)
                
            elif choice == '3':
                camera_index = input(f"Camera index to preview (default: {config.CAMERA_INDEX}): ").strip()
                camera_index = int(camera_index) if camera_index else config.CAMERA_INDEX
                preview_camera(camera_index)
                
            elif choice == '4':
                print("\n📝 Update Configuration:")
                new_index = input(f"Camera index (current: {config.CAMERA_INDEX}): ").strip()
                if new_index:
                    config.CAMERA_INDEX = int(new_index)
                    print(f"✅ Camera index updated to {config.CAMERA_INDEX}")
                
            elif choice == '5':
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
            
            print()

if __name__ == "__main__":
    main() 