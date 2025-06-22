#!/usr/bin/env python3
"""
📱 Simple iPhone Camera Test
- Auto-detects iPhone camera
- Live video feed
- No AI dependencies - works immediately!
"""

import cv2
import time

def find_iphone_camera():
    """Auto-detect iPhone camera index"""
    print("🔍 Searching for iPhone camera...")
    
    cameras_found = []
    
    for i in range(8):  # Check more indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Log all cameras
                camera_info = f"Camera {i}: {width}x{height} @ {fps}fps"
                cameras_found.append((i, width, height, fps))
                print(f"  📷 {camera_info}")
                
                # iPhone cameras are typically high resolution
                if width >= 1920 and height >= 1080:
                    cap.release()
                    print(f"📱 iPhone camera selected: Index {i}")
                    return i
            cap.release()
    
    if cameras_found:
        print(f"\n📋 Found {len(cameras_found)} cameras:")
        for i, (idx, w, h, fps) in enumerate(cameras_found):
            print(f"  {i+1}. Camera {idx}: {w}x{h} @ {fps}fps")
        
        # If no high-res camera found, use the first one
        first_cam = cameras_found[0][0]
        print(f"🎯 Using Camera {first_cam} as default")
        return first_cam
    
    print("❌ No cameras found")
    return None

def test_iphone_camera():
    """Test iPhone camera with live feed"""
    camera_index = find_iphone_camera()
    
    if camera_index is None:
        print("💡 Make sure your iPhone is connected via USB and trusted")
        return
    
    print(f"\n📱 Starting iPhone camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened: {width}x{height} @ {fps}fps")
    print("\n🚀 LIVE CAMERA FEED")
    print("Controls:")
    print("  📸 Press SPACE to take photo")
    print("  🎬 Press 'r' to record 5 seconds")
    print("  ❌ Press 'q' to quit")
    print("=" * 50)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Add overlay with info
            cv2.rectangle(frame, (10, 10), (500, 80), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (500, 80), (0, 255, 0), 2)
            
            cv2.putText(frame, "📱 iPhone Camera Live Feed", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # FPS calculation
            elapsed = current_time - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"Frame: {frame_count} | FPS: {current_fps:.1f}", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Controls info
            cv2.putText(frame, "SPACE=Photo | R=Record | Q=Quit", (20, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            cv2.imshow('📱 iPhone Camera Test', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord(' '):
                # Take photo
                filename = f"iphone_photo_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Photo saved: {filename}")
            elif key == ord('r'):
                # Record 5 seconds
                print("🎬 Recording 5 seconds...")
                record_frames = []
                record_start = time.time()
                
                while time.time() - record_start < 5.0:
                    ret, record_frame = cap.read()
                    if ret:
                        record_frames.append(record_frame)
                        # Show recording indicator
                        cv2.circle(record_frame, (width - 30, 30), 10, (0, 0, 255), -1)
                        cv2.putText(record_frame, "REC", (width - 55, 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.imshow('📱 iPhone Camera Test', record_frame)
                        cv2.waitKey(1)
                
                # Save recording
                if record_frames:
                    filename = f"iphone_recording_{int(time.time())}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                    
                    for record_frame in record_frames:
                        out.write(record_frame)
                    
                    out.release()
                    print(f"🎬 Recording saved: {filename}")
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final stats
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n📊 Session Stats:")
        print(f"   Duration: {total_time:.1f}s")
        print(f"   Frames: {frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("✅ Camera test completed!")

if __name__ == "__main__":
    print("📱 iPhone Camera Test - No AI Required!")
    test_iphone_camera() 