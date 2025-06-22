#!/usr/bin/env python3
"""
Camera management for the Liquid Handler Monitor
"""

import cv2
import logging
from typing import Optional, Tuple
from config import Config

class CameraManager:
    """Manages camera connection and frame capture"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index: Optional[int] = None
        self.frame_width = Config.CAMERA.default_width
        self.frame_height = Config.CAMERA.default_height
        
    def find_camera(self) -> Optional[int]:
        """Auto-detect camera index, preferring iPhone camera"""
        print("🔍 Searching for camera...")
        
        # Try each index in order of preference
        for i in Config.CAMERA.search_indices:
            print(f"📱 Trying camera index {i}...")
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    cap.release()
                    
                    # Check if it's a reasonable camera resolution
                    if width >= 640 and height >= 480:
                        print(f"✅ Camera found at index {i}: {width}x{height}")
                        self.frame_width = width
                        self.frame_height = height
                        return i
                cap.release()
        
        print("❌ No suitable camera found")
        return None
    
    def connect(self) -> bool:
        """Connect to the camera"""
        self.camera_index = self.find_camera()
        if self.camera_index is None:
            return False
        
        print(f"📱 Connecting to camera at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("❌ Camera connection failed")
            return False
        
        # Test frame capture
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Cannot capture frames")
            return False
        
        height, width = frame.shape[:2]
        self.frame_width = width
        self.frame_height = height
        
        print(f"✅ Connected: {width}x{height}")
        return True
    
    def capture_frame(self) -> Tuple[bool, Optional]:
        """Capture a frame from the camera"""
        if not self.cap:
            return False, None
        
        return self.cap.read()
    
    def get_frame_dimensions(self) -> Tuple[int, int]:
        """Get current frame dimensions"""
        return self.frame_width, self.frame_height
    
    def is_connected(self) -> bool:
        """Check if camera is connected"""
        return self.cap is not None and self.cap.isOpened()
    
    def disconnect(self):
        """Disconnect from the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
        print("📱 Camera disconnected")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.disconnect() 