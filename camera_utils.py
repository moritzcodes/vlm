import cv2
import numpy as np
from PIL import Image
import threading
import queue
import time
from typing import Optional, Tuple, Generator
import logging

from config import config

logger = logging.getLogger(__name__)

class CameraHandler:
    """Handle camera input and frame processing for VLM analysis"""
    
    def __init__(self, camera_index: int = None):
        self.camera_index = camera_index or config.CAMERA_INDEX
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.capture_thread = None
        self.frame_count = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture"""
        try:
            # Try different camera backends for better iPhone compatibility
            backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                if self.cap.isOpened():
                    # Give camera time to initialize
                    time.sleep(0.3)
                    
                    # Test frame capture to ensure camera is working
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"Camera initialized successfully with backend: {backend}")
                        break
                    else:
                        logger.warning(f"Camera opened but cannot capture frames with backend: {backend}")
                        self.cap.release()
                        continue
            else:
                logger.error("Failed to initialize camera with any backend")
                return False
            
            # Configure camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
            
            # iPhone-specific optimizations
            if config.IPHONE_CAMERA_OPTIMIZATION:
                # Minimize buffer size for lower latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
                
                # iPhone camera optimizations
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
                
                # Set high quality settings
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                logger.info("📱 iPhone camera optimizations enabled")
            else:
                # Standard optimization for low latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Log camera details
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📱 Camera {self.camera_index}: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Detect if this might be an iPhone camera based on resolution
            if actual_width == 1920 and actual_height == 1080:
                logger.info("🎯 High-resolution camera detected - likely iPhone camera")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def start_capture(self):
        """Start the camera capture thread"""
        if not self.is_running and self.cap and self.cap.isOpened():
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            logger.info("Camera capture started")
    
    def stop_capture(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info("Camera capture stopped")
    
    def _capture_frames(self):
        """Continuously capture frames in a separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    
                    # Skip frames based on configuration for performance
                    if self.frame_count % config.FRAME_SKIP_RATIO == 0:
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            # Remove oldest frame to make space
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame)
                            except queue.Empty:
                                pass
                else:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from the queue"""
        try:
            # Get the most recent frame, discarding older ones
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
    
    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously"""
        while self.is_running:
            frame = self.get_latest_frame()
            if frame is not None:
                yield frame
            else:
                time.sleep(0.01)  # Small delay if no frame available
    
    def frame_to_pil(self, frame: np.ndarray) -> Image.Image:
        """Convert OpenCV frame to PIL Image"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    
    def frame_to_bytes(self, frame: np.ndarray, format: str = 'JPEG', quality: int = 85) -> bytes:
        """Convert frame to bytes for API transmission"""
        pil_image = self.frame_to_pil(frame)
        
        # Optimize image size for faster transmission
        if format.upper() == 'JPEG':
            # Compress JPEG for faster upload
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format, quality=quality, optimize=True)
            return buffer.getvalue()
        else:
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            return buffer.getvalue()
    
    def resize_frame(self, frame: np.ndarray, max_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        max_width, max_height = max_size
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """Draw detection bounding boxes and labels on frame"""
        frame_copy = frame.copy()
        height, width = frame.shape[:2]
        
        for detection in detections:
            # Assuming detection format: {'bbox': [x_min, y_min, x_max, y_max], 'label': 'object', 'confidence': 0.9}
            if 'bbox' in detection and 'label' in detection:
                bbox = detection['bbox']
                label = detection['label']
                confidence = detection.get('confidence', 0.0)
                
                # Convert normalized coordinates to pixel coordinates if needed
                if all(0 <= coord <= 1 for coord in bbox):
                    x_min = int(bbox[0] * width)
                    y_min = int(bbox[1] * height)
                    x_max = int(bbox[2] * width)
                    y_max = int(bbox[3] * height)
                else:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw label
                label_text = f"{label}: {confidence:.2f}" if confidence > 0 else label
                cv2.putText(frame_copy, label_text, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_copy
    
    def __enter__(self):
        """Context manager entry"""
        if self.initialize_camera():
            self.start_capture()
            return self
        else:
            raise RuntimeError("Failed to initialize camera")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_capture()


class FrameProcessor:
    """Utility class for frame processing and optimization"""
    
    @staticmethod
    def preprocess_frame_for_vlm(frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for optimal VLM analysis"""
        # Resize for faster processing
        frame = CameraHandler().resize_frame(frame, max_size=(512, 512))
        
        # Enhance contrast and brightness for better detection
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        # Reduce noise
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        
        return frame
    
    @staticmethod
    def extract_roi(frame: np.ndarray, roi_percent: float = 0.8) -> np.ndarray:
        """Extract region of interest from center of frame"""
        height, width = frame.shape[:2]
        
        # Calculate ROI coordinates
        roi_width = int(width * roi_percent)
        roi_height = int(height * roi_percent)
        
        x_start = (width - roi_width) // 2
        y_start = (height - roi_height) // 2
        
        return frame[y_start:y_start + roi_height, x_start:x_start + roi_width] 