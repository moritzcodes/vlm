#!/usr/bin/env python3
"""
Frame processing module for cropping and filtering
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
from config import Config

class FrameProcessor:
    """Handles frame cropping and filtering operations"""
    
    def __init__(self):
        # Crop settings
        self.crop_enabled = False
        self.crop_region: Optional[Tuple[int, int, int, int]] = None
        
        # Filter settings
        self.current_filter = "none"
        self.filter_params = {
            'blur_kernel': Config.FILTER_PARAMS.blur_kernel,
            'edge_threshold1': Config.FILTER_PARAMS.edge_threshold1,
            'edge_threshold2': Config.FILTER_PARAMS.edge_threshold2,
            'contrast_alpha': Config.FILTER_PARAMS.contrast_alpha,
            'brightness_beta': Config.FILTER_PARAMS.brightness_beta,
            'gamma': Config.FILTER_PARAMS.gamma
        }
        
        # Crop history for undo
        self.crop_history = []
        
    def process_frame(self, frame) -> np.ndarray:
        """Apply cropping and filtering to frame"""
        processed_frame = frame.copy()
        
        # Apply cropping first
        if self.crop_enabled and self.crop_region:
            processed_frame = self._apply_crop(processed_frame)
        
        # Apply current filter
        processed_frame = self._apply_filter(processed_frame)
        
        return processed_frame
    
    def _apply_crop(self, frame) -> np.ndarray:
        """Apply crop region to frame"""
        x, y, w, h = self.crop_region
        height, width = frame.shape[:2]
        
        # Ensure crop region is within frame bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Crop the frame
        cropped = frame[y:y+h, x:x+w]
        
        # Resize back to original dimensions for consistent UI
        return cv2.resize(cropped, (width, height))
    
    def _apply_filter(self, frame) -> np.ndarray:
        """Apply the current filter to the frame"""
        if self.current_filter == "none":
            return frame
        elif self.current_filter == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == "hsv":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif self.current_filter == "blur":
            kernel_size = self.filter_params['blur_kernel']
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        elif self.current_filter == "edge":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, self.filter_params['edge_threshold1'], 
                            self.filter_params['edge_threshold2'])
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif self.current_filter == "contrast":
            alpha = self.filter_params['contrast_alpha']
            return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        elif self.current_filter == "brightness":
            beta = self.filter_params['brightness_beta']
            return cv2.convertScaleAbs(frame, alpha=1.0, beta=beta)
        elif self.current_filter == "gamma":
            gamma = self.filter_params['gamma']
            # Build lookup table for gamma correction
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(frame, table)
        else:
            return frame
    
    def set_filter(self, filter_name: str):
        """Set the current filter"""
        valid_filters = ["none", "grayscale", "hsv", "blur", "edge", "contrast", "brightness", "gamma"]
        if filter_name in valid_filters:
            self.current_filter = filter_name
            print(f"🎨 Filter: {filter_name.title()}")
        else:
            print(f"❌ Invalid filter: {filter_name}")
    
    def adjust_filter_param(self, direction: int):
        """Adjust current filter parameters"""
        if self.current_filter == "blur":
            self.filter_params['blur_kernel'] = max(1, min(21, self.filter_params['blur_kernel'] + direction * 2))
            print(f"🔧 Blur kernel: {self.filter_params['blur_kernel']}")
        elif self.current_filter == "edge":
            if direction > 0:
                self.filter_params['edge_threshold1'] = min(255, self.filter_params['edge_threshold1'] + 10)
                self.filter_params['edge_threshold2'] = min(255, self.filter_params['edge_threshold2'] + 10)
            else:
                self.filter_params['edge_threshold1'] = max(1, self.filter_params['edge_threshold1'] - 10)
                self.filter_params['edge_threshold2'] = max(1, self.filter_params['edge_threshold2'] - 10)
            print(f"🔧 Edge thresholds: {self.filter_params['edge_threshold1']}, {self.filter_params['edge_threshold2']}")
        elif self.current_filter == "contrast":
            self.filter_params['contrast_alpha'] = max(0.1, min(3.0, self.filter_params['contrast_alpha'] + direction * 0.1))
            print(f"🔧 Contrast alpha: {self.filter_params['contrast_alpha']:.1f}")
        elif self.current_filter == "brightness":
            self.filter_params['brightness_beta'] = max(-100, min(100, self.filter_params['brightness_beta'] + direction * 10))
            print(f"🔧 Brightness beta: {self.filter_params['brightness_beta']}")
        elif self.current_filter == "gamma":
            self.filter_params['gamma'] = max(0.1, min(3.0, self.filter_params['gamma'] + direction * 0.1))
            print(f"🔧 Gamma: {self.filter_params['gamma']:.1f}")
    
    def set_crop_region(self, x: int, y: int, width: int, height: int):
        """Set crop region"""
        # Store current crop for undo
        if self.crop_enabled and self.crop_region:
            self.crop_history.append(self.crop_region)
            if len(self.crop_history) > 5:
                self.crop_history.pop(0)
        
        self.crop_region = (x, y, width, height)
        self.crop_enabled = True
        print(f"📐 Crop region set: {width}x{height} at ({x},{y})")
    
    def apply_crop_preset(self, preset_name: str, frame_width: int = 640, frame_height: int = 480):
        """Apply a predefined crop preset"""
        if preset_name not in Config.CROP_PRESETS:
            print(f"❌ Unknown preset: {preset_name}")
            return
        
        # Store current crop for undo
        if self.crop_enabled and self.crop_region:
            self.crop_history.append(self.crop_region)
            if len(self.crop_history) > 5:
                self.crop_history.pop(0)
        
        # Convert relative coordinates to absolute
        rel_x, rel_y, rel_w, rel_h = Config.CROP_PRESETS[preset_name]
        x = int(rel_x * frame_width)
        y = int(rel_y * frame_height)
        w = int(rel_w * frame_width)
        h = int(rel_h * frame_height)
        
        self.crop_region = (x, y, w, h)
        self.crop_enabled = True
        
        preset_names = {
            'center': 'Center 50%',
            'wellplate': 'Well Plate Area',
            'top_half': 'Top Half',
            'bottom_half': 'Bottom Half',
            'left_half': 'Left Half',
            'right_half': 'Right Half'
        }
        
        print(f"📐 Applied preset: {preset_names.get(preset_name, preset_name)}")
        print(f"   🔲 Region: {w}x{h} at ({x},{y})")
    
    def reset_crop(self):
        """Reset crop region completely"""
        if self.crop_enabled and self.crop_region:
            # Store current crop for undo
            self.crop_history.append(self.crop_region)
            if len(self.crop_history) > 5:
                self.crop_history.pop(0)
        
        self.crop_enabled = False
        self.crop_region = None
        print("🔄 Crop region reset")
    
    def undo_crop(self):
        """Restore previous crop region"""
        if self.crop_history:
            self.crop_region = self.crop_history.pop()
            self.crop_enabled = True
            x, y, w, h = self.crop_region
            print(f"↩️  Crop restored: {w}x{h} at ({x},{y})")
        else:
            print("❌ No previous crop to restore")
    
    def get_crop_info(self) -> Dict[str, Any]:
        """Get current crop information"""
        return {
            'enabled': self.crop_enabled,
            'region': self.crop_region,
            'filter': self.current_filter,
            'filter_params': self.filter_params.copy()
        }
    
    def draw_crop_overlay(self, frame):
        """Draw crop region overlay on frame"""
        if self.crop_enabled and self.crop_region:
            x, y, w, h = self.crop_region
            
            # Draw crop border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 255), 2)
            
            # Draw corner indicators
            corner_size = 8
            cv2.line(frame, (x, y), (x + corner_size, y), (100, 200, 255), 3)
            cv2.line(frame, (x, y), (x, y + corner_size), (100, 200, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), (100, 200, 255), 3)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), (100, 200, 255), 3)
    
    def draw_filter_overlay(self, frame):
        """Draw current filter information overlay"""
        if self.current_filter != "none":
            filter_text = f"FILTER: {self.current_filter.upper()}"
            cv2.rectangle(frame, (10, 60), (250, 90), (0, 0, 0), -1)
            cv2.putText(frame, filter_text, (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        if self.crop_enabled and self.crop_region:
            x, y, w, h = self.crop_region
            cv2.rectangle(frame, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"CROPPED: {w}x{h} at ({x},{y})", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) 