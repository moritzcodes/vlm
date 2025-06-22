#!/usr/bin/env python3
"""
Refactored Liquid Handler Monitor - Main Application
"""

import cv2
import os
import logging
from datetime import datetime
from typing import Optional, Dict, List
import threading
import time

# Import our modular components
from config import Config
from data_models import AnalysisResult, ErrorEvent, ProcedureDefinitions
from camera_manager import CameraManager
from ai_analyzer import AIAnalyzer
from frame_processor import FrameProcessor
from ui_components import UIRenderer, CropUI

class LiquidHandlerMonitor:
    """Main liquid handler monitoring application"""
    
    def __init__(self):
        # Initialize components
        self.camera_manager = CameraManager()
        self.ai_analyzer = AIAnalyzer()
        self.frame_processor = FrameProcessor()
        self.ui_renderer = UIRenderer()
        self.crop_ui = CropUI()
        
        # Application state
        self.running = False
        self.paused = False
        self.current_procedure = None
        self.current_step = 0
        self.error_count = 0
        self.error_history = []
        self.well_status = {}
        self.failed_wells = []
        
        # Analysis state
        self.last_analysis = AnalysisResult()
        self.analysis_thread = None
        self.analysis_lock = threading.Lock()
        
        # UI state
        self.show_crop_ui = False
        self.last_mouse_pos = (0, 0)
        
        # Setup logging
        self._setup_logging()
        
        # Load procedures
        self.procedures = ProcedureDefinitions.get_procedures()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('liquid_handler_monitor.log'),
                logging.StreamHandler()
            ]
        )
        
    def initialize(self) -> bool:
        """Initialize all components"""
        print("🚀 Initializing Liquid Handler Monitor...")
        
        # Initialize camera
        if not self.camera_manager.initialize():
            print("❌ Failed to initialize camera")
            return False
            
        # Check AI analyzer
        if not self.ai_analyzer.is_ready():
            print("❌ AI analyzer not ready")
            return False
            
        # Set up mouse callback
        cv2.namedWindow('Liquid Handler Monitor')
        cv2.setMouseCallback('Liquid Handler Monitor', self._mouse_callback)
        
        print("✅ All components initialized successfully")
        return True
    
    def start_procedure(self, procedure_name: str):
        """Start a specific procedure"""
        if procedure_name in self.procedures:
            self.current_procedure = procedure_name
            self.current_step = 0
            self.error_count = 0
            self.error_history.clear()
            self.well_status.clear()
            self.failed_wells.clear()
            
            print(f"🎯 Started procedure: {procedure_name}")
            print(f"📋 Total steps: {len(self.procedures[procedure_name])}")
        else:
            print(f"❌ Unknown procedure: {procedure_name}")
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
            
        self.running = True
        print("🎬 Starting monitoring loop...")
        print("\n📋 CONTROLS:")
        print("  SPACE - Pause/Resume")
        print("  1 - Blue-Red Mixing")
        print("  2 - PCR Master-Mix")
        print("  C - Toggle Crop UI")
        print("  ESC - Exit")
        print("  F1-F8 - Filters")
        print("  +/- - Adjust filter parameters")
        
        try:
            while self.running:
                if not self._process_frame():
                    break
                    
                if not self._handle_keyboard():
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            logging.error(f"Main loop error: {e}")
        finally:
            self._cleanup()
    
    def _process_frame(self) -> bool:
        """Process a single frame"""
        # Capture frame
        frame = self.camera_manager.get_frame()
        if frame is None:
            return False
            
        # Process frame (cropping and filtering)
        processed_frame = self.frame_processor.process_frame(frame)
        
        # AI analysis (if not paused)
        if not self.paused:
            self._update_analysis(processed_frame)
        
        # Render UI
        self._render_ui(frame)
        
        # Show frame
        cv2.imshow('Liquid Handler Monitor', frame)
        return True
    
    def _update_analysis(self, frame):
        """Update AI analysis in background thread"""
        # Skip if analysis is already running
        if self.analysis_thread and self.analysis_thread.is_alive():
            return
            
        # Start new analysis thread
        current_step_data = None
        if self.current_procedure and self.current_step < len(self.procedures[self.current_procedure]):
            current_step_data = self.procedures[self.current_procedure][self.current_step]
            
        self.analysis_thread = threading.Thread(
            target=self._analyze_frame_threaded,
            args=(frame.copy(), self.current_procedure, current_step_data)
        )
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def _analyze_frame_threaded(self, frame, procedure, step_data):
        """Analyze frame in background thread"""
        try:
            result = self.ai_analyzer.analyze_frame(frame, procedure, step_data)
            
            with self.analysis_lock:
                self.last_analysis = result
                self._process_analysis_result(result)
                
        except Exception as e:
            logging.error(f"Analysis thread error: {e}")
    
    def _process_analysis_result(self, result: AnalysisResult):
        """Process analysis results and update state"""
        # Update error count
        if result.status in ["ERROR", "CRITICAL"]:
            self.error_count += 1
            
            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now().isoformat(),
                error_type=result.status,
                severity="HIGH" if result.status == "CRITICAL" else "MEDIUM",
                description=result.description,
                procedure_step=f"{self.current_procedure}:{self.current_step}" if self.current_procedure else "unknown",
                expected_state="",
                actual_state="",
                confidence=result.confidence
            )
            
            self.error_history.append(error_event)
            
            # Keep only last 50 errors
            if len(self.error_history) > 50:
                self.error_history.pop(0)
        
        # Update well status
        if result.well_analysis:
            self.well_status.update(result.well_analysis)
            
        # Update failed wells
        if result.failed_wells:
            self.failed_wells = result.failed_wells
        
        # Auto-advance procedure step if successful
        if (result.status == "NORMAL" and result.compliance and 
            self.current_procedure and not result.arm_blocking):
            # Simple auto-advance logic - could be made more sophisticated
            pass
    
    def _render_ui(self, frame):
        """Render all UI elements"""
        # Draw main professional UI
        self.ui_renderer.draw_professional_ui(
            frame, 
            self.last_analysis,
            self.current_procedure,
            self.current_step,
            self.error_count,
            self.well_status,
            self.failed_wells
        )
        
        # Draw crop overlays
        self.frame_processor.draw_crop_overlay(frame)
        self.frame_processor.draw_filter_overlay(frame)
        
        # Draw crop UI if visible
        if self.crop_ui.visible:
            self.crop_ui.draw(frame)
        
        # Draw pause indicator
        if self.paused:
            cv2.rectangle(frame, (10, 100), (200, 140), (0, 0, 0), -1)
            cv2.putText(frame, "PAUSED", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    def _handle_keyboard(self) -> bool:
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            return False
        elif key == ord(' '):  # Space - Pause/Resume
            self.paused = not self.paused
            print(f"⏸️  {'Paused' if self.paused else 'Resumed'}")
        elif key == ord('1'):  # Start blue-red mixing
            self.start_procedure("blue_red_mixing")
        elif key == ord('2'):  # Start PCR master-mix
            self.start_procedure("pcr_master_mix")
        elif key == ord('c') or key == ord('C'):  # Toggle crop UI
            self.crop_ui.toggle_visibility()
            print(f"🎛️  Crop UI {'shown' if self.crop_ui.visible else 'hidden'}")
        elif key == 255:  # No key pressed
            pass
        else:
            self._handle_filter_keys(key)
            
        return True
    
    def _handle_filter_keys(self, key):
        """Handle filter-related keyboard shortcuts"""
        # Filter selection (F1-F8)
        filter_map = {
            ord('1'): "none",
            ord('2'): "grayscale", 
            ord('3'): "hsv",
            ord('4'): "blur",
            ord('5'): "edge",
            ord('6'): "contrast",
            ord('7'): "brightness",
            ord('8'): "gamma"
        }
        
        if key in filter_map:
            self.frame_processor.set_filter(filter_map[key])
        elif key == ord('+') or key == ord('='):
            self.frame_processor.adjust_filter_param(1)
        elif key == ord('-'):
            self.frame_processor.adjust_filter_param(-1)
        elif key == ord('r'):  # Reset crop
            self.frame_processor.reset_crop()
        elif key == ord('u'):  # Undo crop
            self.frame_processor.undo_crop()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.last_mouse_pos = (x, y)
        
        # Handle crop UI interactions
        if self.crop_ui.visible:
            element = self.crop_ui.get_element_at_position(x, y)
            
            if event == cv2.EVENT_MOUSEMOVE:
                self.crop_ui.hover_element = element
            elif event == cv2.EVENT_LBUTTONDOWN and element:
                self.crop_ui.active_element = element
                self._handle_crop_ui_click(element)
    
    def _handle_crop_ui_click(self, element):
        """Handle crop UI button clicks"""
        if not element:
            return
            
        height, width = self.camera_manager.frame_height, self.camera_manager.frame_width
        
        if 'preset' in element:
            preset = element['preset']
            if preset == 'full':
                self.frame_processor.reset_crop()
            else:
                self.frame_processor.apply_crop_preset(preset, width, height)
        elif 'action' in element:
            action = element['action']
            if action == 'reset':
                self.frame_processor.reset_crop()
            elif action == 'undo':
                self.frame_processor.undo_crop()
            elif action == 'apply':
                print("✅ Crop settings applied")
    
    def _cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        self.running = False
        
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
        
        # Release camera
        self.camera_manager.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("✅ Cleanup complete")

def main():
    """Main entry point"""
    print("🧪 Liquid Handler Monitor v2.0 - Refactored")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create and run monitor
    monitor = LiquidHandlerMonitor()
    monitor.run()

if __name__ == "__main__":
    main() 