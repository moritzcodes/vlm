#!/usr/bin/env python3
"""
VLM Real-Time Object Detection
Main application for running VLM object detection with iPhone camera feed
"""

import asyncio
import cv2
import argparse
import logging
import time
import json
import sys
import signal
from typing import Dict, List

from config import config
from camera_utils import CameraHandler, FrameProcessor
from vlm_processor import VertexAIVLMProcessor, ObjectDetectionAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLMRealTimeApp:
    """Command-line VLM real-time object detection application"""
    
    def __init__(self, args):
        self.args = args
        self.camera_handler = None
        self.vlm_processor = None
        self.is_running = False
        self.detection_count = 0
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Initializing VLM Real-Time Object Detection...")
        
        try:
            # Initialize VLM processor
            self.vlm_processor = VertexAIVLMProcessor(config)
            if not self.vlm_processor.initialize_model():
                logger.error("Failed to initialize VLM processor")
                return False
            
            # Initialize camera handler
            self.camera_handler = CameraHandler(camera_index=self.args.camera_index)
            if not self.camera_handler.initialize_camera():
                logger.error("Failed to initialize camera")
                return False
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def start_camera_preview(self):
        """Start camera preview window"""
        # Initialize camera handler if not already done
        if not self.camera_handler:
            self.camera_handler = CameraHandler(camera_index=self.args.camera_index)
            if not self.camera_handler.initialize_camera():
                logger.error("Failed to initialize camera")
                return
        
        logger.info("Starting camera preview... Press 'q' to quit, 's' to save snapshot")
        
        self.camera_handler.start_capture()
        
        while True:
            frame = self.camera_handler.get_latest_frame()
            if frame is not None:
                # Show preview
                cv2.imshow('Camera Preview', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save snapshot
                    timestamp = int(time.time())
                    filename = f"snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"📸 Snapshot saved as {filename}")
        
        cv2.destroyAllWindows()
        self.camera_handler.stop_capture()
    
    async def run_realtime_detection(self):
        """Run real-time object detection"""
        if not await self.initialize():
            return False
        
        logger.info("🚀 Starting real-time VLM object detection...")
        logger.info("Press Ctrl+C to stop")
        
        self.is_running = True
        self.start_time = time.time()
        self.camera_handler.start_capture()
        
        try:
            # Create display window if not headless
            if not self.args.headless:
                cv2.namedWindow('VLM Object Detection', cv2.WINDOW_RESIZABLE)
            
            frame_count = 0
            last_detection_time = 0
            
            while self.is_running:
                frame = self.camera_handler.get_latest_frame()
                
                if frame is not None:
                    frame_count += 1
                    current_time = time.time()
                    
                    # Process frame with VLM at specified interval
                    if (current_time - last_detection_time) >= self.args.detection_interval:
                        logger.info(f"🔍 Processing frame {frame_count}...")
                        
                        # Run VLM detection
                        result = await self.vlm_processor.process_frame_async(
                            frame, self.args.prompt_type
                        )
                        
                        # Process and display results
                        self._process_detection_result(result, frame_count)
                        
                        # Draw detections on frame (if detection result has objects)
                        if result.get('success') and result.get('detection_result'):
                            detection_result = result['detection_result']
                            if detection_result.objects:
                                # Convert to format expected by camera handler
                                detections = []
                                for i, obj in enumerate(detection_result.objects):
                                    detection = {
                                        'label': obj,
                                        'confidence': detection_result.confidence_scores[i] if i < len(detection_result.confidence_scores) else 0.8,
                                        'bbox': detection_result.bounding_boxes[i] if i < len(detection_result.bounding_boxes) else [0.1, 0.1, 0.9, 0.9],
                                        'description': detection_result.descriptions[i] if i < len(detection_result.descriptions) else ''
                                    }
                                    if detection['confidence'] >= self.args.confidence_threshold:
                                        detections.append(detection)
                                
                                if detections:
                                    frame = self.camera_handler.draw_detections(frame, detections)
                        
                        last_detection_time = current_time
                        self.detection_count += 1
                    
                    # Display frame if not headless
                    if not self.args.headless:
                        # Add info overlay
                        self._add_info_overlay(frame, frame_count)
                        cv2.imshow('VLM Object Detection', frame)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Detection error: {e}")
        finally:
            self.stop()
        
        return True
    
    def _process_detection_result(self, result: Dict, frame_count: int):
        """Process and log detection results"""
        processing_time = result.get('processing_time', 0)
        
        if result.get('success') and result.get('detection_result'):
            detection_result = result['detection_result']
            objects = detection_result.objects
            
            logger.info(f"Frame {frame_count}: {len(objects)} objects detected in {processing_time:.3f}s")
            
            # Log raw response if available
            raw_response = result.get('raw_response', '')
            if raw_response and len(raw_response) < 200:  # Only log short responses
                logger.info(f"Scene: {raw_response}")
            
            # Log individual objects
            for i, obj in enumerate(objects):
                confidence = detection_result.confidence_scores[i] if i < len(detection_result.confidence_scores) else 0.8
                description = detection_result.descriptions[i] if i < len(detection_result.descriptions) else ''
                bbox = detection_result.bounding_boxes[i] if i < len(detection_result.bounding_boxes) else []
                
                logger.info(f"  Object {i+1}: {obj} (confidence: {confidence:.2f})")
                if description:
                    logger.info(f"    Description: {description}")
                if bbox:
                    logger.info(f"    Position: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
        else:
            error = result.get('error', 'Unknown error')
            logger.error(f"Frame {frame_count}: Detection failed - {error}")
        
        # Save results to file if requested
        if self.args.save_results:
            self._save_detection_result(result, frame_count)
    
    def _save_detection_result(self, result: Dict, frame_count: int):
        """Save detection results to JSON file"""
        try:
            filename = f"detection_results_{int(time.time())}.json"
            
            # Add frame information
            result['frame_number'] = frame_count
            result['timestamp'] = time.time()
            
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.debug(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _add_info_overlay(self, frame, frame_count: int):
        """Add information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection counter
        cv2.putText(frame, f"Detections: {self.detection_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add uptime
        if self.start_time:
            uptime = int(time.time() - self.start_time)
            cv2.putText(frame, f"Uptime: {uptime}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    async def test_single_frame(self):
        """Test VLM processing on a single frame"""
        if not await self.initialize():
            return False
        
        logger.info("📸 Capturing and analyzing single frame...")
        
        self.camera_handler.start_capture()
        await asyncio.sleep(1)  # Wait for camera to stabilize
        
        frame = self.camera_handler.get_latest_frame()
        if frame is None:
            logger.error("Failed to capture frame")
            return False
        
        # Process frame
        result = await self.vlm_processor.process_frame_async(frame, self.args.prompt_type)
        
        # Display results
        self._process_detection_result(result, 1)
        
        # Save frame with detections
        if result.get('success') and result.get('detection_result'):
            detection_result = result['detection_result']
            if detection_result.objects:
                # Convert to format expected by camera handler
                detections = []
                for i, obj in enumerate(detection_result.objects):
                    detection = {
                        'label': obj,
                        'confidence': detection_result.confidence_scores[i] if i < len(detection_result.confidence_scores) else 0.8,
                        'bbox': detection_result.bounding_boxes[i] if i < len(detection_result.bounding_boxes) else [0.1, 0.1, 0.9, 0.9],
                        'description': detection_result.descriptions[i] if i < len(detection_result.descriptions) else ''
                    }
                    detections.append(detection)
                
                if detections:
                    frame_with_detections = self.camera_handler.draw_detections(frame, detections)
                    cv2.imwrite('test_detection.jpg', frame_with_detections)
                    logger.info("Test result saved as 'test_detection.jpg'")
        
        self.camera_handler.stop_capture()
        return True
    
    def stop(self):
        """Stop all components"""
        self.is_running = False
        
        if self.camera_handler:
            self.camera_handler.stop_capture()
        
        # VLM processor cleanup (if needed)
        
        cv2.destroyAllWindows()
        
        if self.start_time:
            total_time = time.time() - self.start_time
            fps = self.detection_count / total_time if total_time > 0 else 0
            logger.info(f"📊 Session complete: {self.detection_count} detections in {total_time:.1f}s (avg {fps:.2f} FPS)")


def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="VLM Real-Time Object Detection using Google Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode realtime                    # Start real-time detection
  python main.py --mode test                        # Test single frame
  python main.py --mode preview                     # Camera preview only
  python main.py --mode realtime --headless         # Run without display
  python main.py --mode realtime --save-results     # Save detection results
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['realtime', 'test', 'preview'],
        default='realtime',
        help='Application mode (default: realtime)'
    )
    
    parser.add_argument(
        '--camera-index', 
        type=int, 
        default=config.CAMERA_INDEX,
        help=f'Camera device index (default: {config.CAMERA_INDEX})'
    )
    
    parser.add_argument(
        '--prompt-type',
        choices=['realtime', 'objects', 'detailed'],
        default='realtime',
        help='VLM prompt type for detection (default: realtime)'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=config.DETECTION_CONFIDENCE_THRESHOLD,
        help=f'Minimum confidence threshold for detections (default: {config.DETECTION_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--detection-interval',
        type=float,
        default=2.0,
        help='Interval between VLM detections in seconds (default: 2.0)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without GUI display'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save detection results to JSON files'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


async def main():
    """Main application entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate configuration
    try:
        config._validate_config()
    except ValueError as e:
        logger.error(e)
        sys.exit(1)
    
    # Create and run application
    app = VLMRealTimeApp(args)
    
    try:
        if args.mode == 'realtime':
            success = await app.run_realtime_detection()
        elif args.mode == 'test':
            success = await app.test_single_frame()
        elif args.mode == 'preview':
            app.start_camera_preview()
            success = True
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 