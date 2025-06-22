import streamlit as st
import asyncio
import cv2
import numpy as np
import time
import json
import threading
from typing import Dict, List
import logging
import matplotlib.pyplot as plt
import pandas as pd

from config import config
from camera_utils import CameraHandler, FrameProcessor
from vlm_processor import VertexAIVLMProcessor, ObjectDetectionAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="VLM Real-Time Object Detection",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitVLMApp:
    """Streamlit application for VLM real-time object detection"""
    
    def __init__(self):
        self.camera_handler = None
        self.vlm_processor = None
        self.is_running = False
        self.detection_history = []
        self.performance_data = []
    
    def initialize_components(self):
        """Initialize camera and VLM processor"""
        try:
            # Initialize VLM processor
            if 'vlm_processor' not in st.session_state:
                st.session_state.vlm_processor = VertexAIVLMProcessor(config)
                if not st.session_state.vlm_processor.initialize_model():
                    st.error("Failed to initialize Gemini VLM model. Please check your API key.")
                    return False
            
            self.vlm_processor = st.session_state.vlm_processor
            
            # Initialize camera handler with current camera index
            if 'camera_handler' not in st.session_state or st.session_state.get('camera_index') != config.CAMERA_INDEX:
                # Create new camera handler if index changed
                camera_handler = CameraHandler(camera_index=config.CAMERA_INDEX)
                if camera_handler.initialize_camera():
                    st.session_state.camera_handler = camera_handler
                    st.session_state.camera_index = config.CAMERA_INDEX
                    logger.info(f"Camera handler initialized for index {config.CAMERA_INDEX}")
                else:
                    st.error(f"Failed to initialize camera {config.CAMERA_INDEX}")
                    return False
            
            self.camera_handler = st.session_state.camera_handler
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            logger.error(f"Component initialization error: {e}")
            return False
    
    def render_sidebar(self):
        """Render the sidebar with controls and settings"""
        st.sidebar.header("🎮 Controls")
        
        # API Configuration
        with st.sidebar.expander("🔑 API Configuration"):
            api_key = st.text_input(
                "Google API Key", 
                value=config.GOOGLE_API_KEY if config.GOOGLE_API_KEY else "",
                type="password",
                help="Enter your Google API key for Gemini"
            )
            
            if api_key and api_key != config.GOOGLE_API_KEY:
                config.GOOGLE_API_KEY = api_key
                # Reinitialize VLM processor with new API key
                if 'vlm_processor' in st.session_state:
                    del st.session_state['vlm_processor']
                st.rerun()
        
        # Camera Settings
        with st.sidebar.expander("📷 Camera Settings"):
            # Import camera switcher
            try:
                from camera_switcher import CameraSwitcher
                switcher = CameraSwitcher()
                
                # Quick camera switch buttons
                st.write("**Quick Switch:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📱 iPhone", help="Switch to iPhone camera"):
                        camera_index = switcher.switch_to_iphone()
                        if camera_index is not None:
                            config.CAMERA_INDEX = camera_index
                            st.success(f"Switched to iPhone Camera {camera_index}")
                            st.rerun()
                
                with col2:
                    if st.button("💻 Desktop", help="Switch to desktop camera"):
                        camera_index = switcher.switch_to_desktop()
                        if camera_index is not None:
                            config.CAMERA_INDEX = camera_index
                            st.success(f"Switched to Desktop Camera {camera_index}")
                            st.rerun()
                
                # Camera selection
                cameras = switcher.detect_cameras()
                available_cameras = {k: v for k, v in cameras.items() if v.get('available', False)}
                
                if available_cameras:
                    st.write("**Available Cameras:**")
                    camera_options = {}
                    for idx, cam in available_cameras.items():
                        camera_options[f"Camera {idx}: {cam['type']} ({cam['resolution']})"] = idx
                    
                    current_camera_key = None
                    for key, idx in camera_options.items():
                        if idx == config.CAMERA_INDEX:
                            current_camera_key = key
                            break
                    
                    selected_camera = st.selectbox(
                        "Select Camera",
                        options=list(camera_options.keys()),
                        index=list(camera_options.keys()).index(current_camera_key) if current_camera_key else 0,
                        help="Choose camera device"
                    )
                    
                    if selected_camera:
                        new_camera_index = camera_options[selected_camera]
                        if new_camera_index != config.CAMERA_INDEX:
                            config.CAMERA_INDEX = new_camera_index
                            switcher.switch_to_camera(new_camera_index)
                            st.success(f"Switched to {selected_camera}")
                            st.rerun()
                
            except ImportError:
                # Fallback to manual camera index input
                camera_index = st.number_input(
                    "Camera Index", 
                    min_value=0, 
                    max_value=10, 
                    value=config.CAMERA_INDEX,
                    help="Select camera device (0 for default, 1 for external)"
                )
                config.CAMERA_INDEX = camera_index
            
            frame_skip = st.slider(
                "Frame Skip Ratio", 
                min_value=1, 
                max_value=10, 
                value=config.FRAME_SKIP_RATIO,
                help="Skip frames for better performance (higher = faster)"
            )
            
            config.FRAME_SKIP_RATIO = frame_skip
        
        # Detection Settings
        with st.sidebar.expander("🎯 Detection Settings"):
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=config.DETECTION_CONFIDENCE_THRESHOLD,
                step=0.05,
                help="Minimum confidence for object detection"
            )
            
            prompt_type = st.selectbox(
                "Detection Mode",
                options=['realtime', 'objects', 'detailed'],
                index=0,
                help="Choose detection mode: realtime (fast), objects (structured), detailed (comprehensive)"
            )
            
            config.DETECTION_CONFIDENCE_THRESHOLD = confidence_threshold
            
            return prompt_type
        
        # Performance Stats
        if self.vlm_processor:
            stats = self.vlm_processor.get_performance_stats()
            if 'avg_processing_time' in stats:
                st.sidebar.header("📊 Performance")
                st.sidebar.metric("Avg Processing Time", f"{stats['avg_processing_time']:.3f}s")
                st.sidebar.metric("Estimated FPS", f"{stats['fps_estimate']:.1f}")
                st.sidebar.metric("Frames Processed", stats['total_frames_processed'])
    
    def render_main_interface(self, prompt_type: str):
        """Render the main interface with video feed and detections"""
        st.title("📸 VLM Real-Time Object Detection")
        st.markdown("Real-time object detection using Google's Gemini Vision Language Model")
        
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎥 Live Camera Feed")
            
            # Placeholder for video feed
            video_placeholder = st.empty()
            
            # Control buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                start_button = st.button("▶️ Start Detection", type="primary")
            
            with button_col2:
                stop_button = st.button("⏹️ Stop Detection")
            
            with button_col3:
                snapshot_button = st.button("📸 Take Snapshot")
        
        with col2:
            st.subheader("🎯 Detection Results")
            detection_placeholder = st.empty()
            
            st.subheader("📈 Performance Metrics")
            metrics_placeholder = st.empty()
        
        # Handle button actions
        if start_button:
            self.start_detection(video_placeholder, detection_placeholder, metrics_placeholder, prompt_type)
        
        if stop_button:
            self.stop_detection()
        
        if snapshot_button:
            self.take_snapshot()
    
    def start_detection(self, video_placeholder, detection_placeholder, metrics_placeholder, prompt_type):
        """Start the real-time detection process"""
        if not self.initialize_components():
            return
        
        try:
            # Initialize camera
            if not self.camera_handler.initialize_camera():
                st.error("Failed to initialize camera. Please check your camera connection.")
                return
            
            self.camera_handler.start_capture()
            self.is_running = True
            
            st.success("🎬 Detection started! Point your camera at objects to detect them.")
            
            # Start the detection loop
            self.run_detection_loop(video_placeholder, detection_placeholder, metrics_placeholder, prompt_type)
            
        except Exception as e:
            st.error(f"Error starting detection: {e}")
            logger.error(f"Detection start error: {e}")
    
    def stop_detection(self):
        """Stop the real-time detection process"""
        self.is_running = False
        
        if self.camera_handler:
            self.camera_handler.stop_capture()
        
        if self.vlm_processor:
            self.vlm_processor.stop_processing()
        
        st.info("🛑 Detection stopped.")
    
    def take_snapshot(self):
        """Take a snapshot of the current frame"""
        if self.camera_handler and self.camera_handler.is_running:
            frame = self.camera_handler.get_latest_frame()
            if frame is not None:
                # Save snapshot
                timestamp = int(time.time())
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                st.success(f"📸 Snapshot saved as {filename}")
            else:
                st.warning("No frame available for snapshot")
        else:
            st.warning("Camera not running. Start detection first.")
    
    def run_detection_loop(self, video_placeholder, detection_placeholder, metrics_placeholder, prompt_type):
        """Main detection loop"""
        frame_count = 0
        
        while self.is_running:
            try:
                # Get latest frame
                frame = self.camera_handler.get_latest_frame()
                
                if frame is not None:
                    frame_count += 1
                    
                    # Process frame with VLM (async)
                    if frame_count % config.FRAME_SKIP_RATIO == 0:
                        # Run VLM processing in background
                        result = asyncio.run(self.vlm_processor.process_frame_async(frame, prompt_type))
                        
                        # Filter detections
                        if 'detections' in result:
                            result['detections'] = ObjectDetectionAnalyzer.filter_detections_by_confidence(
                                result['detections'], config.DETECTION_CONFIDENCE_THRESHOLD
                            )
                        
                        # Store detection history
                        self.detection_history.append(result)
                        if len(self.detection_history) > 100:  # Keep last 100 results
                            self.detection_history.pop(0)
                        
                        # Draw detections on frame
                        if 'detections' in result and result['detections']:
                            frame_with_detections = self.camera_handler.draw_detections(frame, result['detections'])
                        else:
                            frame_with_detections = frame
                        
                        # Update detection results
                        self.update_detection_display(detection_placeholder, result)
                    else:
                        frame_with_detections = frame
                        result = None
                    
                    # Update video display
                    self.update_video_display(video_placeholder, frame_with_detections)
                    
                    # Update metrics
                    if result and 'metadata' in result:
                        self.update_metrics_display(metrics_placeholder, result['metadata'])
                    
                    # Small delay to prevent overwhelming the interface
                    time.sleep(0.03)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                st.error(f"Detection error: {e}")
                break
    
    def update_video_display(self, placeholder, frame):
        """Update the video display with current frame"""
        # Convert BGR to RGB for Streamlit
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
    
    def update_detection_display(self, placeholder, result):
        """Update the detection results display"""
        with placeholder.container():
            if 'scene_description' in result:
                st.write("**Scene Description:**")
                st.write(result['scene_description'])
            
            if 'detections' in result and result['detections']:
                st.write("**Detected Objects:**")
                for i, detection in enumerate(result['detections']):
                    with st.expander(f"Object {i+1}: {detection.get('label', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Label:** {detection.get('label', 'N/A')}")
                            st.write(f"**Confidence:** {detection.get('confidence', 0):.2f}")
                        with col2:
                            bbox = detection.get('bbox', [])
                            if bbox:
                                st.write(f"**Position:** [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")
                            if 'description' in detection:
                                st.write(f"**Description:** {detection['description']}")
            else:
                st.write("*No objects detected in current frame*")
    
    def update_metrics_display(self, placeholder, metadata):
        """Update the performance metrics display"""
        with placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Processing Time", f"{metadata.get('processing_time', 0):.3f}s")
                st.metric("Frame Count", metadata.get('frame_count', 0))
            
            with col2:
                st.metric("Timestamp", time.strftime("%H:%M:%S", time.localtime(metadata.get('timestamp', 0))))
                if 'error' in metadata:
                    st.error(f"Error: {metadata['error']}")
    
    def render_analytics_page(self):
        """Render analytics and history page"""
        st.title("📊 Analytics & History")
        
        if not self.detection_history:
            st.info("No detection data available. Start detection to collect data.")
            return
        
        # Performance analytics
        st.subheader("⚡ Performance Analytics")
        
        # Extract performance data
        processing_times = []
        timestamps = []
        detection_counts = []
        
        for result in self.detection_history[-50:]:  # Last 50 results
            if 'metadata' in result:
                processing_times.append(result['metadata'].get('processing_time', 0))
                timestamps.append(result['metadata'].get('timestamp', 0))
                detection_counts.append(len(result.get('detections', [])))
        
        if processing_times:
            # Create performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.line_chart(pd.DataFrame({
                    'Processing Time (s)': processing_times
                }))
            
            with col2:
                st.line_chart(pd.DataFrame({
                    'Objects Detected': detection_counts
                }))
        
        # Detection history
        st.subheader("🎯 Recent Detections")
        
        for i, result in enumerate(reversed(self.detection_history[-10:])):  # Last 10 results
            with st.expander(f"Detection {len(self.detection_history) - i}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if 'metadata' in result:
                        st.write(f"**Time:** {time.strftime('%H:%M:%S', time.localtime(result['metadata'].get('timestamp', 0)))}")
                        st.write(f"**Processing:** {result['metadata'].get('processing_time', 0):.3f}s")
                
                with col2:
                    if 'scene_description' in result:
                        st.write(f"**Scene:** {result['scene_description']}")
                    
                    if 'detections' in result and result['detections']:
                        objects = [d.get('label', 'Unknown') for d in result['detections']]
                        st.write(f"**Objects:** {', '.join(objects)}")
    
    def run(self):
        """Main application runner"""
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "📄 Navigation",
            ["🎥 Live Detection", "📊 Analytics"]
        )
        
        if page == "🎥 Live Detection":
            prompt_type = self.render_sidebar()
            self.render_main_interface(prompt_type)
        elif page == "📊 Analytics":
            self.render_analytics_page()


def main():
    """Main function to run the Streamlit app"""
    try:
        app = StreamlitVLMApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        logger.error(f"App error: {e}")


if __name__ == "__main__":
    main() 