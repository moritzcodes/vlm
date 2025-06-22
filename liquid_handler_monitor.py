#!/usr/bin/env python3
"""
🧪 Liquid Handler Robot Monitor with AI-Powered Error Tracking
Advanced monitoring system for laboratory liquid handling robots
- Real-time procedure verification
- Color coding validation  
- Error detection and logging
- Compliance monitoring
"""

import cv2
import os
import time
import threading
import queue
import json
import logging
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from PIL import Image
from dotenv import load_dotenv

# Import for NATS messaging sketch
import nats
import asyncio

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('liquid_handler_monitor.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ErrorEvent:
    """Data class for tracking error events"""
    timestamp: str
    error_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    procedure_step: str
    expected_state: str
    actual_state: str
    confidence: float
    image_path: Optional[str] = None

@dataclass
class ProcedureStep:
    """Definition of a procedure step"""
    step_id: str
    name: str
    description: str
    expected_colors: List[str]
    expected_positions: List[str]
    duration_range: Tuple[int, int]  # min, max seconds
    critical: bool = False

@dataclass
class TrackedObject:
    """Data class for tracked objects"""
    object_id: int
    object_type: str  # 'well_plate', 'pipette', 'container', 'robot_arm'
    position: Tuple[int, int, int, int]  # x, y, width, height
    center: Tuple[int, int]
    confidence: float
    color: Optional[str] = None
    label: Optional[str] = None
    last_seen: float = 0

@dataclass
class WellPlateInfo:
    """Information about detected well plate"""
    position: Tuple[int, int, int, int]  # x, y, width, height
    wells: Dict[str, Tuple[int, int]]  # well_id -> (x, y) center coordinates
    grid_size: Tuple[int, int]  # rows, cols
    well_size: int  # approximate well diameter

class LiquidHandlerMonitor:
    def __init__(self):
        self.cap = None
        self.model = None
        self.running = False
        self.analysis_queue = queue.Queue(maxsize=3)
        self.last_analysis_time = 0
        self.analysis_interval = 1.5  # Analyze every 1.5 seconds (faster)
        
        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.current_procedure = None
        self.current_step = 0
        self.step_start_time = None
        
        # Procedure definitions
        self.procedures = self._load_procedures()
        
        # Camera detection
        self.camera_index = None
        
        # Status tracking
        self.last_description = "Initializing..."
        self.last_analysis_result = {}
        
        # Smart analysis tracking
        self.skip_analysis_count = 0
        self.consecutive_arm_blocking = 0
        
        # Simple experimental goals
        self.current_goal = ""
        self.available_goals = {
            "1": "Wells should turn purple (blue + red mixing)",
            "2": "Column 1 wells get blue water then red reagent", 
            "3": "Proper PCR master mix without air bubbles",
            "4": "Clean tip washing between reagents",
            "5": "No spills or contamination",
            "6": "All pipetting operations complete successfully"
        }
        
        # Image saving for error documentation
        self.save_error_images = True
        self.error_image_dir = "error_images"
        os.makedirs(self.error_image_dir, exist_ok=True)
        
        # Well tracking for blue-red mixing
        self.well_status = {}  # Track individual well states
        self.failed_wells = []  # Track wells that failed mixing
        
        # Computer vision tracking
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.well_plate_info: Optional[WellPlateInfo] = None
        self.cv_enabled = True
        
        # Opentrons protocol tracking
        self.protocol_labware = {}
        self.expected_positions = {
            'plate': (1, 'nest_96_wellplate_2ml_deep'),      # Position 1
            'reservoir': (2, 'nest_12_reservoir_15ml'),       # Position 2  
            'tips': (4, 'opentrons_flex_96_filtertiprack_200ul')  # Position 4
        }
        self.protocol_step_tracking = {
            'current_operation': None,
            'target_wells': [],
            'source_well': None
        }
        
        # Detection parameters
        self.well_plate_detector = None
        self.setup_cv_detectors()
        
        # Visual overlay settings
        self.show_overlays = True
        self.show_well_grid = True
        self.show_object_labels = True
        
        # Frame control and filtering
        self.crop_enabled = False
        self.crop_region = None  # (x, y, width, height)
        self.current_filter = "none"  # none, grayscale, hsv, blur, edge, contrast, brightness
        self.filter_params = {
            'blur_kernel': 5,
            'edge_threshold1': 50,
            'edge_threshold2': 150,
            'contrast_alpha': 1.0,  # 1.0-3.0
            'brightness_beta': 0,   # -100 to 100
            'gamma': 1.0           # 0.1-3.0
        }
        
        # Modern crop interface state
        self.crop_mode = "off"  # off, ui_active, preview, adjusting
        self.crop_ui_visible = False
        self.crop_ui_position = (50, 100)  # Top-left of crop UI panel
        self.crop_buttons = []  # UI buttons for crop operations
        self.crop_sliders = {}  # Position and size sliders
        self.crop_preview_enabled = True
        self.crop_grid_overlay = True
        self.crop_snap_to_grid = False
        
        # Crop state
        self.crop_handles = []
        self.selected_handle = None
        self.crop_presets = {
            'full': (0.0, 0.0, 1.0, 1.0),         # Full frame
            'center': (0.25, 0.25, 0.5, 0.5),     # Center 50%
            'wellplate': (0.2, 0.3, 0.6, 0.4),   # Typical well plate area
            'top_half': (0.0, 0.0, 1.0, 0.5),    # Top half
            'bottom_half': (0.0, 0.5, 1.0, 0.5), # Bottom half
            'left_half': (0.0, 0.0, 0.5, 1.0),   # Left half
            'right_half': (0.5, 0.0, 0.5, 1.0),  # Right half
            'thirds_left': (0.0, 0.0, 0.33, 1.0), # Left third
            'thirds_center': (0.33, 0.0, 0.34, 1.0), # Center third
            'thirds_right': (0.67, 0.0, 0.33, 1.0)   # Right third
        }
        self.crop_history = []
        
        # UI interaction state
        self.mouse_pos = (0, 0)
        self.ui_hover_element = None
        self.ui_active_element = None
        self.ui_drag_start = None

        # NATS Messaging for error reporting
        self.nats_client = None
        self.sent_error_count = 0
        self.nats_message_queue = queue.Queue()
        self._setup_nats_connection()

    def _setup_nats_connection(self):
        """Setup NATS connection with a single dedicated event loop"""
        def nats_worker():
            try:
                # Create one event loop for all NATS operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def run_nats():
                    # Connect to NATS
                    self.nats_client = await nats.connect("nats://localhost:4222")
                    logging.info("Connected to NATS server for error broadcasting.")
                    
                    # Process messages from queue
                    while True:
                        try:
                            # Check for new messages to send (non-blocking)
                            try:
                                subject, message = self.nats_message_queue.get_nowait()
                                if self.nats_client and self.nats_client.is_connected:
                                    await self.nats_client.publish(subject, message.encode())
                                    logging.info(f"Broadcasted error to NATS: {subject}")
                                self.nats_message_queue.task_done()
                            except queue.Empty:
                                pass
                            
                            # Small delay to prevent busy waiting
                            await asyncio.sleep(0.1)
                            
                        except Exception as e:
                            logging.error(f"Error processing NATS message: {e}")
                            await asyncio.sleep(1)
                
                loop.run_until_complete(run_nats())
                
            except Exception as e:
                logging.error(f"NATS worker thread failed: {e}")
        
        nats_thread = threading.Thread(target=nats_worker, daemon=True)
        nats_thread.start()

    def _send_error_to_nats(self, error_event: ErrorEvent):
        """Send error as a simple string message to NATS via queue"""
        try:
            # Create a simple string message
            error_message = f"ERROR: {error_event.error_type} | {error_event.severity} | {error_event.description} | Step: {error_event.procedure_step} | Time: {error_event.timestamp}"
            nats_subject = f"lab.error.liquid_handler.{error_event.error_type.replace(' ', '_')}"
            
            # Put message in queue for the NATS worker to send
            self.nats_message_queue.put((nats_subject, error_message))
            
        except Exception as e:
            logging.error(f"Error queuing NATS message: {e}")

    def _load_procedures(self) -> Dict[str, List[ProcedureStep]]:
        """Load standard operating procedures"""
        return {
            "blue_red_mixing": [
                ProcedureStep(
                    step_id="BRM001",
                    name="Add Blue Liquid",
                    description="Add blue liquid to all wells first",
                    expected_colors=["blue"],
                    expected_positions=["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6"],
                    duration_range=(60, 180),
                    critical=True
                ),
                ProcedureStep(
                    step_id="BRM002",
                    name="Mix with Red",
                    description="Add red liquid and mix with blue in each well",
                    expected_colors=["blue", "red", "purple", "mixed"],
                    expected_positions=["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6"],
                    duration_range=(90, 240),
                    critical=True
                ),
                ProcedureStep(
                    step_id="BRM003",
                    name="Verify Mixing",
                    description="Verify all wells have proper purple/mixed color",
                    expected_colors=["purple", "mixed"],
                    expected_positions=["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "B5", "B6"],
                    duration_range=(30, 90),
                    critical=True
                )
            ],
            "sample_preparation": [
                ProcedureStep(
                    step_id="SP001",
                    name="Load Samples",
                    description="Load red samples into positions B1-B2",
                    expected_colors=["red_sample"],
                    expected_positions=["position_B1", "position_B2"],
                    duration_range=(30, 120),
                    critical=True
                ),
                ProcedureStep(
                    step_id="SP002", 
                    name="Add Buffer",
                    description="Add yellow buffer to sample positions",
                    expected_colors=["yellow_buffer", "red_sample"],
                    expected_positions=["position_B1", "position_B2"],
                    duration_range=(45, 90)
                ),
                ProcedureStep(
                    step_id="SP003",
                    name="Mix and Transfer",
                    description="Mix samples and transfer to analysis positions",
                    expected_colors=["red_sample", "yellow_buffer"],
                    expected_positions=["position_A1", "position_A2"],
                    duration_range=(60, 180)
                ),
                ProcedureStep(
                    step_id="SP004",
                    name="Wash Cycle",
                    description="Clean pipette tips with clear water",
                    expected_colors=["clear_water"],
                    expected_positions=["wash_station"],
                    duration_range=(15, 45)
                )
            ],
            "pcr_master_mix": [
                ProcedureStep(
                    step_id="PCR001",
                    name="Pre-wet with water",
                    description="Pre-wet column 1 wells (A1-H1) with 50µL blue water",
                    expected_colors=["blue"],
                    expected_positions=["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"],
                    duration_range=(60, 120),
                    critical=True
                ),
                ProcedureStep(
                    step_id="PCR002", 
                    name="Add viscous master-mix",
                    description="Add 100µL red master-mix (50% glycerol) to A1, C1, E1, G1 - BUG: flow rates too fast",
                    expected_colors=["red", "purple"],
                    expected_positions=["A1", "C1", "E1", "G1"],
                    duration_range=(90, 180),
                    critical=True
                ),
                ProcedureStep(
                    step_id="PCR003",
                    name="Top-off with tracking dye",
                    description="Add 10µL blue tracking dye to all column 1 wells (A1-H1)",
                    expected_colors=["blue", "purple", "mixed"],
                    expected_positions=["A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1"],
                    duration_range=(30, 90),
                    critical=False
                )
            ]
        }

    def find_iphone_camera(self):
        """Auto-detect iPhone camera index"""
        print("🔍 Searching for iPhone camera...")
        
        # First try index 1 (common iPhone camera)
        print("📱 Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                cap.release()
                print(f"✅ iPhone camera found at index 1: {width}x{height}")
                return 1
            cap.release()
        
        # Fallback to auto-detection
        for i in range(5):
            if i == 1:
                continue
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    if width >= 640 and height >= 480:
                        cap.release()
                        print(f"📱 Camera found at index {i}: {width}x{height}")
                        return i
                cap.release()
        
        print("❌ No suitable camera found")
        return None

    def setup_ai_model(self):
        """Setup Google Gen AI SDK for Gemini 2.5 Pro"""
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            return None
        
        try:
            from google import genai
            
            client = genai.Client(
                vertexai=False, 
                api_key=api_key
            )
            print("🤖 Google Gen AI Gemini 2.5 Pro loaded for liquid handler monitoring")
            return client
            
        except Exception as e:
            print(f"❌ Google Gen AI setup failed: {e}")
            return None

    def setup(self):
        """Setup monitor components"""
        # Find camera
        self.camera_index = self.find_iphone_camera()
        if self.camera_index is None:
            return False
        
        # Setup AI model
        self.model = self.setup_ai_model()
        if self.model is None:
            return False
        
        return True

    def connect_camera(self):
        """Connect to camera"""
        print(f"📱 Connecting to camera at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("❌ Camera connection failed")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Cannot capture frames")
            return False
        
        print(f"✅ Connected: {frame.shape[1]}x{frame.shape[0]}")
        return True

    def analyze_frame_with_ai(self, frame):
        """Send frame to AI for comprehensive analysis"""
        try:
            # Resize frame for faster processing while maintaining quality
            height, width = frame.shape[:2]
            if width > 1280:  # Downscale large frames
                scale_factor = 1280 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            import io
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=75)  # Reduced quality for speed
            img_bytes = img_buffer.getvalue()
            
            from google.genai import types
            
            image_part = types.Part.from_bytes(
                data=img_bytes, 
                mime_type="image/jpeg"
            )
            
            # Get current procedure context
            procedure_context = ""
            if self.current_procedure and self.current_step < len(self.procedures[self.current_procedure]):
                step = self.procedures[self.current_procedure][self.current_step]
                procedure_context = f"""
CURRENT PROCEDURE: {self.current_procedure}
CURRENT STEP: {step.name} ({step.description})
EXPECTED COLORS: {', '.join(step.expected_colors)}
EXPECTED POSITIONS: {', '.join(step.expected_positions)}
"""

            # Add simple goal context
            experiment_context = ""
            if self.current_goal:
                experiment_context = f"""
EXPERIMENT GOAL: {self.current_goal}
MONITOR FOR: Success/failure relative to this goal
"""
            
            # Add Opentrons labware context
            labware_context = """
OPENTRONS LABWARE CONTEXT:
- Position 1: nest_96_wellplate_2ml_deep (96 deep well plate)  
- Position 2: nest_12_reservoir_15ml (12-channel reservoir for reagents)
- Position 4: opentrons_flex_96_filtertiprack_200ul (200µL filter tip rack)

PROTOCOL SPECIFICS (from failing-protocol-5.py):
- Water (blue) is in reservoir well 0
- Master-mix (red, 50% glycerol) is in reservoir well 1  
- Target wells: Column 1 (A1, B1, C1, D1, E1, F1, G1, H1)
- Viscous liquid issue: Flow rates too fast for glycerol, causes poor mixing
"""

            prompt = f"""LIQUID HANDLER MONITORING - FAST ANALYSIS
{procedure_context}
{experiment_context}

ANALYZE:
1. STATUS: Robot arm blocking view? (YES/NO)
2. PLATE: Can see well contents clearly? (YES/NO) 
3. WELLS: Color in each well (A1-A6, B1-B6)
4. ERRORS: Any failed wells or safety issues?
5. COMPLIANCE: Following procedure correctly?

CONTEXT:
- Blue+Red mixing → expect purple result
- Robot arm blocking = normal operation
- Focus on visible well plate when clear

FORMAT:
STATUS: [NORMAL/WARNING/ERROR/CRITICAL]
ARM_BLOCKING: [YES/NO]
PLATE_VISIBLE: [YES/NO]
WELL_ANALYSIS: [A1:color, A2:color, A3:color, A4:color, A5:color, A6:color, B1:color, B2:color, B3:color, B4:color, B5:color, B6:color]
FAILED_WELLS: [wells with issues or "NONE"]
COMPLIANCE: [YES/NO]
ERRORS: [issues found or "NONE"]
SAFETY: [OK or describe issue]
CONFIDENCE: [0.0-1.0]
DESCRIPTION: [brief status]"""

            response = self.model.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text=prompt),
                        image_part
                    ])
                ]
            )
            
            return self.parse_ai_response(response.text)
            
        except Exception as e:
            logging.error(f"AI analysis failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "confidence": 0.0
            }

    def parse_ai_response(self, response_text: str) -> dict:
        """Parse AI response into structured data"""
        result = {
            "status": "UNKNOWN",
            "arm_blocking": False,
            "plate_visible": False,
            "well_analysis": {},
            "failed_wells": [],
            "compliance": False,
            "errors": [],
            "safety": "UNKNOWN",
            "confidence": 0.0,
            "description": "",
            "raw_response": response_text
        }
        
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STATUS:'):
                    result["status"] = line.replace('STATUS:', '').strip()
                elif line.startswith('ARM_BLOCKING:'):
                    arm_str = line.replace('ARM_BLOCKING:', '').strip().upper()
                    result["arm_blocking"] = arm_str.startswith('YES')
                elif line.startswith('PLATE_VISIBLE:'):
                    plate_str = line.replace('PLATE_VISIBLE:', '').strip().upper()
                    result["plate_visible"] = plate_str.startswith('YES')
                elif line.startswith('WELL_ANALYSIS:'):
                    wells_str = line.replace('WELL_ANALYSIS:', '').strip()
                    # Parse well analysis: A1:blue, A2:red, etc.
                    well_pairs = [w.strip() for w in wells_str.split(',') if ':' in w]
                    for pair in well_pairs:
                        if ':' in pair:
                            well, color = pair.split(':', 1)
                            result["well_analysis"][well.strip()] = color.strip()
                elif line.startswith('FAILED_WELLS:'):
                    failed_str = line.replace('FAILED_WELLS:', '').strip()
                    if failed_str.upper() != "NONE":
                        result["failed_wells"] = [w.strip() for w in failed_str.split(',') if w.strip()]
                elif line.startswith('COMPLIANCE:'):
                    compliance_str = line.replace('COMPLIANCE:', '').strip().upper()
                    result["compliance"] = compliance_str.startswith('YES')
                elif line.startswith('ERRORS:'):
                    errors_str = line.replace('ERRORS:', '').strip()
                    if errors_str.upper() != "NONE":
                        result["errors"] = [e.strip() for e in errors_str.split(',') if e.strip()]
                elif line.startswith('SAFETY:'):
                    result["safety"] = line.replace('SAFETY:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        result["confidence"] = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        result["confidence"] = 0.5
                elif line.startswith('DESCRIPTION:'):
                    result["description"] = line.replace('DESCRIPTION:', '').strip()
                    
        except Exception as e:
            logging.error(f"Failed to parse AI response: {e}")
            result["description"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
        return result

    def setup_cv_detectors(self):
        """Setup computer vision detectors"""
        try:
            # Setup contour-based detection for well plates and containers
            self.well_plate_detector = cv2.SimpleBlobDetector_create()
            
            # Parameters for blob detection (wells, containers)
            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 50
            params.maxArea = 5000
            params.filterByCircularity = True
            params.minCircularity = 0.3
            params.filterByConvexity = True
            params.minConvexity = 0.3
            
            self.blob_detector = cv2.SimpleBlobDetector_create(params)
            
            print("✅ Computer vision detectors initialized")
            
        except Exception as e:
            print(f"⚠️  CV detector setup failed: {e}")
            self.cv_enabled = False

    def detect_well_plate(self, frame):
        """Detect well plate using computer vision"""
        if not self.cv_enabled:
            return None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold to find well plate outline
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contour that could be a well plate
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum area for well plate
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # Roughly rectangular
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Well plates are typically rectangular
                        if 0.5 < aspect_ratio < 2.0 and w > 200 and h > 150:
                            return self.analyze_well_plate_region(frame, (x, y, w, h))
            
            return None
            
        except Exception as e:
            logging.error(f"Well plate detection failed: {e}")
            return None

    def analyze_well_plate_region(self, frame, plate_region):
        """Analyze detected well plate region to find individual wells"""
        x, y, w, h = plate_region
        
        # Extract plate region
        plate_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect circular wells using HoughCircles
        circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=5, maxRadius=50)
        
        wells = {}
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Sort circles to create grid pattern
            circle_list = []
            for circle in circles[0, :]:
                cx, cy, r = circle
                # Convert back to full frame coordinates
                full_x = x + cx
                full_y = y + cy
                circle_list.append((full_x, full_y, r))
            
            # Sort by y then x to create grid
            circle_list.sort(key=lambda c: (c[1], c[0]))
            
            # Create well labels (A1-A6, B1-B6, etc.)
            rows = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            row_groups = []
            current_row = []
            last_y = -1
            
            for cx, cy, r in circle_list:
                if last_y == -1 or abs(cy - last_y) < 30:  # Same row
                    current_row.append((cx, cy, r))
                    last_y = cy
                else:  # New row
                    if current_row:
                        row_groups.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [(cx, cy, r)]
                    last_y = cy
            
            if current_row:
                row_groups.append(sorted(current_row, key=lambda c: c[0]))
            
            # Assign well labels
            for row_idx, row in enumerate(row_groups[:8]):  # Max 8 rows
                for col_idx, (cx, cy, r) in enumerate(row[:12]):  # Max 12 columns
                    well_id = f"{rows[row_idx]}{col_idx + 1}"
                    wells[well_id] = (cx, cy)
        
        # Estimate grid size and well size
        if wells:
            grid_rows = len(row_groups)
            grid_cols = max(len(row) for row in row_groups) if row_groups else 0
            avg_radius = np.mean([r for _, _, r in circle_list]) if circle_list else 10
            
            return WellPlateInfo(
                position=plate_region,
                wells=wells,
                grid_size=(grid_rows, grid_cols),
                well_size=int(avg_radius * 2)
            )
        
        return None

    def detect_objects(self, frame):
        """Detect and track containers only"""
        if not self.cv_enabled:
            return
            
        current_time = time.time()
        
        # Clear old tracked objects
        self.tracked_objects.clear()
        
        # Only detect containers (simplified)
        self.detect_containers_simplified(frame)

    def detect_pipettes(self, frame):
        """Detect pipette tips and pipette system"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for vertical lines (pipettes are typically vertical)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Check if line is roughly vertical (pipette orientation)
                    angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                    if angle > 80 or angle < 10:  # Vertical or horizontal
                        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if length > 40:  # Minimum pipette length
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Create tracked object
                            obj = TrackedObject(
                                object_id=self.next_object_id,
                                object_type='pipette',
                                position=(min(x1,x2)-5, min(y1,y2)-5, 10, int(length)),
                                center=(center_x, center_y),
                                confidence=0.7,
                                label='Pipette',
                                last_seen=time.time()
                            )
                            
                            self.tracked_objects[self.next_object_id] = obj
                            self.next_object_id += 1
                            
        except Exception as e:
            logging.error(f"Pipette detection failed: {e}")

    def detect_containers(self, frame):
        """Detect liquid containers and bottles"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use blob detection for round containers
            keypoints = self.blob_detector.detect(gray)
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                size = int(kp.size)
                
                # Filter for reasonable container sizes
                if 20 < size < 200:
                    obj = TrackedObject(
                        object_id=self.next_object_id,
                        object_type='container',
                        position=(x-size//2, y-size//2, size, size),
                        center=(x, y),
                        confidence=0.6,
                        label='Container',
                        last_seen=time.time()
                    )
                    
                    self.tracked_objects[self.next_object_id] = obj
                    self.next_object_id += 1
                    
        except Exception as e:
            logging.error(f"Container detection failed: {e}")

    def detect_containers_simplified(self, frame):
        """Detect Opentrons labware based on protocol specifications"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            height, width = frame.shape[:2]
            
            # Enhanced preprocessing for better labware detection
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            
            # Use multiple thresholding techniques
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Combine thresholds
            combined_thresh = cv2.bitwise_or(thresh1, thresh2)
            
            # Morphological operations to clean up
            kernel = np.ones((3,3), np.uint8)
            cleaned = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze frame regions for expected Opentrons positions
            labware_candidates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 3000 < area < 80000:  # Opentrons labware size range
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Skip very edge objects (likely partial)
                    if x < 20 or y < 20 or (x + w) > (width - 20) or (y + h) > (height - 20):
                        continue
                    
                    # Calculate metrics
                    aspect_ratio = w / h
                    extent = area / (w * h)
                    
                    # Look for rectangular labware
                    if 0.4 < aspect_ratio < 4.0 and extent > 0.3:
                        # Extract ROI for analysis
                        roi = frame[y:y+h, x:x+w]
                        roi_hsv = hsv[y:y+h, x:x+w]
                        
                        # Determine deck position and labware type
                        deck_position = self.estimate_deck_position(x, y, w, h, width, height)
                        labware_type, confidence = self.classify_opentrons_labware(roi, roi_hsv, w, h, aspect_ratio, deck_position)
                        
                        if confidence > 0.4:
                            labware_candidates.append({
                                'position': (x, y, w, h),
                                'center': (x + w//2, y + h//2),
                                'confidence': confidence,
                                'area': area,
                                'type': labware_type,
                                'deck_position': deck_position
                            })
            
            # Sort by confidence and prioritize expected positions
            labware_candidates.sort(key=lambda c: c['confidence'] * (1.2 if c['deck_position'] in [1, 2, 4] else 1.0), reverse=True)
            
            # Select best candidates, prioritizing different types
            selected_labware = []
            used_positions = set()
            
            # First pass: one labware per expected position
            for labware in labware_candidates:
                pos = labware['deck_position']
                if pos not in used_positions and pos in [1, 2, 4] and len(selected_labware) < 3:
                    selected_labware.append(labware)
                    used_positions.add(pos)
            
            # Second pass: fill remaining slots with highest confidence
            for labware in labware_candidates:
                if len(selected_labware) >= 3:
                    break
                if labware not in selected_labware:
                    selected_labware.append(labware)
            
            # Create tracked objects with protocol context
            for labware in selected_labware[:3]:
                # Enhanced label with position info
                label = f"{labware['type']} (Pos {labware['deck_position']})"
                
                obj = TrackedObject(
                    object_id=self.next_object_id,
                    object_type='container',
                    position=labware['position'],
                    center=labware['center'],
                    confidence=labware['confidence'],
                    label=label,
                    last_seen=time.time()
                )
                
                self.tracked_objects[self.next_object_id] = obj
                self.protocol_labware[labware['deck_position']] = obj
                self.next_object_id += 1
                
        except Exception as e:
            logging.error(f"Opentrons labware detection failed: {e}")

    def estimate_deck_position(self, x, y, w, h, frame_width, frame_height):
        """Estimate Opentrons deck position based on location in frame"""
        # Rough mapping of frame regions to deck positions
        # Opentrons Flex has positions 1-12 in a 3x4 grid
        
        center_x = x + w//2
        center_y = y + h//2
        
        # Normalize coordinates
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height
        
        # Map to deck positions (rough estimation)
        if norm_x < 0.33:  # Left column
            if norm_y < 0.25:
                return 10
            elif norm_y < 0.5:
                return 7
            elif norm_y < 0.75:
                return 4
            else:
                return 1
        elif norm_x < 0.67:  # Middle column
            if norm_y < 0.25:
                return 11
            elif norm_y < 0.5:
                return 8
            elif norm_y < 0.75:
                return 5
            else:
                return 2
        else:  # Right column
            if norm_y < 0.25:
                return 12
            elif norm_y < 0.5:
                return 9
            elif norm_y < 0.75:
                return 6
            else:
                return 3

    def classify_opentrons_labware(self, roi, roi_hsv, width, height, aspect_ratio, deck_position):
        """Classify Opentrons labware type based on visual characteristics and expected position"""
        try:
            # Analyze color and texture
            h_mean = np.mean(roi_hsv[:, :, 0])
            s_mean = np.mean(roi_hsv[:, :, 1])
            v_mean = np.mean(roi_hsv[:, :, 2])
            
            # Base confidence from shape analysis
            base_confidence = 0.5
            
            # Check against expected labware at this position
            if deck_position == 1:  # Expected: nest_96_wellplate_2ml_deep
                # Deep well plates are typically darker, larger
                if 0.8 < aspect_ratio < 1.5 and width > 80 and height > 80:
                    base_confidence = 0.9
                return "96 Deep Well Plate", base_confidence
                
            elif deck_position == 2:  # Expected: nest_12_reservoir_15ml
                # Reservoirs are typically longer/wider aspect ratio
                if aspect_ratio > 1.2 and width > 100:
                    base_confidence = 0.9
                return "12-Channel Reservoir", base_confidence
                
            elif deck_position == 4:  # Expected: opentrons_flex_96_filtertiprack_200ul
                # Tip racks have distinctive pattern, moderate size
                if 0.7 < aspect_ratio < 1.3 and 60 < width < 120:
                    # Check for tip pattern (multiple small circles/holes)
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    circles = cv2.HoughCircles(gray_roi, cv2.HOUGH_GRADIENT, 1, 10,
                                             param1=50, param2=15, minRadius=2, maxRadius=8)
                    if circles is not None and len(circles[0]) > 20:  # Many tips visible
                        base_confidence = 0.95
                return "200µL Filter Tip Rack", base_confidence
            
            # General classification for other positions
            else:
                # Analyze general characteristics
                if aspect_ratio < 0.7:  # Tall container
                    return "Vertical Labware", 0.4
                elif aspect_ratio > 2.0:  # Wide container
                    return "Horizontal Labware", 0.4
                elif 0.8 < aspect_ratio < 1.3:  # Square-ish
                    if width > 100:
                        return "Large Plate", 0.6
                    else:
                        return "Small Plate", 0.5
                else:
                    return "Unknown Labware", 0.3
                    
        except Exception as e:
            return f"Labware (Pos {deck_position})", 0.2

    def classify_container(self, roi, roi_hsv, width, height, aspect_ratio):
        """Legacy fallback container classification"""
        return "Legacy Container", 0.3

    def detect_robot_arm(self, frame):
        """Detect robot arm components"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for large moving objects (robot arm segments)
            # Use contour detection for robot arm
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Large objects likely to be robot arm
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (robot arms are typically elongated)
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio > 2:  # Elongated object
                        obj = TrackedObject(
                            object_id=self.next_object_id,
                            object_type='robot_arm',
                            position=(x, y, w, h),
                            center=(x + w//2, y + h//2),
                            confidence=0.8,
                            label='Robot Arm',
                            last_seen=time.time()
                        )
                        
                        self.tracked_objects[self.next_object_id] = obj
                        self.next_object_id += 1
                        
        except Exception as e:
            logging.error(f"Robot arm detection failed: {e}")

    def draw_cv_overlays(self, frame):
        """Draw computer vision overlays on the frame - DISABLED"""
        # CV overlays disabled - using VLM feedback instead
        return

    def draw_well_plate_overlay(self, frame):
        """Draw well plate grid overlay"""
        if not self.well_plate_info:
            return
            
        plate = self.well_plate_info
        x, y, w, h = plate.position
        
        # Draw well plate boundary
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Well Plate ({plate.grid_size[0]}x{plate.grid_size[1]})", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw individual wells
        for well_id, (wx, wy) in plate.wells.items():
            # Get well color status
            well_color = self.well_status.get(well_id, "unknown")
            is_failed = well_id in self.failed_wells
            
            # Choose overlay color
            if is_failed:
                color = (0, 0, 255)  # Red for failed wells
            elif well_color != "unknown":
                color = (0, 255, 0)  # Green for wells with known status
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            # Draw well circle
            radius = plate.well_size // 2
            cv2.circle(frame, (wx, wy), radius, color, 2)
            
            # Draw well label
            cv2.putText(frame, well_id, (wx - 10, wy - radius - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw color status if known
            if well_color != "unknown":
                cv2.putText(frame, well_color[:4], (wx - 15, wy + radius + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    def draw_tracked_objects_overlay(self, frame):
        """Draw tracked objects overlay"""
        for obj in self.tracked_objects.values():
            x, y, w, h = obj.position
            
            # Choose color based on object type
            colors = {
                'pipette': (255, 255, 0),     # Cyan
                'container': (255, 0, 255),   # Magenta  
                'robot_arm': (0, 165, 255),   # Orange
                'well_plate': (0, 255, 0)     # Green
            }
            color = colors.get(obj.object_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{obj.label} ({obj.confidence:.1f})"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            cv2.circle(frame, obj.center, 3, color, -1)

    def draw_containers_overlay(self, frame):
        """Draw subtle Iron Man-style container overlay"""
        if not self.show_object_labels:
            return
            
        for obj in self.tracked_objects.values():
            if obj.object_type == 'container':
                x, y, w, h = obj.position
                
                # Iron Man color scheme for labware
                container_colors = {
                    '96 Deep Well Plate': (100, 200, 255),    # Arc reactor blue
                    '12-Channel Reservoir': (64, 255, 128),   # Bright green
                    '200µL Filter Tip Rack': (255, 180, 64), # Warm orange
                    'Labware': (150, 220, 255),              # Light blue
                    'Legacy Container': (140, 160, 180)       # Muted blue-gray
                }
                
                # Extract labware type from label
                labware_type = obj.label.split(' (Pos')[0] if '(Pos' in obj.label else obj.label
                color = container_colors.get(labware_type, (150, 220, 255))
                
                # Subtle thin border - Iron Man style
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                
                # Arc reactor style corner indicators
                corner_size = 8
                cv2.line(frame, (x, y), (x + corner_size, y), color, 2)
                cv2.line(frame, (x, y), (x, y + corner_size), color, 2)
                cv2.line(frame, (x + w, y), (x + w - corner_size, y), color, 2)
                cv2.line(frame, (x + w, y), (x + w, y + corner_size), color, 2)
                cv2.line(frame, (x, y + h), (x + corner_size, y + h), color, 2)
                cv2.line(frame, (x, y + h), (x, y + h - corner_size), color, 2)
                cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), color, 2)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), color, 2)
                
                # Minimal label
                label_text = labware_type
                if self.current_procedure == "pcr_master_mix":
                    if "Well Plate" in labware_type:
                        label_text = "TARGET PLATE"
                    elif "Reservoir" in labware_type:
                        label_text = "REAGENTS"
                    elif "Tip Rack" in labware_type:
                        label_text = "TIPS"
                
                # Enhanced glass label with better visibility
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                if label_size[0] > 0:
                    # Enhanced background panel
                    label_height = 22
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y - label_height), (x + label_size[0] + 12, y - 2), (15, 20, 25), -1)
                    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
                    
                    # Enhanced border
                    cv2.rectangle(frame, (x, y - label_height), (x + label_size[0] + 12, y - 2), color, 2)
                    
                    # Enhanced text
                    cv2.putText(frame, label_text, (x + 6, y - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Subtle confidence dot
                if obj.confidence > 0.7:
                    cv2.circle(frame, (x + w - 8, y + 8), 3, color, -1)
                    cv2.circle(frame, (x + w - 8, y + 8), 3, (255, 255, 255), 1)

    def save_error_image(self, frame, error_type: str) -> str:
        """Save frame as error documentation"""
        if not self.save_error_images:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{error_type}_{timestamp}.jpg"
        filepath = os.path.join(self.error_image_dir, filename)
        
        cv2.imwrite(filepath, frame)
        return filepath

    def log_error_event(self, frame, error_type: str, description: str, 
                       severity: str = "MEDIUM", expected_state: str = "", 
                       actual_state: str = "", confidence: float = 0.0):
        """Log an error event with full context"""
        
        # Save error image
        image_path = self.save_error_image(frame, error_type.lower().replace(' ', '_'))
        
        # Get current procedure context
        procedure_step = "UNKNOWN"
        if self.current_procedure and self.current_step < len(self.procedures[self.current_procedure]):
            step = self.procedures[self.current_procedure][self.current_step]
            procedure_step = f"{step.step_id}: {step.name}"
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now().isoformat(),
            error_type=error_type,
            severity=severity,
            description=description,
            procedure_step=procedure_step,
            expected_state=expected_state,
            actual_state=actual_state,
            confidence=confidence,
            image_path=image_path
        )
        
        self.error_events.append(error_event)
        
        # Send new errors to NATS
        if len(self.error_events) > self.sent_error_count:
            self._send_error_to_nats(error_event)
            self.sent_error_count = len(self.error_events)
        
        # Log to file
        logging.error(f"ERROR EVENT: {error_type} - {description}")
        
        # Print to console with formatting
        severity_colors = {
            "LOW": "\033[93m",      # Yellow
            "MEDIUM": "\033[91m",   # Red  
            "HIGH": "\033[95m",     # Magenta
            "CRITICAL": "\033[41m"  # Red background
        }
        color = severity_colors.get(severity, "\033[91m")
        reset = "\033[0m"
        
        print(f"\n{color}🚨 {severity} ERROR: {error_type}{reset}")
        print(f"📝 {description}")
        print(f"🔬 Procedure: {procedure_step}")
        if image_path:
            print(f"📸 Image saved: {image_path}")
        print("-" * 60)

    def check_procedure_compliance(self, analysis_result: dict, frame):
        """Check if current state complies with expected procedure"""
        if not self.current_procedure:
            return
            
        if self.current_step >= len(self.procedures[self.current_procedure]):
            return
            
        step = self.procedures[self.current_procedure][self.current_step]
        
        # Check compliance
        if not analysis_result.get("compliance", False):
            severity = "CRITICAL" if step.critical else "HIGH"
            self.log_error_event(
                frame,
                "PROCEDURE_NON_COMPLIANCE",
                f"Step '{step.name}' not being followed correctly",
                severity=severity,
                expected_state=f"Expected: {step.description}",
                actual_state=analysis_result.get("description", "Unknown state"),
                confidence=analysis_result.get("confidence", 0.0)
            )
        
        # Check step duration
        if self.step_start_time:
            step_duration = time.time() - self.step_start_time
            min_duration, max_duration = step.duration_range
            
            if step_duration > max_duration:
                self.log_error_event(
                    frame,
                    "STEP_TIMEOUT",
                    f"Step '{step.name}' taking too long ({step_duration:.1f}s > {max_duration}s)",
                    severity="MEDIUM",
                    expected_state=f"Duration: {min_duration}-{max_duration}s",
                    actual_state=f"Duration: {step_duration:.1f}s",
                    confidence=1.0
                )

    def check_color_compliance(self, analysis_result: dict, frame):
        """Verify color coding compliance"""
        detected_colors = analysis_result.get("colors_detected", [])
        
        if not self.current_procedure or self.current_step >= len(self.procedures[self.current_procedure]):
            return
            
        step = self.procedures[self.current_procedure][self.current_step]
        expected_colors = step.expected_colors
        
        # Check for unexpected colors
        for detected in detected_colors:
            color_match = False
            for expected in expected_colors:
                if expected.lower() in detected.lower() or detected.lower() in expected.lower():
                    color_match = True
                    break
            
            if not color_match:
                self.log_error_event(
                    frame,
                    "UNEXPECTED_COLOR",
                    f"Detected unexpected color: {detected}",
                    severity="MEDIUM",
                    expected_state=f"Expected colors: {', '.join(expected_colors)}",
                    actual_state=f"Detected: {', '.join(detected_colors)}",
                    confidence=analysis_result.get("confidence", 0.0)
                )

    def check_well_compliance(self, analysis_result: dict, frame):
        """Check individual well compliance for blue-red mixing procedure"""
        if not self.current_procedure or self.current_procedure != "blue_red_mixing":
            return
            
        # Skip check if robot arm is blocking or plate not visible
        if analysis_result.get("arm_blocking", False):
            print("🤖 Robot arm active - analysis paused")
            return
            
        if not analysis_result.get("plate_visible", False):
            print("👁️  Waiting for clear plate visibility...")
            return
            
        well_analysis = analysis_result.get("well_analysis", {})
        failed_wells = analysis_result.get("failed_wells", [])
        
        if self.current_step >= len(self.procedures[self.current_procedure]):
            return
            
        step = self.procedures[self.current_procedure][self.current_step]
        
        # Update well status tracking
        for well, color in well_analysis.items():
            self.well_status[well] = color
            
        # Check for failed wells based on current step
        if failed_wells:
            for well in failed_wells:
                if well not in self.failed_wells:
                    self.failed_wells.append(well)
                    
                # Log specific well failure
                well_color = well_analysis.get(well, "unknown")
                expected_colors = step.expected_colors
                
                self.log_error_event(
                    frame,
                    "WELL_MIXING_FAILURE",
                    f"Well {well} failed - found {well_color}, expected one of: {', '.join(expected_colors)}",
                    severity="HIGH",
                    expected_state=f"Well {well}: {', '.join(expected_colors)}",
                    actual_state=f"Well {well}: {well_color}",
                    confidence=analysis_result.get("confidence", 0.0)
                )
                
                print(f"❌ WELL FAILURE: {well} has incorrect color ({well_color})")
        
        # Print well status summary
        if well_analysis:
            print(f"🧪 Well Status:")
            for row in ['A', 'B']:
                row_status = []
                for col in range(1, 7):
                    well = f"{row}{col}"
                    color = well_analysis.get(well, "empty")
                    status = "✅" if well not in self.failed_wells else "❌"
                    row_status.append(f"{well}:{color}{status}")
                print(f"   Row {row}: {' | '.join(row_status)}")
            
            if self.failed_wells:
                print(f"⚠️  Failed Wells: {', '.join(self.failed_wells)}")

    def check_safety_compliance(self, analysis_result: dict, frame):
        """Check for safety issues"""
        safety_status = analysis_result.get("safety", "UNKNOWN")
        errors = analysis_result.get("errors", [])
        
        if safety_status.upper() not in ["OK", "GOOD", "SAFE"]:
            self.log_error_event(
                frame,
                "SAFETY_CONCERN",
                f"Safety issue detected: {safety_status}",
                severity="HIGH",
                expected_state="Safe operation",
                actual_state=safety_status,
                confidence=analysis_result.get("confidence", 0.0)
            )
        
        # Check for specific errors
        for error in errors:
            if any(keyword in error.lower() for keyword in ["spill", "contamination", "leak", "overflow"]):
                self.log_error_event(
                    frame,
                    "CONTAMINATION_RISK",
                    f"Contamination risk: {error}",
                    severity="CRITICAL",
                    expected_state="Clean operation",
                    actual_state=error,
                    confidence=analysis_result.get("confidence", 0.0)
                )

    def analysis_worker(self):
        """Background thread for AI analysis and error checking"""
        while self.running:
            try:
                if not self.analysis_queue.empty():
                    frame = self.analysis_queue.get(timeout=1)
                    
                    print("🧠 Fast AI analysis...")
                    start_time = time.time()
                    analysis_result = self.analyze_frame_with_ai(frame)
                    analysis_time = time.time() - start_time
                    print(f"⚡ Analysis completed in {analysis_time:.2f}s")
                    
                    # Store result
                    self.last_analysis_result = analysis_result
                    self.last_description = analysis_result.get("description", "Analysis failed")
                    
                    # Only do expensive compliance checks if not just robot arm blocking
                    arm_blocking = analysis_result.get("arm_blocking", False)
                    if not arm_blocking:
                        # Check compliance (these functions may add to self.error_events)
                        self.check_procedure_compliance(analysis_result, frame)
                        self.check_color_compliance(analysis_result, frame)
                        self.check_well_compliance(analysis_result, frame)
                        self.check_safety_compliance(analysis_result, frame)

                    else:
                        # Skip expensive compliance checks when arm is blocking
                        print("🤖 Robot arm active - skipping detailed compliance checks")
                    
                    # Print status
                    status = analysis_result.get("status", "UNKNOWN")
                    arm_blocking = analysis_result.get("arm_blocking", False)
                    plate_visible = analysis_result.get("plate_visible", False)
                    
                    status_colors = {
                        "NORMAL": "\033[92m",   # Green
                        "WARNING": "\033[93m",  # Yellow
                        "ERROR": "\033[91m",    # Red
                        "CRITICAL": "\033[41m"  # Red background
                    }
                    color = status_colors.get(status, "\033[0m")
                    reset = "\033[0m"
                    
                    print(f"{color}🔬 Status: {status}{reset}")
                    
                    if arm_blocking:
                        print("🤖 Robot arm active (normal operation)")
                    elif not plate_visible:
                        print("👁️  Waiting for clear plate view...")
                    else:
                        print("✅ Clear plate view available")
                    
                    if analysis_result.get("failed_wells"):
                        failed = ', '.join(analysis_result["failed_wells"])
                        print(f"❌ Failed wells: {failed}")
                    
                    print(f"📝 {self.last_description[:100]}...")
                    print("=" * 60)
                    
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Analysis worker error: {e}")
                time.sleep(1)

    def start_procedure(self, procedure_name: str):
        """Start monitoring a specific procedure"""
        if procedure_name not in self.procedures:
            print(f"❌ Unknown procedure: {procedure_name}")
            return False
            
        self.current_procedure = procedure_name
        self.current_step = 0
        self.step_start_time = time.time()
        
        # Reset well tracking for new procedure
        if procedure_name == "blue_red_mixing":
            self.well_status = {}
            self.failed_wells = []
            print("🧪 Well tracking initialized for blue-red mixing procedure")
        
        print(f"🧬 Starting procedure: {procedure_name}")
        step = self.procedures[procedure_name][0]
        print(f"📋 Step 1: {step.name} - {step.description}")
        
        if procedure_name == "blue_red_mixing":
            print("🔵 Expected flow: Blue liquid → Red liquid → Purple mix")
            print("📊 Wells will be monitored individually (A1-A6, B1-B6)")
        
        return True

    def next_step(self):
        """Advance to next procedure step"""
        if not self.current_procedure:
            print("❌ No active procedure")
            return False
            
        if self.current_step >= len(self.procedures[self.current_procedure]) - 1:
            print("✅ Procedure completed!")
            self.current_procedure = None
            self.current_step = 0
            self.step_start_time = None
            return True
            
        self.current_step += 1
        self.step_start_time = time.time()
        
        step = self.procedures[self.current_procedure][self.current_step]
        print(f"📋 Step {self.current_step + 1}: {step.name} - {step.description}")
        
        return True

    def add_overlay(self, frame, frame_count):
        """Add professional monitoring overlay to video"""
        # Run computer vision detection
        if self.cv_enabled and frame_count % 10 == 0:
            self.detect_objects(frame)
        
        # Skip CV overlays - focus on VLM feedback integration
        
        # Draw professional UI overlay
        self.draw_professional_ui(frame)

    def draw_professional_ui(self, frame):
        """Draw a subtle Iron Man-inspired UI overlay"""
        height, width = frame.shape[:2]
        
        # Iron Man color scheme - subtle and elegant
        ui_colors = {
            'background': (15, 20, 25),      # Very dark blue
            'panel': (25, 35, 45),           # Subtle dark panel
            'accent': (100, 200, 255),       # Arc reactor blue
            'accent_glow': (150, 220, 255),  # Brighter glow
            'text_primary': (240, 245, 250), # Soft white
            'text_secondary': (140, 160, 180), # Muted blue-gray
            'success': (64, 255, 128),       # Bright green
            'warning': (255, 180, 64),       # Warm orange
            'error': (255, 80, 80),          # Soft red
            'critical': (255, 64, 150),      # Pink accent
            'glass': (40, 60, 80),           # Glass effect
        }
        
        # Get current status
        status = self.last_analysis_result.get("status", "UNKNOWN")
        status_color_map = {
            "NORMAL": ui_colors['success'],
            "WARNING": ui_colors['warning'],
            "ERROR": ui_colors['error'],
            "CRITICAL": ui_colors['critical'],
            "UNKNOWN": (128, 128, 128)
        }
        status_color = status_color_map.get(status, (128, 128, 128))
        
        # Draw minimal header HUD
        self.draw_minimal_header(frame, ui_colors, status, status_color)
        
        # Draw compact status indicators
        self.draw_compact_status(frame, ui_colors, status_color)
        
        # Draw subtle procedure info
        self.draw_subtle_procedure(frame, ui_colors)
        
        # Draw minimal bottom HUD
        self.draw_minimal_bottom_hud(frame, ui_colors)
        
        # Draw VLM feedback overlay
        self.draw_vlm_feedback_overlay(frame, ui_colors)

    def draw_glass_panel(self, frame, top_left, bottom_right, color, alpha=0.3):
        """Draw a subtle glass-like panel with transparency"""
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Thin border
        cv2.rectangle(frame, top_left, bottom_right, color, 1)

    def draw_minimal_header(self, frame, ui_colors, status, status_color):
        """Draw a high-resolution header HUD"""
        height, width = frame.shape[:2]
        
        # Larger top bar
        self.draw_glass_panel(frame, (0, 0), (width, 90), ui_colors['background'], 0.8)
        
        # Much larger arc reactor style status indicator
        cv2.circle(frame, (50, 45), 25, status_color, -1)
        cv2.circle(frame, (50, 45), 25, ui_colors['accent_glow'], 3)
        cv2.circle(frame, (50, 45), 18, (255, 255, 255), 3)
        cv2.circle(frame, (50, 45), 12, status_color, 2)
        
        # Larger title with bigger typography
        cv2.putText(frame, "LIQUID HANDLER MONITOR", (90, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_colors['text_primary'], 3)
        cv2.putText(frame, f"STATUS: {status}", (90, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Larger time display
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, "TIME", (width - 180, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_secondary'], 2)
        cv2.putText(frame, timestamp, (width - 180, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, ui_colors['accent'], 3)

    def draw_compact_status(self, frame, ui_colors, status_color):
        """Draw enhanced status indicators"""
        height, width = frame.shape[:2]
        
        # Much larger error panel
        self.draw_enhanced_error_panel(frame, ui_colors, width - 300, 105)
        
        # Larger confidence indicator
        self.draw_confidence_indicator(frame, ui_colors, width - 300, 190)
        
        # Larger procedure indicator on left
        if self.current_procedure:
            self.draw_procedure_indicator(frame, ui_colors, 20, 105)

    def draw_enhanced_error_panel(self, frame, ui_colors, x, y):
        """Draw much larger and more prominent error display panel"""
        error_count = len(self.error_events)
        error_color = ui_colors['error'] if error_count > 0 else ui_colors['success']
        
        panel_width = 280
        panel_height = 75
        
        # Much larger glass panel with better visibility
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.6)
        
        # Much larger status indicator
        cv2.circle(frame, (x + 30, y + 37), 15, error_color, -1)
        cv2.circle(frame, (x + 30, y + 37), 15, ui_colors['accent_glow'], 3)
        cv2.circle(frame, (x + 30, y + 37), 10, (255, 255, 255), 2)
        
        # Larger error display text
        cv2.putText(frame, "SYSTEM STATUS", (x + 55, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_primary'], 2)
        
        if error_count > 0:
            cv2.putText(frame, f"{error_count} ERRORS DETECTED", (x + 55, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
            
            # Show latest error type if available
            if self.error_events:
                latest_error = self.error_events[-1].error_type.replace('_', ' ')[:25]
                cv2.putText(frame, f"Latest: {latest_error}", (x + 10, y + 68), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_secondary'], 1)
        else:
            cv2.putText(frame, "ALL SYSTEMS OK", (x + 55, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_colors['success'], 2)

    def draw_confidence_indicator(self, frame, ui_colors, x, y):
        """Draw much larger confidence indicator"""
        confidence = self.last_analysis_result.get("confidence", 0.0)
        conf_color = ui_colors['success'] if confidence > 0.8 else ui_colors['warning'] if confidence > 0.5 else ui_colors['error']
        
        panel_width = 280
        panel_height = 55
        
        # Much larger glass panel
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.5)
        
        # Larger label
        cv2.putText(frame, "AI CONFIDENCE", (x + 15, y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_primary'], 2)
        
        # Much larger confidence bar
        bar_width = 180
        bar_height = 12
        bar_x = x + 15
        bar_y = y + 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['background'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['accent'], 2)
        
        # Fill bar with gradient effect
        fill_width = int(bar_width * confidence)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), conf_color, -1)
        
        # Larger percentage display
        cv2.putText(frame, f"{confidence:.0%}", (x + 210, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)

    def draw_procedure_indicator(self, frame, ui_colors, x, y):
        """Draw much larger procedure status panel"""
        if not self.current_procedure and not self.current_goal:
            return
            
        # Determine panel content and size
        has_procedure = bool(self.current_procedure)
        has_goal = bool(self.current_goal)
        
        panel_width = 450
        panel_height = 90 if not has_goal else 120
        
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.6)
        
        # Show experiment goal if set
        if has_goal:
            cv2.circle(frame, (x + 30, y + 25), 12, ui_colors['success'], -1)
            cv2.circle(frame, (x + 30, y + 25), 12, ui_colors['accent_glow'], 2)
            cv2.circle(frame, (x + 30, y + 25), 8, (255, 255, 255), 2)
            
            cv2.putText(frame, "EXPERIMENT GOAL", (x + 50, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_primary'], 2)
            
            # Truncate long goals for display
            goal_text = self.current_goal[:60] + "..." if len(self.current_goal) > 60 else self.current_goal
            cv2.putText(frame, goal_text, (x + 50, y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['success'], 1)
            
            # Adjust y position for procedure info if both are present
            if has_procedure:
                y += 50
        
        if not has_procedure:
            return
            
        proc_name = self.current_procedure.replace('_', ' ').upper()
        
        # Procedure indicator  
        proc_y = y + 30 if not has_goal else y + 25
        
        # Much larger procedure indicator
        cv2.circle(frame, (x + 30, proc_y), 15, ui_colors['accent'], -1)
        cv2.circle(frame, (x + 30, proc_y), 15, ui_colors['accent_glow'], 3)
        cv2.circle(frame, (x + 30, proc_y), 10, (255, 255, 255), 2)
        
        # Larger procedure name
        cv2.putText(frame, "ACTIVE PROCEDURE", (x + 55, proc_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_primary'], 2)
        
        # Much larger procedure name and step
        step_text = f"{proc_name}"
        cv2.putText(frame, step_text, (x + 55, proc_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_colors['accent'], 2)
        
        # Larger step progress
        step_progress = f"STEP {self.current_step + 1} OF {len(self.procedures[self.current_procedure])}"
        cv2.putText(frame, step_progress, (x + 55, proc_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_secondary'], 1)
        
        # Larger progress bar for steps
        bar_width = 300
        bar_height = 6
        bar_x = x + 55
        bar_y = proc_y + 48
        progress = (self.current_step + 1) / len(self.procedures[self.current_procedure])
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['background'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['accent'], 1)
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), ui_colors['accent'], -1)

    def draw_subtle_procedure(self, frame, ui_colors):
        """Draw subtle procedure progress"""
        if not self.current_procedure:
            return
            
        height, width = frame.shape[:2]
        
        # Thin progress line at bottom
        progress = (self.current_step + 1) / len(self.procedures[self.current_procedure])
        line_width = int(width * progress)
        
        cv2.line(frame, (0, height - 3), (line_width, height - 3), ui_colors['accent'], 2)
        cv2.line(frame, (0, height - 1), (width, height - 1), ui_colors['glass'], 1)

    def draw_minimal_bottom_hud(self, frame, ui_colors):
        """Draw enhanced bottom HUD with essential info"""
        height, width = frame.shape[:2]
        
        # Much larger robot arm status (only if active)
        arm_blocking = self.last_analysis_result.get("arm_blocking", False)
        if arm_blocking:
            self.draw_glass_panel(frame, (20, height - 100), (420, height - 20), ui_colors['warning'], 0.7)
            cv2.circle(frame, (50, height - 60), 15, ui_colors['warning'], -1)
            cv2.circle(frame, (50, height - 60), 15, (255, 255, 255), 3)
            cv2.putText(frame, "ROBOT ARM ACTIVE", (80, height - 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_colors['text_primary'], 2)
            cv2.putText(frame, "Analysis paused during operation", (80, height - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_secondary'], 1)
        
        # Much larger container tracking indicator
        if self.cv_enabled:
            container_count = len([obj for obj in self.tracked_objects.values() if obj.object_type == 'container'])
            if container_count > 0:
                self.draw_glass_panel(frame, (20, 220), (320, 280), ui_colors['glass'], 0.5)
                cv2.circle(frame, (45, 250), 12, ui_colors['accent'], -1)
                cv2.circle(frame, (45, 250), 12, ui_colors['accent_glow'], 2)
                cv2.putText(frame, "CONTAINER TRACKING", (70, 245), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_primary'], 2)
                cv2.putText(frame, f"{container_count} LABWARE DETECTED", (70, 265), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['accent'], 1)
        
        # Much larger wells status display (for blue-red mixing)
        if self.current_procedure == "blue_red_mixing":
            failed_wells_count = len(self.failed_wells)
            wells_color = ui_colors['error'] if failed_wells_count > 0 else ui_colors['success']
            
            panel_width = 380
            self.draw_glass_panel(frame, (width - panel_width - 20, height - 100), 
                                (width - 20, height - 20), wells_color, 0.7)
            
            cv2.circle(frame, (width - panel_width + 30, height - 60), 15, wells_color, -1)
            cv2.circle(frame, (width - panel_width + 30, height - 60), 15, (255, 255, 255), 3)
            
            if failed_wells_count > 0:
                cv2.putText(frame, f"{failed_wells_count} WELLS FAILED", (width - panel_width + 60, height - 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_colors['text_primary'], 2)
                # Show failed well names
                if len(self.failed_wells) <= 6:
                    failed_text = ", ".join(self.failed_wells)
                    cv2.putText(frame, failed_text, (width - panel_width + 60, height - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_secondary'], 1)
            else:
                cv2.putText(frame, "ALL WELLS OK", (width - panel_width + 60, height - 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_colors['text_primary'], 2)

    def draw_vlm_feedback_overlay(self, frame, ui_colors):
        """Draw intelligent VLM feedback overlay with actionable insights"""
        height, width = frame.shape[:2]
        
        # Only show VLM feedback when we have recent analysis
        if not self.last_analysis_result or not self.last_description:
            return
        
        # Get current analysis data
        status = self.last_analysis_result.get("status", "UNKNOWN")
        compliance = self.last_analysis_result.get("compliance", False)
        failed_wells = self.last_analysis_result.get("failed_wells", [])
        errors = self.last_analysis_result.get("errors", [])
        well_analysis = self.last_analysis_result.get("well_analysis", {})
        arm_blocking = self.last_analysis_result.get("arm_blocking", False)
        plate_visible = self.last_analysis_result.get("plate_visible", False)
        
        # Determine feedback priority and content
        feedback_items = self.generate_vlm_feedback_items(status, compliance, failed_wells, errors, well_analysis, arm_blocking, plate_visible)
        
        if not feedback_items:
            return
        
        # Draw smart feedback panel
        self.draw_smart_feedback_panel(frame, ui_colors, feedback_items)
        
        # Draw well status visualization (if relevant)
        if self.current_procedure and well_analysis and plate_visible:
            self.draw_well_status_grid(frame, ui_colors, well_analysis, failed_wells)

    def generate_vlm_feedback_items(self, status, compliance, failed_wells, errors, well_analysis, arm_blocking, plate_visible):
        """Generate prioritized feedback items based on VLM analysis"""
        items = []
        
        # Priority 1: Critical errors and safety issues
        if status == "CRITICAL":
            items.append({
                'type': 'critical',
                'icon': '🚨',
                'title': 'CRITICAL ISSUE DETECTED',
                'message': 'Immediate attention required',
                'action': 'Check procedure and safety'
            })
        
        # Priority 2: Procedure compliance issues
        if not compliance and not arm_blocking:
            items.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'PROCEDURE NON-COMPLIANCE',
                'message': 'Current step not being followed correctly',
                'action': 'Review procedure requirements'
            })
        
        # Priority 3: Failed wells (specific actionable feedback)
        if failed_wells:
            wells_text = ", ".join(failed_wells[:4])  # Show max 4 wells
            if len(failed_wells) > 4:
                wells_text += f" +{len(failed_wells) - 4} more"
            
            items.append({
                'type': 'error',
                'icon': '🧪',
                'title': f'{len(failed_wells)} WELLS FAILED',
                'message': f'Wells {wells_text} need attention',
                'action': 'Check liquid colors and mixing'
            })
        
        # Priority 4: Specific errors from VLM
        if errors and errors != ["NONE"]:
            for error in errors[:2]:  # Show max 2 errors
                items.append({
                    'type': 'error',
                    'icon': '❌',
                    'title': 'OPERATION ERROR',
                    'message': error[:50] + "..." if len(error) > 50 else error,
                    'action': 'Investigate and correct'
                })
        
        # Priority 5: Operational status
        if arm_blocking:
            items.append({
                'type': 'info',
                'icon': '🤖',
                'title': 'ROBOT OPERATING',
                'message': 'Analysis will resume when arm clears',
                'action': 'Monitoring paused - normal operation'
            })
        elif not plate_visible:
            items.append({
                'type': 'info',
                'icon': '👁️',
                'title': 'WAITING FOR CLEAR VIEW',
                'message': 'Cannot analyze well plate contents',
                'action': 'Ensure camera has clear view'
            })
        
        # Priority 6: Success feedback
        if status == "NORMAL" and compliance and not failed_wells and plate_visible:
            items.append({
                'type': 'success',
                'icon': '✅',
                'title': 'PROCEDURE ON TRACK',
                'message': 'All operations proceeding correctly',
                'action': 'Continue monitoring'
            })
        
        return items[:3]  # Show max 3 feedback items

    def draw_smart_feedback_panel(self, frame, ui_colors, feedback_items):
        """Draw intelligent feedback panel with actionable insights"""
        if not feedback_items:
            return
            
        height, width = frame.shape[:2]
        
        # Calculate much larger panel dimensions
        panel_height = 45 + (len(feedback_items) * 70)
        panel_width = 600
        panel_x = width - panel_width - 25
        panel_y = height // 2 - panel_height // 2
        
        # Ensure panel stays on screen
        panel_y = max(110, min(panel_y, height - panel_height - 120))
        
        # Draw much larger main feedback panel
        self.draw_glass_panel(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                             ui_colors['background'], 0.85)
        
        # Larger panel header
        cv2.putText(frame, "AI ANALYSIS FEEDBACK", (panel_x + 25, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_colors['text_primary'], 2)
        
        # Draw feedback items with more spacing
        item_y = panel_y + 55
        for item in feedback_items:
            self.draw_feedback_item(frame, ui_colors, panel_x, item_y, panel_width, item)
            item_y += 70

    def draw_feedback_item(self, frame, ui_colors, x, y, width, item):
        """Draw individual feedback item"""
        # Color coding
        colors = {
            'critical': ui_colors['critical'],
            'error': ui_colors['error'],
            'warning': ui_colors['warning'],
            'info': ui_colors['accent'],
            'success': ui_colors['success']
        }
        item_color = colors.get(item['type'], ui_colors['text_primary'])
        
        # Much larger status indicator
        cv2.circle(frame, (x + 30, y + 25), 12, item_color, -1)
        cv2.circle(frame, (x + 30, y + 25), 12, (255, 255, 255), 2)
        
        # Larger title
        cv2.putText(frame, item['title'], (x + 55, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['text_primary'], 2)
        
        # Larger message
        cv2.putText(frame, item['message'], (x + 55, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_secondary'], 1)
        
        # Larger action (if not info type)
        if item['type'] != 'info':
            cv2.putText(frame, f"→ {item['action']}", (x + 55, y + 58), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, item_color, 1)

    def draw_well_status_grid(self, frame, ui_colors, well_analysis, failed_wells):
        """Draw compact well status grid overlay"""
        if not well_analysis:
            return
            
        height, width = frame.shape[:2]
        
        # Position for much larger well grid (bottom left)
        grid_x = 25
        grid_y = height - 180
        cell_size = 25
        spacing = 5
        
        # Draw much larger grid background
        grid_width = 6 * (cell_size + spacing) - spacing + 30
        grid_height = 2 * (cell_size + spacing) - spacing + 60
        
        self.draw_glass_panel(frame, (grid_x, grid_y), (grid_x + grid_width, grid_y + grid_height), 
                             ui_colors['glass'], 0.6)
        
        # Larger grid title
        cv2.putText(frame, "WELL STATUS", (grid_x + 15, grid_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_primary'], 2)
        
        # Draw much larger wells
        for row_idx, row in enumerate(['A', 'B']):
            for col_idx in range(1, 7):  # Columns 1-6
                well_id = f"{row}{col_idx}"
                
                cell_x = grid_x + 15 + col_idx * (cell_size + spacing)
                cell_y = grid_y + 40 + row_idx * (cell_size + spacing)
                
                # Determine well color
                if well_id in failed_wells:
                    well_color = ui_colors['error']
                elif well_id in well_analysis:
                    well_color = ui_colors['success']
                else:
                    well_color = ui_colors['text_secondary']
                
                # Draw larger well
                cv2.rectangle(frame, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), well_color, -1)
                cv2.rectangle(frame, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (255, 255, 255), 2)
                
                # Larger well label
                cv2.putText(frame, well_id, (cell_x + 4, cell_y + 17), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def run(self):
        """Main monitoring loop"""
        if not self.setup():
            print("❌ Setup failed")
            return
            
        if not self.connect_camera():
            print("❌ Camera setup failed")
            return
        
        print("\n🧪 LIQUID HANDLER MONITOR READY!")
        print("🔬 Real-time procedure monitoring and error detection")
        print("📊 Advanced color verification and compliance checking")
        print("🤖 Robot arm blocking detection (no false alarms)")
        print("🧪 Individual well tracking for blue-red mixing")
        print("🤖 AI-powered VLM feedback with actionable insights")
        print("🎨 Frame cropping and filtering capabilities")
        print("⚡ PERFORMANCE OPTIMIZED:")
        print(f"   • Analysis interval: {self.analysis_interval}s (faster)")
        print("   • Smart frame skipping (3x speed boost)")
        print("   • Optimized image processing (reduced quality/size)")
        print("   • Streamlined AI prompts (faster responses)")
        
        # Simple goal setup at startup
        print("\n🎯 Quick Goal Setup:")
        setup_now = input("Set experiment goal now? (y/n): ").strip().lower()
        if setup_now in ['y', 'yes']:
            self.set_experiment_goal()
        else:
            print("💡 Press 'e' during monitoring to set goal")
        
        print("\n⚠️  Controls:")
        print("   Experiment Setup:")
        print("      'e' = Set experiment goal (quick selection)")
        print("      'i' = Show current setup")
        print("   Procedures:")
        print("      's' = Start sample preparation procedure")
        print("      'b' = Start blue-red mixing procedure")
        print("      'p' = Start PCR master-mix protocol (failing-protocol-5.py)")
        print("      'n' = Next step")
        print("   Frame Control:")
        print("      'c' = Open crop interface")
        print("      'r' = Reset crop region")
        print("      'u' = Undo last crop")
        print("      'g' = Toggle grid overlay (when crop UI open)")
        print("      'v' = Toggle preview mode (when crop UI open)")
        print("      'ESC' = Close crop interface")
        print("   Filters:")
        print("      '1' = No filter")
        print("      '2' = Grayscale")
        print("      '3' = HSV color space")
        print("      '4' = Gaussian blur")
        print("      '5' = Edge detection")
        print("      '6' = High contrast")
        print("      '7' = Brightness boost")
        print("      '8' = Gamma correction")
        print("      '+/-' = Adjust filter parameters")
        print("   System:")
        print("      'q' = Quit")
        print("=" * 70)
        
        self.running = True
        
        # Start AI analysis thread
        ai_thread = threading.Thread(target=self.analysis_worker, daemon=True)
        ai_thread.start()
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera disconnected")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Apply frame processing (crop and filters)
                processed_frame = self.process_frame(frame.copy())
                
                # Add overlay to processed frame
                self.add_overlay(processed_frame, frame_count)
                
                # Show live feed
                cv2.imshow('🧪 Liquid Handler Monitor', processed_frame)
                
                # Handle mouse events for crop selection
                cv2.setMouseCallback('🧪 Liquid Handler Monitor', self.mouse_callback)
                
                # Smart frame queuing for analysis
                if (current_time - self.last_analysis_time) >= self.analysis_interval:
                    should_analyze = self.should_analyze_frame()
                    
                    if should_analyze:
                        try:
                            # Clear old frames from queue
                            while not self.analysis_queue.empty():
                                try:
                                    self.analysis_queue.get_nowait()
                                except queue.Empty:
                                    break
                            
                            # Use original frame for AI analysis (before filters)
                            self.analysis_queue.put_nowait(frame.copy())
                            self.last_analysis_time = current_time
                            self.skip_analysis_count = 0
                            
                        except queue.Full:
                            pass
                    else:
                        self.skip_analysis_count += 1
                        # Still update timing to prevent backlog
                        self.last_analysis_time = current_time
                        
                        if self.skip_analysis_count % 5 == 0:  # Every 5 skipped frames
                            print(f"⏭️  Skipped {self.skip_analysis_count} analyses (optimizing speed)")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    # Set experiment goal
                    self.set_experiment_goal()
                elif key == ord('i'):
                    # Show current experimental setup
                    self.show_experiment_info()
                elif key == ord('s'):
                    # Start sample preparation procedure
                    self.start_procedure("sample_preparation")
                elif key == ord('b'):
                    # Start blue-red mixing procedure
                    self.start_procedure("blue_red_mixing")
                elif key == ord('p'):
                    # Start PCR master-mix protocol
                    self.start_procedure("pcr_master_mix")
                elif key == ord('n'):
                    # Next step
                    self.next_step()
                # Modern crop interface controls
                elif key == ord('c'):
                    # Toggle crop interface
                    self.toggle_crop_interface()
                elif key == ord('r'):
                    # Reset crop
                    self.reset_crop()
                elif key == ord('u'):
                    # Undo last crop
                    self.undo_crop()
                elif key == ord('g'):
                    # Toggle grid overlay
                    if self.crop_ui_visible:
                        self.crop_grid_overlay = not self.crop_grid_overlay
                        print(f"🔲 Grid overlay: {'ON' if self.crop_grid_overlay else 'OFF'}")
                elif key == ord('v'):
                    # Toggle preview mode
                    if self.crop_ui_visible:
                        self.crop_preview_enabled = not self.crop_preview_enabled
                        print(f"👁️ Preview: {'ON' if self.crop_preview_enabled else 'OFF'}")
                elif key == 27:  # ESC key
                    # Close crop interface
                    if self.crop_ui_visible:
                        self.close_crop_interface()
                elif key == ord('1'):
                    # No filter
                    self.current_filter = "none"
                    print("🎨 Filter: None")
                elif key == ord('2'):
                    # Grayscale
                    self.current_filter = "grayscale"
                    print("🎨 Filter: Grayscale")
                elif key == ord('3'):
                    # HSV
                    self.current_filter = "hsv"
                    print("🎨 Filter: HSV")
                elif key == ord('4'):
                    # Blur
                    self.current_filter = "blur"
                    print("🎨 Filter: Blur")
                elif key == ord('5'):
                    # Edge detection
                    self.current_filter = "edge"
                    print("🎨 Filter: Edge Detection")
                elif key == ord('6'):
                    # High contrast
                    self.current_filter = "contrast"
                    print("🎨 Filter: High Contrast")
                elif key == ord('7'):
                    # Brightness adjustment
                    self.current_filter = "brightness"
                    print("🎨 Filter: Brightness Boost")
                elif key == ord('8'):
                    # Gamma correction
                    self.current_filter = "gamma"
                    print("🎨 Filter: Gamma Correction")
                elif key == ord('=') or key == ord('+'):
                    # Increase filter parameter
                    self.adjust_filter_param(1)
                elif key == ord('-'):
                    # Decrease filter parameter
                    self.adjust_filter_param(-1)
        
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
        
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Save error report
            self.generate_error_report()
            print("✅ Liquid Handler Monitor stopped")

    def set_experiment_goal(self):
        """Set simple experimental goal from predefined list"""
        print("\n" + "="*50)
        print("🎯 SELECT EXPERIMENT GOAL")
        print("="*50)
        
        for key, goal in self.available_goals.items():
            print(f"  {key}. {goal}")
        print("  0. No specific goal (general monitoring)")
        
        choice = input("\nSelect goal (0-6): ").strip()
        
        if choice in self.available_goals:
            self.current_goal = self.available_goals[choice]
            print(f"\n✅ Goal set: {self.current_goal}")
        elif choice == "0":
            self.current_goal = ""
            print("\n✅ General monitoring mode")
        else:
            print("❌ Invalid choice - no goal set")
        
        print("="*50)
        return bool(self.current_goal)

    def show_experiment_info(self):
        """Display current experimental setup information"""
        print("\n" + "="*50)
        print("📊 CURRENT SETUP")
        print("="*50)
        
        if self.current_goal:
            print(f"🎯 Goal: {self.current_goal}")
        else:
            print("❌ No goal set - Press 'e' to select one")
        
        if self.current_procedure:
            print(f"🧬 Procedure: {self.current_procedure}")
            print(f"📋 Step: {self.current_step + 1}/{len(self.procedures[self.current_procedure])}")
        else:
            print("🔄 No active procedure")
        
        print(f"⚡ Speed: {self.analysis_interval}s intervals")
        errors = len(self.error_events) if hasattr(self, 'error_events') else 0
        print(f"🔍 Errors: {errors}")
        print("="*50)

    def should_analyze_frame(self):
        """Determine if current frame should be analyzed (smart skipping)"""
        # Always analyze if we haven't analyzed recently
        if not self.last_analysis_result:
            return True
        
        # Check if robot arm was blocking in last analysis
        arm_blocking = self.last_analysis_result.get("arm_blocking", False)
        
        if arm_blocking:
            self.consecutive_arm_blocking += 1
            # Skip analysis if arm has been blocking for multiple frames
            if self.consecutive_arm_blocking > 3:
                return False  # Skip until arm moves
        else:
            self.consecutive_arm_blocking = 0
        
        # Always analyze if we have a procedure running and plate is visible
        if self.current_procedure and not arm_blocking:
            return True
        
        # Skip some frames during normal operation to speed up
        if self.skip_analysis_count < 2:  # Analyze every 3rd eligible frame
            return True
        
        return False

    def generate_error_report(self):
        """Generate comprehensive error report"""
        if not self.error_events:
            print("✅ No errors detected during monitoring session")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"error_report_{timestamp}.json"
        
        report_data = {
            "monitoring_session": {
                "timestamp": timestamp,
                "total_errors": len(self.error_events),
                "procedures_monitored": list(self.procedures.keys()) if self.current_procedure else []
            },
            "error_events": [asdict(event) for event in self.error_events],
            "error_summary": self._generate_error_summary()
        }
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📊 Error report saved: {report_filename}")
        print(f"📈 Total errors: {len(self.error_events)}")

    def _generate_error_summary(self) -> dict:
        """Generate error summary statistics"""
        summary = {
            "by_severity": {},
            "by_type": {},
            "by_procedure_step": {}
        }
        
        for event in self.error_events:
            # By severity
            severity = event.severity
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            
            # By type
            error_type = event.error_type
            summary["by_type"][error_type] = summary["by_type"].get(error_type, 0) + 1
            
            # By procedure step
            step = event.procedure_step
            summary["by_procedure_step"][step] = summary["by_procedure_step"].get(step, 0) + 1
        
        return summary

    def process_frame(self, frame):
        """Apply cropping and filtering to frame"""
        # Apply cropping first
        if self.crop_enabled and self.crop_region:
            x, y, w, h = self.crop_region
            # Ensure crop region is within frame bounds
            height, width = frame.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Crop the frame
            cropped = frame[y:y+h, x:x+w]
            
            # Resize back to original dimensions for consistent UI
            frame = cv2.resize(cropped, (width, height))
        
        # Apply current filter
        filtered_frame = self.apply_filter(frame)
        
        # Draw enhanced crop overlays
        self.draw_crop_overlays(filtered_frame)
        
        # Draw current crop region indicator
        if self.crop_enabled and self.crop_region:
            x, y, w, h = self.crop_region
            cv2.rectangle(filtered_frame, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.putText(filtered_frame, f"CROPPED: {w}x{h} at ({x},{y})", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw current filter indicator
        if self.current_filter != "none":
            filter_text = f"FILTER: {self.current_filter.upper()}"
            cv2.rectangle(filtered_frame, (10, 60), (250, 90), (0, 0, 0), -1)
            cv2.putText(filtered_frame, filter_text, (15, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
        
        return filtered_frame

    def apply_filter(self, frame):
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

    def adjust_filter_param(self, direction):
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

    def mouse_callback(self, event, x, y, flags, param):
        """Modern UI-based mouse handling for crop interface"""
        self.mouse_pos = (x, y)
        
        if self.crop_ui_visible:
            self.handle_crop_ui_interaction(event, x, y)
        elif self.crop_enabled and self.crop_region:
            self.handle_crop_adjustment(event, x, y)

    def handle_crop_ui_interaction(self, event, x, y):
        """Handle mouse interactions with the crop UI interface"""
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_element = self.get_ui_element_at_position(x, y)
            if clicked_element:
                self.ui_active_element = clicked_element
                self.handle_ui_element_click(clicked_element)
        elif event == cv2.EVENT_LBUTTONUP:
            self.ui_active_element = None
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update hover state
            self.ui_hover_element = self.get_ui_element_at_position(x, y)

    def handle_crop_adjustment(self, event, x, y):
        """Handle direct crop region adjustment when UI is not visible"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a handle
            self.selected_handle = self.get_handle_at_position(x, y)
            if self.selected_handle is None:
                # Click outside handles - move entire crop region
                if self.crop_region and self.point_in_crop_region(x, y):
                    self.selected_handle = "move"
                    self.ui_drag_start = (x, y)
                    print("🔄 Moving crop region...")
        elif event == cv2.EVENT_MOUSEMOVE and self.selected_handle:
            self.adjust_crop_region_direct(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.selected_handle:
                print("✅ Crop adjustment complete")
                self.selected_handle = None
                self.ui_drag_start = None

    def adjust_crop_region_direct(self, x, y):
        """Adjust crop region directly when UI is not visible"""
        if not self.crop_region or not self.selected_handle:
            return
            
        cx, cy, cw, ch = self.crop_region
        
        if self.selected_handle == "move":
            # Move entire region
            if self.ui_drag_start:
                dx = x - self.ui_drag_start[0]
                dy = y - self.ui_drag_start[1]
                self.crop_region = (cx + dx, cy + dy, cw, ch)
                self.ui_drag_start = (x, y)
        else:
            # Handle resize operations
            if self.selected_handle == "nw":
                new_w = max(30, cw + (cx - x))
                new_h = max(30, ch + (cy - y))
                self.crop_region = (x, y, new_w, new_h)
            elif self.selected_handle == "ne":
                new_w = max(30, x - cx)
                new_h = max(30, ch + (cy - y))
                self.crop_region = (cx, y, new_w, new_h)
            elif self.selected_handle == "sw":
                new_w = max(30, cw + (cx - x))
                new_h = max(30, y - cy)
                self.crop_region = (x, cy, new_w, new_h)
            elif self.selected_handle == "se":
                new_w = max(30, x - cx)
                new_h = max(30, y - cy)
                self.crop_region = (cx, cy, new_w, new_h)

    def reset_crop(self):
        """Reset crop region completely"""
        if self.crop_enabled:
            # Store current crop for undo
            if self.crop_region:
                self.crop_history.append(self.crop_region)
                if len(self.crop_history) > 5:
                    self.crop_history.pop(0)
        
        self.crop_enabled = False
        self.crop_region = None
        self.crop_mode = "off"
        self.crop_start = None
        self.crop_temp = None
        self.selected_handle = None
        self.crop_handles = []
        print("🔄 Crop region reset")

    def undo_crop(self):
        """Restore previous crop region"""
        if self.crop_history:
            self.crop_region = self.crop_history.pop()
            self.crop_enabled = True
            self.crop_mode = "off"
            x, y, w, h = self.crop_region
            print(f"↩️  Crop restored: {w}x{h} at ({x},{y})")
        else:
            print("❌ No previous crop to restore")

    def apply_crop_preset(self, preset_name):
        """Apply a predefined crop preset"""
        if preset_name not in self.crop_presets:
            print(f"❌ Unknown preset: {preset_name}")
            return
        
        # Store current crop for undo
        if self.crop_enabled and self.crop_region:
            self.crop_history.append(self.crop_region)
            if len(self.crop_history) > 5:
                self.crop_history.pop(0)
        
        # Get frame dimensions (approximate)
        frame_w, frame_h = 640, 480
        
        # Convert relative coordinates to absolute
        rel_x, rel_y, rel_w, rel_h = self.crop_presets[preset_name]
        x = int(rel_x * frame_w)
        y = int(rel_y * frame_h)
        w = int(rel_w * frame_w)
        h = int(rel_h * frame_h)
        
        self.crop_region = (x, y, w, h)
        self.crop_enabled = True
        self.crop_mode = "off"
        
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

    def update_crop_handles(self):
        """Update resize handle positions"""
        if not self.crop_region:
            return
            
        x, y, w, h = self.crop_region
        handle_size = 8
        
        self.crop_handles = [
            ("nw", x - handle_size//2, y - handle_size//2),           # Top-left
            ("ne", x + w - handle_size//2, y - handle_size//2),       # Top-right
            ("sw", x - handle_size//2, y + h - handle_size//2),       # Bottom-left
            ("se", x + w - handle_size//2, y + h - handle_size//2),   # Bottom-right
            ("n", x + w//2 - handle_size//2, y - handle_size//2),     # Top-center
            ("s", x + w//2 - handle_size//2, y + h - handle_size//2), # Bottom-center
            ("w", x - handle_size//2, y + h//2 - handle_size//2),     # Left-center
            ("e", x + w - handle_size//2, y + h//2 - handle_size//2)  # Right-center
        ]

    def get_handle_at_position(self, x, y):
        """Get the handle at the given position"""
        if not self.crop_region:
            return None
            
        cx, cy, cw, ch = self.crop_region
        handle_size = 8
        
        # Check corner handles
        handles = {
            "nw": (cx, cy),
            "ne": (cx + cw, cy),
            "sw": (cx, cy + ch),
            "se": (cx + cw, cy + ch)
        }
        
        for handle_id, (hx, hy) in handles.items():
            if (hx - handle_size <= x <= hx + handle_size and 
                hy - handle_size <= y <= hy + handle_size):
                return handle_id
        
        return None

    def point_in_crop_region(self, x, y):
        """Check if point is inside the crop region"""
        if not self.crop_region:
            return False
            
        cx, cy, cw, ch = self.crop_region
        return cx <= x <= cx + cw and cy <= y <= cy + ch

    def adjust_crop_region(self, x, y):
        """Adjust crop region based on selected handle"""
        if not self.crop_region or not self.selected_handle:
            return
            
        cx, cy, cw, ch = self.crop_region
        
        if self.selected_handle == "move":
            # Move entire region
            if self.crop_start:
                dx = x - self.crop_start[0]
                dy = y - self.crop_start[1]
                self.crop_region = (cx + dx, cy + dy, cw, ch)
                self.crop_start = (x, y)
        else:
            # Resize based on handle
            if self.selected_handle == "nw":
                new_w = max(30, cw + (cx - x))
                new_h = max(30, ch + (cy - y))
                self.crop_region = (x, y, new_w, new_h)
            elif self.selected_handle == "ne":
                new_w = max(30, x - cx)
                new_h = max(30, ch + (cy - y))
                self.crop_region = (cx, y, new_w, new_h)
            elif self.selected_handle == "sw":
                new_w = max(30, cw + (cx - x))
                new_h = max(30, y - cy)
                self.crop_region = (x, cy, new_w, new_h)
            elif self.selected_handle == "se":
                new_w = max(30, x - cx)
                new_h = max(30, y - cy)
                self.crop_region = (cx, cy, new_w, new_h)
            elif self.selected_handle == "n":
                new_h = max(30, ch + (cy - y))
                self.crop_region = (cx, y, cw, new_h)
            elif self.selected_handle == "s":
                new_h = max(30, y - cy)
                self.crop_region = (cx, cy, cw, new_h)
            elif self.selected_handle == "w":
                new_w = max(30, cw + (cx - x))
                self.crop_region = (x, cy, new_w, ch)
            elif self.selected_handle == "e":
                new_w = max(30, x - cx)
                self.crop_region = (cx, cy, new_w, ch)
        
        # Update handles for real-time feedback
        self.update_crop_handles()

    def draw_crop_overlays(self, frame):
        """Draw enhanced crop overlays with better visual feedback"""
        height, width = frame.shape[:2]
        
        if self.crop_mode == "selecting" and self.crop_start and self.crop_temp:
            self.draw_selection_overlay(frame)
        elif self.crop_mode == "adjusting" and self.crop_region:
            self.draw_adjustment_overlay(frame)
        elif self.crop_enabled and self.crop_region:
            self.draw_active_crop_overlay(frame)

    def draw_selection_overlay(self, frame):
        """Draw overlay during crop selection"""
        if not self.crop_start or not self.crop_temp:
            return
            
        x1, y1 = self.crop_start
        x2, y2 = self.crop_temp
        
        # Draw selection rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Draw corner markers
        corner_size = 10
        cv2.line(frame, (x1, y1), (x1 + corner_size, y1), (0, 255, 255), 3)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_size), (0, 255, 255), 3)
        cv2.line(frame, (x2, y2), (x2 - corner_size, y2), (0, 255, 255), 3)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_size), (0, 255, 255), 3)
        
        # Draw dimensions
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w > 50 and h > 30:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (center_x - 40, center_y - 10), (center_x + 40, center_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, f"{w}x{h}", (center_x - 35, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw instruction
        cv2.putText(frame, "SELECTING CROP REGION", (x1, y1 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def draw_adjustment_overlay(self, frame):
        """Draw overlay during crop adjustment"""
        if not self.crop_region:
            return
            
        x, y, w, h = self.crop_region
        
        # Draw crop region with thicker border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
        
        # Draw resize handles
        handle_size = 8
        for handle_type, hx, hy in self.crop_handles:
            color = (0, 255, 0) if handle_type == self.selected_handle else (255, 255, 255)
            cv2.rectangle(frame, (hx, hy), (hx + handle_size, hy + handle_size), color, -1)
            cv2.rectangle(frame, (hx, hy), (hx + handle_size, hy + handle_size), (0, 0, 0), 1)
        
        # Draw instruction
        cv2.putText(frame, "ADJUSTING CROP - Drag handles to resize", (x, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def draw_active_crop_overlay(self, frame):
        """Draw overlay for active crop region"""
        if not self.crop_region:
            return
            
        x, y, w, h = self.crop_region
        
        # Draw subtle crop border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 255), 1)
        
        # Draw corner indicators
        corner_size = 6
        cv2.line(frame, (x, y), (x + corner_size, y), (100, 200, 255), 2)
        cv2.line(frame, (x, y), (x, y + corner_size), (100, 200, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w - corner_size, y + h), (100, 200, 255), 2)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_size), (100, 200, 255), 2)

    def toggle_crop_interface(self):
        """Toggle the modern crop interface"""
        if self.crop_ui_visible:
            self.close_crop_interface()
        else:
            self.open_crop_interface()

    def open_crop_interface(self):
        """Open the modern crop interface panel"""
        self.crop_ui_visible = True
        self.crop_mode = "ui_active"
        self.initialize_crop_ui_elements()
        print("🎛️ CROP INTERFACE OPENED")
        print("   • Use mouse to interact with controls")
        print("   • ESC to close interface")
        print("   • All changes apply in real-time")

    def close_crop_interface(self):
        """Close the crop interface"""
        self.crop_ui_visible = False
        self.crop_mode = "off"
        self.ui_hover_element = None
        self.ui_active_element = None
        print("✅ Crop interface closed")

    def initialize_crop_ui_elements(self):
        """Initialize all UI elements for the crop interface"""
        ui_x, ui_y = self.crop_ui_position
        
        # Create UI elements layout
        self.crop_buttons = [
            # Preset buttons (2 columns)
            {'id': 'preset_full', 'type': 'button', 'rect': (ui_x + 10, ui_y + 40, 80, 25), 'text': 'Full Frame', 'preset': 'full'},
            {'id': 'preset_center', 'type': 'button', 'rect': (ui_x + 100, ui_y + 40, 80, 25), 'text': 'Center 50%', 'preset': 'center'},
            {'id': 'preset_wellplate', 'type': 'button', 'rect': (ui_x + 10, ui_y + 70, 80, 25), 'text': 'Well Plate', 'preset': 'wellplate'},
            {'id': 'preset_top', 'type': 'button', 'rect': (ui_x + 100, ui_y + 70, 80, 25), 'text': 'Top Half', 'preset': 'top_half'},
            {'id': 'preset_bottom', 'type': 'button', 'rect': (ui_x + 10, ui_y + 100, 80, 25), 'text': 'Bottom Half', 'preset': 'bottom_half'},
            {'id': 'preset_left', 'type': 'button', 'rect': (ui_x + 100, ui_y + 100, 80, 25), 'text': 'Left Half', 'preset': 'left_half'},
            
            # Action buttons
            {'id': 'reset', 'type': 'button', 'rect': (ui_x + 10, ui_y + 140, 60, 25), 'text': 'Reset', 'action': 'reset'},
            {'id': 'undo', 'type': 'button', 'rect': (ui_x + 80, ui_y + 140, 60, 25), 'text': 'Undo', 'action': 'undo'},
            {'id': 'apply', 'type': 'button', 'rect': (ui_x + 150, ui_y + 140, 60, 25), 'text': 'Apply', 'action': 'apply'},
            
            # Toggle buttons
            {'id': 'toggle_grid', 'type': 'toggle', 'rect': (ui_x + 10, ui_y + 180, 90, 20), 'text': 'Grid Overlay', 'state': self.crop_grid_overlay},
            {'id': 'toggle_preview', 'type': 'toggle', 'rect': (ui_x + 110, ui_y + 180, 90, 20), 'text': 'Live Preview', 'state': self.crop_preview_enabled},
        ]
        
        # Initialize sliders for precise control
        self.crop_sliders = {
            'x': {'rect': (ui_x + 10, ui_y + 220, 180, 15), 'value': 0.0, 'label': 'X Position'},
            'y': {'rect': (ui_x + 10, ui_y + 245, 180, 15), 'value': 0.0, 'label': 'Y Position'},
            'width': {'rect': (ui_x + 10, ui_y + 270, 180, 15), 'value': 1.0, 'label': 'Width'},
            'height': {'rect': (ui_x + 10, ui_y + 295, 180, 15), 'value': 1.0, 'label': 'Height'},
        }
        
        # Update slider values if crop exists
        if self.crop_enabled and self.crop_region:
            frame_w, frame_h = 640, 480  # Approximate frame size
            x, y, w, h = self.crop_region
            self.crop_sliders['x']['value'] = x / frame_w
            self.crop_sliders['y']['value'] = y / frame_h
            self.crop_sliders['width']['value'] = w / frame_w
            self.crop_sliders['height']['value'] = h / frame_h

    def get_ui_element_at_position(self, x, y):
        """Get the UI element at the given mouse position"""
        # Check buttons
        for button in self.crop_buttons:
            bx, by, bw, bh = button['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button
        
        # Check sliders
        for slider_id, slider in self.crop_sliders.items():
            sx, sy, sw, sh = slider['rect']
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                return {'type': 'slider', 'id': slider_id, **slider}
        
        return None

    def handle_ui_element_click(self, element):
        """Handle clicks on UI elements"""
        if element['type'] == 'button':
            if 'preset' in element:
                self.apply_crop_preset(element['preset'])
            elif 'action' in element:
                if element['action'] == 'reset':
                    self.reset_crop()
                elif element['action'] == 'undo':
                    self.undo_crop()
                elif element['action'] == 'apply':
                    self.apply_current_crop()
        elif element['type'] == 'toggle':
            if element['id'] == 'toggle_grid':
                self.crop_grid_overlay = not self.crop_grid_overlay
                element['state'] = self.crop_grid_overlay
            elif element['id'] == 'toggle_preview':
                self.crop_preview_enabled = not self.crop_preview_enabled
                element['state'] = self.crop_preview_enabled
        elif element['type'] == 'slider':
            self.handle_slider_interaction(element)

    def handle_slider_interaction(self, slider):
        """Handle slider interactions for precise crop control"""
        # Calculate new value based on mouse position
        sx, sy, sw, sh = slider['rect']
        mouse_x, mouse_y = self.mouse_pos
        
        # Calculate relative position on slider
        relative_x = max(0, min(1, (mouse_x - sx) / sw))
        
        # Update slider value
        self.crop_sliders[slider['id']]['value'] = relative_x
        
        # Apply crop changes in real-time
        self.update_crop_from_sliders()

    def update_crop_from_sliders(self):
        """Update crop region based on slider values"""
        frame_w, frame_h = 640, 480  # Approximate frame size
        
        x_val = self.crop_sliders['x']['value']
        y_val = self.crop_sliders['y']['value']
        w_val = self.crop_sliders['width']['value']
        h_val = self.crop_sliders['height']['value']
        
        # Convert to pixel coordinates
        x = int(x_val * frame_w)
        y = int(y_val * frame_h)
        w = int(w_val * frame_w)
        h = int(h_val * frame_h)
        
        # Ensure minimum size
        w = max(30, w)
        h = max(30, h)
        
        # Ensure within bounds
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        
        # Update crop region
        self.crop_region = (x, y, w, h)
        self.crop_enabled = True

    def apply_current_crop(self):
        """Apply the current crop settings and close interface"""
        if self.crop_enabled and self.crop_region:
            x, y, w, h = self.crop_region
            print(f"✅ Crop applied: {w}x{h} at ({x},{y})")
            self.close_crop_interface()
        else:
            print("❌ No crop region to apply")





    def draw_crop_ui_interface(self, frame):
        """Draw the modern crop interface panel"""
        ui_x, ui_y = self.crop_ui_position
        panel_width = 220
        panel_height = 320
        
        # Draw main panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (ui_x, ui_y), (ui_x + panel_width, ui_y + panel_height), (25, 35, 45), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Panel border and header
        cv2.rectangle(frame, (ui_x, ui_y), (ui_x + panel_width, ui_y + panel_height), (100, 200, 255), 2)
        cv2.rectangle(frame, (ui_x, ui_y), (ui_x + panel_width, ui_y + 30), (100, 200, 255), -1)
        cv2.putText(frame, "CROP CONTROLS", (ui_x + 10, ui_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw all buttons
        for button in self.crop_buttons:
            self.draw_ui_button(frame, button)
        
        # Draw all sliders
        for slider_id, slider in self.crop_sliders.items():
            self.draw_ui_slider(frame, slider_id, slider)

    def draw_ui_button(self, frame, button):
        """Draw a UI button with hover and active states"""
        bx, by, bw, bh = button['rect']
        
        # Determine button state
        is_hovered = self.ui_hover_element == button
        is_active = self.ui_active_element == button
        
        # Choose colors based on state
        if button['type'] == 'toggle':
            if button['state']:
                bg_color = (64, 255, 128) if not is_active else (48, 192, 96)
                text_color = (0, 0, 0)
            else:
                bg_color = (60, 70, 80) if not is_hovered else (80, 90, 100)
                text_color = (200, 200, 200)
        else:
            if is_active:
                bg_color = (80, 160, 255)
            elif is_hovered:
                bg_color = (120, 180, 255)
            else:
                bg_color = (60, 70, 80)
            text_color = (255, 255, 255)
        
        # Draw button
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bg_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        
        # Draw button text
        text_size = cv2.getTextSize(button['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = bx + (bw - text_size[0]) // 2
        text_y = by + (bh + text_size[1]) // 2
        cv2.putText(frame, button['text'], (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

    def draw_ui_slider(self, frame, slider_id, slider):
        """Draw a UI slider with current value"""
        sx, sy, sw, sh = slider['rect']
        
        # Draw slider track
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (40, 50, 60), -1)
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (100, 100, 100), 1)
        
        # Draw slider handle
        handle_x = sx + int(slider['value'] * sw)
        handle_w = 8
        cv2.rectangle(frame, (handle_x - handle_w//2, sy - 2), 
                     (handle_x + handle_w//2, sy + sh + 2), (100, 200, 255), -1)
        cv2.rectangle(frame, (handle_x - handle_w//2, sy - 2), 
                     (handle_x + handle_w//2, sy + sh + 2), (255, 255, 255), 1)
        
        # Draw label and value
        label_y = sy - 5
        cv2.putText(frame, slider['label'], (sx, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        value_text = f"{slider['value']:.2f}"
        value_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
        cv2.putText(frame, value_text, (sx + sw - value_size[0], label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 255), 1)

    def draw_crop_preview(self, frame):
        """Draw crop preview with modern styling"""
        if not self.crop_region or not self.crop_preview_enabled:
            return
            
        x, y, w, h = self.crop_region
        
        # Draw crop region with gradient border
        cv2.rectangle(frame, (x-2, y-2), (x + w + 2, y + h + 2), (100, 200, 255), 2)
        cv2.rectangle(frame, (x-1, y-1), (x + w + 1, y + h + 1), (150, 220, 255), 1)
        
        # Draw corner handles
        handle_size = 8
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in corners:
            cv2.rectangle(frame, (cx - handle_size//2, cy - handle_size//2), 
                         (cx + handle_size//2, cy + handle_size//2), (100, 200, 255), -1)
            cv2.rectangle(frame, (cx - handle_size//2, cy - handle_size//2), 
                         (cx + handle_size//2, cy + handle_size//2), (255, 255, 255), 1)
        
        # Draw crop info
        info_text = f"{w}x{h} ({x},{y})"
        cv2.rectangle(frame, (x, y - 25), (x + 150, y - 5), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (x + 5, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

    def draw_grid_overlay(self, frame):
        """Draw grid overlay for alignment"""
        height, width = frame.shape[:2]
        
        # Draw rule of thirds grid
        grid_color = (80, 80, 80)
        
        # Vertical lines
        for i in range(1, 3):
            x = width * i // 3
            cv2.line(frame, (x, 0), (x, height), grid_color, 1)
        
        # Horizontal lines
        for i in range(1, 3):
            y = height * i // 3
            cv2.line(frame, (0, y), (width, y), grid_color, 1)
        
        # Center crosshair
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (100, 100, 100), 1)

def main():
    print("🧪 Starting Liquid Handler Robot Monitor")
    monitor = LiquidHandlerMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
