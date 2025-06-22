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
        self.analysis_queue = queue.Queue(maxsize=2)
        self.last_analysis_time = 0
        self.analysis_interval = 3.0  # Analyze every 3 seconds
        
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            import io
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=90)
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

            prompt = f"""LIQUID HANDLER MONITORING SYSTEM
PRIMARY GOAL: Verify correct liquid handling procedure execution and detect errors immediately.

{procedure_context}

CORE MONITORING OBJECTIVES:
1. VERIFY PROCEDURE COMPLIANCE: Are the expected actions being performed correctly?
2. DETECT LIQUID HANDLING ERRORS: Wrong colors, failed mixing, contamination
3. IDENTIFY FAILED OPERATIONS: Which specific wells/actions have problems?
4. ASSESS SAFETY COMPLIANCE: Any spills, contamination risks, or unsafe conditions?

ANALYSIS FOCUS:
- PROCEDURE EXECUTION: Is the current step being performed as expected?
- LIQUID COLORS: Do well contents match expected colors for this procedure step?
- MIXING QUALITY: Are liquids properly mixed (purple for blue+red mixing)?
- ERROR DETECTION: Identify specific wells or operations that have failed
- SAFETY MONITORING: Detect spills, contamination, or other safety issues

OPERATIONAL CONTEXT:
- Robot arm movement is normal operation (not an error)
- Focus analysis on visible well plate contents when robot is not blocking view
- For blue-red mixing: Expect blue → red addition → purple mixed result
- For PCR prep: Expect blue water → red master-mix → proper mixing

RESPONSE FORMAT:
STATUS: [NORMAL/WARNING/ERROR/CRITICAL]
ARM_BLOCKING: [YES/NO - robot arm currently blocking view]
PLATE_VISIBLE: [YES/NO - can analyze well plate contents]
WELL_ANALYSIS: [A1:color, A2:color, A3:color, A4:color, A5:color, A6:color, B1:color, B2:color, B3:color, B4:color, B5:color, B6:color]
FAILED_WELLS: [specific wells with incorrect results or "NONE"]
COMPLIANCE: [YES/NO - is procedure being followed correctly]
ERRORS: [specific liquid handling errors detected or "NONE"]
SAFETY: [safety issues detected or "OK"]
CONFIDENCE: [0.0-1.0]
DESCRIPTION: [brief description focused on procedure execution status]"""

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
        """Draw computer vision overlays on the frame"""
        if not self.show_overlays or not self.cv_enabled:
            return
            
        # Only draw tracked containers
        if self.show_object_labels:
            self.draw_containers_overlay(frame)

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
                    
                    print("🧠 Analyzing liquid handler operations...")
                    analysis_result = self.analyze_frame_with_ai(frame)
                    
                    # Store result
                    self.last_analysis_result = analysis_result
                    self.last_description = analysis_result.get("description", "Analysis failed")
                    
                    # Check compliance
                    self.check_procedure_compliance(analysis_result, frame)
                    self.check_color_compliance(analysis_result, frame)
                    self.check_well_compliance(analysis_result, frame)
                    self.check_safety_compliance(analysis_result, frame)
                    
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
        
        # Draw computer vision overlays first (behind UI)
        self.draw_cv_overlays(frame)
        
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
        
        # Enhanced top bar
        self.draw_glass_panel(frame, (0, 0), (width, 60), ui_colors['background'], 0.8)
        
        # Larger arc reactor style status indicator
        cv2.circle(frame, (35, 30), 15, status_color, -1)
        cv2.circle(frame, (35, 30), 15, ui_colors['accent_glow'], 2)
        cv2.circle(frame, (35, 30), 10, (255, 255, 255), 2)
        cv2.circle(frame, (35, 30), 6, status_color, 1)
        
        # Enhanced title with better typography
        cv2.putText(frame, "LIQUID HANDLER MONITOR", (65, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_colors['text_primary'], 2)
        cv2.putText(frame, f"STATUS: {status}", (65, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Enhanced time display with better positioning
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, "TIME", (width - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_secondary'], 1)
        cv2.putText(frame, timestamp, (width - 120, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_colors['accent'], 2)

    def draw_compact_status(self, frame, ui_colors, status_color):
        """Draw enhanced status indicators"""
        height, width = frame.shape[:2]
        
        # Enhanced error panel (bigger and more prominent)
        self.draw_enhanced_error_panel(frame, ui_colors, width - 200, 75)
        
        # Confidence indicator
        self.draw_confidence_indicator(frame, ui_colors, width - 200, 140)
        
        # Procedure indicator on left
        if self.current_procedure:
            self.draw_procedure_indicator(frame, ui_colors, 15, 75)

    def draw_enhanced_error_panel(self, frame, ui_colors, x, y):
        """Draw prominent error display panel"""
        error_count = len(self.error_events)
        error_color = ui_colors['error'] if error_count > 0 else ui_colors['success']
        
        panel_width = 180
        panel_height = 50
        
        # Enhanced glass panel with better visibility
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.6)
        
        # Larger status indicator
        cv2.circle(frame, (x + 20, y + 25), 8, error_color, -1)
        cv2.circle(frame, (x + 20, y + 25), 8, ui_colors['accent_glow'], 2)
        cv2.circle(frame, (x + 20, y + 25), 5, (255, 255, 255), 1)
        
        # Enhanced error display
        cv2.putText(frame, "SYSTEM STATUS", (x + 35, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_primary'], 1)
        
        if error_count > 0:
            cv2.putText(frame, f"{error_count} ERRORS DETECTED", (x + 35, y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 2)
            
            # Show latest error type if available
            if self.error_events:
                latest_error = self.error_events[-1].error_type.replace('_', ' ')[:20]
                cv2.putText(frame, f"Latest: {latest_error}", (x + 5, y + 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, ui_colors['text_secondary'], 1)
        else:
            cv2.putText(frame, "ALL SYSTEMS OK", (x + 35, y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['success'], 2)

    def draw_confidence_indicator(self, frame, ui_colors, x, y):
        """Draw enhanced confidence indicator"""
        confidence = self.last_analysis_result.get("confidence", 0.0)
        conf_color = ui_colors['success'] if confidence > 0.8 else ui_colors['warning'] if confidence > 0.5 else ui_colors['error']
        
        panel_width = 180
        panel_height = 35
        
        # Enhanced glass panel
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.5)
        
        # Label
        cv2.putText(frame, "AI CONFIDENCE", (x + 10, y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_primary'], 1)
        
        # Enhanced confidence bar
        bar_width = 120
        bar_height = 8
        bar_x = x + 10
        bar_y = y + 20
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['background'], -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['accent'], 1)
        
        # Fill bar with gradient effect
        fill_width = int(bar_width * confidence)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), conf_color, -1)
        
        # Percentage display
        cv2.putText(frame, f"{confidence:.0%}", (x + 140, y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)

    def draw_procedure_indicator(self, frame, ui_colors, x, y):
        """Draw enhanced procedure status panel"""
        if not self.current_procedure:
            return
            
        proc_name = self.current_procedure.replace('_', ' ').upper()
        
        # Larger panel for better visibility
        panel_width = 300
        panel_height = 60
        self.draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), ui_colors['glass'], 0.6)
        
        # Enhanced procedure indicator
        cv2.circle(frame, (x + 20, y + 20), 8, ui_colors['accent'], -1)
        cv2.circle(frame, (x + 20, y + 20), 8, ui_colors['accent_glow'], 2)
        cv2.circle(frame, (x + 20, y + 20), 5, (255, 255, 255), 1)
        
        # Procedure name
        cv2.putText(frame, "ACTIVE PROCEDURE", (x + 35, y + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_primary'], 1)
        
        # Enhanced procedure name and step
        step_text = f"{proc_name}"
        cv2.putText(frame, step_text, (x + 35, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['accent'], 2)
        
        # Step progress
        step_progress = f"STEP {self.current_step + 1} OF {len(self.procedures[self.current_procedure])}"
        cv2.putText(frame, step_progress, (x + 35, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_secondary'], 1)
        
        # Progress bar for steps
        bar_width = 200
        bar_height = 4
        bar_x = x + 35
        bar_y = y + 55
        progress = (self.current_step + 1) / len(self.procedures[self.current_procedure])
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), ui_colors['background'], -1)
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
        
        # Enhanced robot arm status (only if active)
        arm_blocking = self.last_analysis_result.get("arm_blocking", False)
        if arm_blocking:
            self.draw_glass_panel(frame, (15, height - 65), (280, height - 15), ui_colors['warning'], 0.7)
            cv2.circle(frame, (35, height - 40), 8, ui_colors['warning'], -1)
            cv2.circle(frame, (35, height - 40), 8, (255, 255, 255), 2)
            cv2.putText(frame, "ROBOT ARM ACTIVE", (55, height - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_primary'], 2)
            cv2.putText(frame, "Analysis paused during operation", (55, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, ui_colors['text_secondary'], 1)
        
        # Enhanced container tracking indicator
        if self.cv_enabled:
            container_count = len([obj for obj in self.tracked_objects.values() if obj.object_type == 'container'])
            if container_count > 0:
                self.draw_glass_panel(frame, (15, 150), (220, 190), ui_colors['glass'], 0.5)
                cv2.circle(frame, (30, 170), 6, ui_colors['accent'], -1)
                cv2.circle(frame, (30, 170), 6, ui_colors['accent_glow'], 1)
                cv2.putText(frame, "CONTAINER TRACKING", (45, 167), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_primary'], 1)
                cv2.putText(frame, f"{container_count} LABWARE DETECTED", (45, 185), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['accent'], 1)
        
        # Enhanced wells status display (for blue-red mixing)
        if self.current_procedure == "blue_red_mixing":
            failed_wells_count = len(self.failed_wells)
            wells_color = ui_colors['error'] if failed_wells_count > 0 else ui_colors['success']
            
            panel_width = 250
            self.draw_glass_panel(frame, (width - panel_width - 15, height - 65), 
                                (width - 15, height - 15), wells_color, 0.7)
            
            cv2.circle(frame, (width - panel_width + 20, height - 40), 8, wells_color, -1)
            cv2.circle(frame, (width - panel_width + 20, height - 40), 8, (255, 255, 255), 2)
            
            if failed_wells_count > 0:
                cv2.putText(frame, f"{failed_wells_count} WELLS FAILED", (width - panel_width + 40, height - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_primary'], 2)
                # Show failed well names
                if len(self.failed_wells) <= 6:
                    failed_text = ", ".join(self.failed_wells)
                    cv2.putText(frame, failed_text, (width - panel_width + 40, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, ui_colors['text_secondary'], 1)
            else:
                cv2.putText(frame, "ALL WELLS OK", (width - panel_width + 40, height - 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, ui_colors['text_primary'], 2)



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
        print("📦 Container tracking (focuses on ~3 main containers)")
        print("\n⚠️  Controls:")
        print("   Procedures:")
        print("      's' = Start sample preparation procedure")
        print("      'b' = Start blue-red mixing procedure")
        print("      'p' = Start PCR master-mix protocol (failing-protocol-5.py)")
        print("      'n' = Next step")
        print("   Container Tracking:")
        print("      'c' = Toggle container detection on/off")
        print("      'o' = Toggle container tracking overlays")
        print("      'l' = Toggle container labels")
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
                
                # Add overlay
                self.add_overlay(frame, frame_count)
                
                # Show live feed
                cv2.imshow('🧪 Liquid Handler Monitor', frame)
                
                # Queue frame for analysis
                if (current_time - self.last_analysis_time) >= self.analysis_interval:
                    try:
                        while not self.analysis_queue.empty():
                            try:
                                self.analysis_queue.get_nowait()
                            except queue.Empty:
                                break
                        
                        self.analysis_queue.put_nowait(frame.copy())
                        self.last_analysis_time = current_time
                        
                    except queue.Full:
                        pass
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
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
                elif key == ord('o'):
                    # Toggle container overlays
                    self.show_overlays = not self.show_overlays
                    print(f"📦 Container Tracking: {'ON' if self.show_overlays else 'OFF'}")
                elif key == ord('l'):
                    # Toggle container labels
                    self.show_object_labels = not self.show_object_labels
                    print(f"🏷️  Container Labels: {'ON' if self.show_object_labels else 'OFF'}")
                elif key == ord('c'):
                    # Toggle computer vision
                    self.cv_enabled = not self.cv_enabled
                    print(f"👁️  Computer Vision: {'ON' if self.cv_enabled else 'OFF'}")
        
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

def main():
    print("🧪 Starting Liquid Handler Robot Monitor")
    monitor = LiquidHandlerMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
