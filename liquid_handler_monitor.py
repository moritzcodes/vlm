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

    def _load_procedures(self) -> Dict[str, List[ProcedureStep]]:
        """Load standard operating procedures"""
        return {
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

            prompt = f"""You are monitoring a liquid handler robot in a laboratory. Analyze this image carefully:

{procedure_context}

ANALYSIS REQUIREMENTS:
1. EQUIPMENT STATUS: Describe robot position, pipette status, any visible movement
2. LIQUID COLORS: Identify all visible liquids and their colors (red, blue, yellow, clear, etc.)
3. CONTAINER POSITIONS: Note which positions/wells contain liquids
4. PROCEDURE COMPLIANCE: Does the current state match expected procedure step?
5. ERROR DETECTION: Any spills, contamination, wrong colors, or positioning errors?
6. SAFETY CONCERNS: Any visible safety issues or anomalies?

RESPOND IN THIS EXACT FORMAT:
STATUS: [NORMAL/WARNING/ERROR/CRITICAL]
EQUIPMENT: [equipment description]
COLORS_DETECTED: [list colors and locations]
POSITIONS: [active positions/wells]
COMPLIANCE: [YES/NO with brief reason]
ERRORS: [list any errors or "NONE"]
SAFETY: [any safety concerns or "OK"]
CONFIDENCE: [0.0-1.0]
DESCRIPTION: [detailed scene description]"""

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
            "equipment": "",
            "colors_detected": [],
            "positions": [],
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
                elif line.startswith('EQUIPMENT:'):
                    result["equipment"] = line.replace('EQUIPMENT:', '').strip()
                elif line.startswith('COLORS_DETECTED:'):
                    colors_str = line.replace('COLORS_DETECTED:', '').strip()
                    result["colors_detected"] = [c.strip() for c in colors_str.split(',') if c.strip()]
                elif line.startswith('POSITIONS:'):
                    positions_str = line.replace('POSITIONS:', '').strip()
                    result["positions"] = [p.strip() for p in positions_str.split(',') if p.strip()]
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
                    self.check_safety_compliance(analysis_result, frame)
                    
                    # Print status
                    status = analysis_result.get("status", "UNKNOWN")
                    colors = analysis_result.get("colors_detected", [])
                    
                    status_colors = {
                        "NORMAL": "\033[92m",   # Green
                        "WARNING": "\033[93m",  # Yellow
                        "ERROR": "\033[91m",    # Red
                        "CRITICAL": "\033[41m"  # Red background
                    }
                    color = status_colors.get(status, "\033[0m")
                    reset = "\033[0m"
                    
                    print(f"{color}🔬 Status: {status}{reset}")
                    print(f"🎨 Colors: {', '.join(colors) if colors else 'None detected'}")
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
        
        print(f"�� Starting procedure: {procedure_name}")
        step = self.procedures[procedure_name][0]
        print(f"📋 Step 1: {step.name} - {step.description}")
        
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
        """Add monitoring overlay to video"""
        height, width = frame.shape[:2]
        
        # Status color based on analysis
        status = self.last_analysis_result.get("status", "UNKNOWN")
        status_colors = {
            "NORMAL": (0, 255, 0),    # Green
            "WARNING": (0, 255, 255), # Yellow
            "ERROR": (0, 0, 255),     # Red
            "CRITICAL": (0, 0, 128),  # Dark red
            "UNKNOWN": (128, 128, 128) # Gray
        }
        color = status_colors.get(status, (128, 128, 128))
        
        # Main status box
        cv2.rectangle(frame, (10, 10), (700, 160), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (700, 160), color, 3)
        
        # Title
        cv2.putText(frame, "🧪 Liquid Handler Monitor", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Status
        cv2.putText(frame, f"Status: {status}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Current procedure
        proc_text = f"Procedure: {self.current_procedure or 'None'}"
        if self.current_procedure:
            proc_text += f" (Step {self.current_step + 1})"
        cv2.putText(frame, proc_text, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Error count
        error_count = len(self.error_events)
        error_color = (0, 0, 255) if error_count > 0 else (0, 255, 0)
        cv2.putText(frame, f"Errors: {error_count}", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, error_color, 1)
        
        # Controls
        cv2.putText(frame, "Controls: 'q'=quit, 's'=start procedure, 'n'=next step", (20, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Description at bottom
        if self.last_description:
            desc = self.last_description[:80] + "..." if len(self.last_description) > 80 else self.last_description
            cv2.rectangle(frame, (10, height - 60), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, f"Analysis: {desc}", (20, height - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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
        print("⚠️  Press 's' to start a procedure, 'n' for next step, 'q' to quit")
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
                elif key == ord('n'):
                    # Next step
                    self.next_step()
        
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
