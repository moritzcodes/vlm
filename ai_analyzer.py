#!/usr/bin/env python3
"""
AI analysis module for the Liquid Handler Monitor
"""

import os
import io
import logging
from PIL import Image
import cv2
from typing import Optional, Dict, Any
from data_models import AnalysisResult, ProcedureStep
from config import Config

class AIAnalyzer:
    """Handles AI model setup and frame analysis"""
    
    def __init__(self):
        self.model = None
        self.setup_model()
    
    def setup_model(self) -> bool:
        """Setup Google Gen AI SDK for Gemini 2.5 Pro"""
        api_key = os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            print("❌ GOOGLE_API_KEY not found in .env file")
            return False
        
        try:
            from google import genai
            
            self.model = genai.Client(
                vertexai=False, 
                api_key=api_key
            )
            print("🤖 Google Gen AI Gemini 2.5 Pro loaded for liquid handler monitoring")
            return True
            
        except Exception as e:
            print(f"❌ Google Gen AI setup failed: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if AI model is ready"""
        return self.model is not None
    
    def analyze_frame(self, frame, current_procedure: Optional[str] = None, 
                     current_step: Optional[ProcedureStep] = None) -> AnalysisResult:
        """Send frame to AI for comprehensive analysis"""
        if not self.model:
            return AnalysisResult(
                status="ERROR",
                description="AI model not available",
                confidence=0.0
            )
        
        try:
            # Convert frame to RGB and create image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=90)
            img_bytes = img_buffer.getvalue()
            
            from google.genai import types
            
            image_part = types.Part.from_bytes(
                data=img_bytes, 
                mime_type="image/jpeg"
            )
            
            # Build context-aware prompt
            prompt = self._build_analysis_prompt(current_procedure, current_step)
            
            response = self.model.models.generate_content(
                model=Config.AI.model_name,
                contents=[
                    types.Content(parts=[
                        types.Part.from_text(text=prompt),
                        image_part
                    ])
                ]
            )
            
            return self._parse_response(response.text)
            
        except Exception as e:
            logging.error(f"AI analysis failed: {e}")
            return AnalysisResult(
                status="ERROR",
                description=f"Analysis failed: {str(e)}",
                confidence=0.0
            )
    
    def _build_analysis_prompt(self, current_procedure: Optional[str], 
                              current_step: Optional[ProcedureStep]) -> str:
        """Build context-aware analysis prompt"""
        
        # Get current procedure context
        procedure_context = ""
        if current_procedure and current_step:
            procedure_context = f"""
CURRENT PROCEDURE: {current_procedure}
CURRENT STEP: {current_step.name} ({current_step.description})
EXPECTED COLORS: {', '.join(current_step.expected_colors)}
EXPECTED POSITIONS: {', '.join(current_step.expected_positions)}
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

        return f"""LIQUID HANDLER MONITORING SYSTEM
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
    
    def _parse_response(self, response_text: str) -> AnalysisResult:
        """Parse AI response into structured data"""
        result = AnalysisResult(raw_response=response_text)
        
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('STATUS:'):
                    result.status = line.replace('STATUS:', '').strip()
                elif line.startswith('ARM_BLOCKING:'):
                    arm_str = line.replace('ARM_BLOCKING:', '').strip().upper()
                    result.arm_blocking = arm_str.startswith('YES')
                elif line.startswith('PLATE_VISIBLE:'):
                    plate_str = line.replace('PLATE_VISIBLE:', '').strip().upper()
                    result.plate_visible = plate_str.startswith('YES')
                elif line.startswith('WELL_ANALYSIS:'):
                    wells_str = line.replace('WELL_ANALYSIS:', '').strip()
                    # Parse well analysis: A1:blue, A2:red, etc.
                    well_pairs = [w.strip() for w in wells_str.split(',') if ':' in w]
                    for pair in well_pairs:
                        if ':' in pair:
                            well, color = pair.split(':', 1)
                            result.well_analysis[well.strip()] = color.strip()
                elif line.startswith('FAILED_WELLS:'):
                    failed_str = line.replace('FAILED_WELLS:', '').strip()
                    if failed_str.upper() != "NONE":
                        result.failed_wells = [w.strip() for w in failed_str.split(',') if w.strip()]
                elif line.startswith('COMPLIANCE:'):
                    compliance_str = line.replace('COMPLIANCE:', '').strip().upper()
                    result.compliance = compliance_str.startswith('YES')
                elif line.startswith('ERRORS:'):
                    errors_str = line.replace('ERRORS:', '').strip()
                    if errors_str.upper() != "NONE":
                        result.errors = [e.strip() for e in errors_str.split(',') if e.strip()]
                elif line.startswith('SAFETY:'):
                    result.safety = line.replace('SAFETY:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        result.confidence = float(line.replace('CONFIDENCE:', '').strip())
                    except:
                        result.confidence = 0.5
                elif line.startswith('DESCRIPTION:'):
                    result.description = line.replace('DESCRIPTION:', '').strip()
                    
        except Exception as e:
            logging.error(f"Failed to parse AI response: {e}")
            result.description = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
        return result 