#!/usr/bin/env python3
"""
Data models for the Liquid Handler Monitor
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime

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

@dataclass
class FeedbackItem:
    """VLM feedback item"""
    type: str  # critical, error, warning, info, success
    icon: str
    title: str
    message: str
    action: str

@dataclass
class AnalysisResult:
    """AI analysis result"""
    status: str = "UNKNOWN"
    arm_blocking: bool = False
    plate_visible: bool = False
    well_analysis: Dict[str, str] = None
    failed_wells: List[str] = None
    compliance: bool = False
    errors: List[str] = None
    safety: str = "UNKNOWN"
    confidence: float = 0.0
    description: str = ""
    raw_response: str = ""
    
    def __post_init__(self):
        if self.well_analysis is None:
            self.well_analysis = {}
        if self.failed_wells is None:
            self.failed_wells = []
        if self.errors is None:
            self.errors = []

class ProcedureDefinitions:
    """Standard operating procedures"""
    
    @staticmethod
    def get_procedures() -> Dict[str, List[ProcedureStep]]:
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