#!/usr/bin/env python3
"""
UI components for the Liquid Handler Monitor
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from data_models import AnalysisResult, FeedbackItem
from config import Config

class UIRenderer:
    """Handles all UI rendering and overlay drawing"""
    
    def __init__(self):
        self.ui_colors = Config.UI_COLORS
        
    def draw_professional_ui(self, frame, analysis_result: AnalysisResult, 
                           current_procedure: Optional[str], current_step: int,
                           error_count: int, well_status: Dict, failed_wells: List):
        """Draw the main professional UI overlay"""
        height, width = frame.shape[:2]
        
        # Get status color
        status = analysis_result.status
        status_color = Config.STATUS_COLORS.get(status, (128, 128, 128))
        
        # Draw main UI components
        self._draw_header(frame, status, status_color, width)
        self._draw_status_panels(frame, analysis_result, error_count, width)
        self._draw_procedure_panel(frame, current_procedure, current_step)
        self._draw_vlm_feedback(frame, analysis_result, width, height)
        
        if current_procedure == "blue_red_mixing":
            self._draw_well_status_grid(frame, analysis_result.well_analysis, failed_wells, height)
    
    def _draw_header(self, frame, status: str, status_color: Tuple[int, int, int], width: int):
        """Draw the main header"""
        # Header background
        self._draw_glass_panel(frame, (0, 0), (width, 90), self.ui_colors.background, 0.8)
        
        # Status indicator
        cv2.circle(frame, (50, 45), 25, status_color, -1)
        cv2.circle(frame, (50, 45), 25, self.ui_colors.accent_glow, 3)
        cv2.circle(frame, (50, 45), 18, (255, 255, 255), 3)
        cv2.circle(frame, (50, 45), 12, status_color, 2)
        
        # Title
        cv2.putText(frame, "LIQUID HANDLER MONITOR", (90, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.ui_colors.text_primary, 3)
        cv2.putText(frame, f"STATUS: {status}", (90, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Time display
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, "TIME", (width - 180, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors.text_secondary, 2)
        cv2.putText(frame, timestamp, (width - 180, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.ui_colors.accent, 3)
    
    def _draw_status_panels(self, frame, analysis_result: AnalysisResult, 
                           error_count: int, width: int):
        """Draw status indicator panels"""
        # Error panel
        self._draw_error_panel(frame, error_count, width - 300, 105)
        
        # Confidence panel
        self._draw_confidence_panel(frame, analysis_result.confidence, width - 300, 190)
    
    def _draw_error_panel(self, frame, error_count: int, x: int, y: int):
        """Draw error status panel"""
        error_color = self.ui_colors.error if error_count > 0 else self.ui_colors.success
        panel_width, panel_height = 280, 75
        
        # Panel background
        self._draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), 
                              self.ui_colors.glass, 0.6)
        
        # Status indicator
        cv2.circle(frame, (x + 30, y + 37), 15, error_color, -1)
        cv2.circle(frame, (x + 30, y + 37), 15, self.ui_colors.accent_glow, 3)
        cv2.circle(frame, (x + 30, y + 37), 10, (255, 255, 255), 2)
        
        # Status text
        cv2.putText(frame, "SYSTEM STATUS", (x + 55, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors.text_primary, 2)
        
        if error_count > 0:
            cv2.putText(frame, f"{error_count} ERRORS DETECTED", (x + 55, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 2)
        else:
            cv2.putText(frame, "ALL SYSTEMS OK", (x + 55, y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.ui_colors.success, 2)
    
    def _draw_confidence_panel(self, frame, confidence: float, x: int, y: int):
        """Draw AI confidence indicator"""
        conf_color = (self.ui_colors.success if confidence > 0.8 else 
                     self.ui_colors.warning if confidence > 0.5 else self.ui_colors.error)
        
        panel_width, panel_height = 280, 55
        
        # Panel background
        self._draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), 
                              self.ui_colors.glass, 0.5)
        
        # Label
        cv2.putText(frame, "AI CONFIDENCE", (x + 15, y + 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors.text_primary, 2)
        
        # Progress bar
        bar_width, bar_height = 180, 12
        bar_x, bar_y = x + 15, y + 30
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors.background, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors.accent, 2)
        
        # Fill bar
        fill_width = int(bar_width * confidence)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         conf_color, -1)
        
        # Percentage
        cv2.putText(frame, f"{confidence:.0%}", (x + 210, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
    
    def _draw_procedure_panel(self, frame, current_procedure: Optional[str], current_step: int):
        """Draw procedure status panel"""
        if not current_procedure:
            return
        
        from data_models import ProcedureDefinitions
        procedures = ProcedureDefinitions.get_procedures()
        
        proc_name = current_procedure.replace('_', ' ').upper()
        x, y = 20, 105
        panel_width, panel_height = 450, 90
        
        # Panel background
        self._draw_glass_panel(frame, (x, y), (x + panel_width, y + panel_height), 
                              self.ui_colors.glass, 0.6)
        
        # Procedure indicator
        cv2.circle(frame, (x + 30, y + 30), 15, self.ui_colors.accent, -1)
        cv2.circle(frame, (x + 30, y + 30), 15, self.ui_colors.accent_glow, 3)
        cv2.circle(frame, (x + 30, y + 30), 10, (255, 255, 255), 2)
        
        # Procedure text
        cv2.putText(frame, "ACTIVE PROCEDURE", (x + 55, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors.text_primary, 2)
        cv2.putText(frame, proc_name, (x + 55, y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors.accent, 2)
        
        # Step progress
        total_steps = len(procedures[current_procedure])
        step_progress = f"STEP {current_step + 1} OF {total_steps}"
        cv2.putText(frame, step_progress, (x + 55, y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors.text_secondary, 1)
        
        # Progress bar
        bar_width, bar_height = 300, 6
        bar_x, bar_y = x + 55, y + 78
        progress = (current_step + 1) / total_steps
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors.background, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     self.ui_colors.accent, 1)
        
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                         self.ui_colors.accent, -1)
    
    def _draw_vlm_feedback(self, frame, analysis_result: AnalysisResult, width: int, height: int):
        """Draw VLM feedback overlay"""
        feedback_items = self._generate_feedback_items(analysis_result)
        if not feedback_items:
            return
        
        # Calculate panel dimensions
        panel_height = 45 + (len(feedback_items) * 70)
        panel_width = 600
        panel_x = width - panel_width - 25
        panel_y = height // 2 - panel_height // 2
        
        # Ensure panel stays on screen
        panel_y = max(110, min(panel_y, height - panel_height - 120))
        
        # Draw main feedback panel
        self._draw_glass_panel(frame, (panel_x, panel_y), 
                              (panel_x + panel_width, panel_y + panel_height), 
                              self.ui_colors.background, 0.85)
        
        # Panel header
        cv2.putText(frame, "AI ANALYSIS FEEDBACK", (panel_x + 25, panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.ui_colors.text_primary, 2)
        
        # Draw feedback items
        item_y = panel_y + 55
        for item in feedback_items:
            self._draw_feedback_item(frame, panel_x, item_y, panel_width, item)
            item_y += 70
    
    def _draw_feedback_item(self, frame, x: int, y: int, width: int, item: FeedbackItem):
        """Draw individual feedback item"""
        colors = {
            'critical': self.ui_colors.critical,
            'error': self.ui_colors.error,
            'warning': self.ui_colors.warning,
            'info': self.ui_colors.accent,
            'success': self.ui_colors.success
        }
        item_color = colors.get(item.type, self.ui_colors.text_primary)
        
        # Status indicator
        cv2.circle(frame, (x + 30, y + 25), 12, item_color, -1)
        cv2.circle(frame, (x + 30, y + 25), 12, (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, item.title, (x + 55, y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.ui_colors.text_primary, 2)
        
        # Message
        cv2.putText(frame, item.message, (x + 55, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors.text_secondary, 1)
        
        # Action (if not info type)
        if item.type != 'info':
            cv2.putText(frame, f"→ {item.action}", (x + 55, y + 58), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, item_color, 1)
    
    def _draw_well_status_grid(self, frame, well_analysis: Dict, failed_wells: List, height: int):
        """Draw well status grid"""
        if not well_analysis:
            return
        
        # Position for grid (bottom left)
        grid_x, grid_y = 25, height - 180
        cell_size, spacing = 25, 5
        
        # Grid background
        grid_width = 6 * (cell_size + spacing) - spacing + 30
        grid_height = 2 * (cell_size + spacing) - spacing + 60
        
        self._draw_glass_panel(frame, (grid_x, grid_y), 
                              (grid_x + grid_width, grid_y + grid_height), 
                              self.ui_colors.glass, 0.6)
        
        # Grid title
        cv2.putText(frame, "WELL STATUS", (grid_x + 15, grid_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors.text_primary, 2)
        
        # Draw wells
        for row_idx, row in enumerate(['A', 'B']):
            for col_idx in range(1, 7):  # Columns 1-6
                well_id = f"{row}{col_idx}"
                
                cell_x = grid_x + 15 + col_idx * (cell_size + spacing)
                cell_y = grid_y + 40 + row_idx * (cell_size + spacing)
                
                # Determine well color
                if well_id in failed_wells:
                    well_color = self.ui_colors.error
                elif well_id in well_analysis:
                    well_color = self.ui_colors.success
                else:
                    well_color = self.ui_colors.text_secondary
                
                # Draw well
                cv2.rectangle(frame, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), 
                             well_color, -1)
                cv2.rectangle(frame, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), 
                             (255, 255, 255), 2)
                
                # Well label
                cv2.putText(frame, well_id, (cell_x + 4, cell_y + 17), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _generate_feedback_items(self, analysis_result: AnalysisResult) -> List[FeedbackItem]:
        """Generate prioritized feedback items"""
        items = []
        
        # Priority 1: Critical errors
        if analysis_result.status == "CRITICAL":
            items.append(FeedbackItem(
                type='critical',
                icon='🚨',
                title='CRITICAL ISSUE DETECTED',
                message='Immediate attention required',
                action='Check procedure and safety'
            ))
        
        # Priority 2: Procedure compliance
        if not analysis_result.compliance and not analysis_result.arm_blocking:
            items.append(FeedbackItem(
                type='warning',
                icon='⚠️',
                title='PROCEDURE NON-COMPLIANCE',
                message='Current step not being followed correctly',
                action='Review procedure requirements'
            ))
        
        # Priority 3: Failed wells
        if analysis_result.failed_wells:
            wells_text = ", ".join(analysis_result.failed_wells[:4])
            if len(analysis_result.failed_wells) > 4:
                wells_text += f" +{len(analysis_result.failed_wells) - 4} more"
            
            items.append(FeedbackItem(
                type='error',
                icon='🧪',
                title=f'{len(analysis_result.failed_wells)} WELLS FAILED',
                message=f'Wells {wells_text} need attention',
                action='Check liquid colors and mixing'
            ))
        
        # Priority 4: Specific errors
        if analysis_result.errors and analysis_result.errors != ["NONE"]:
            for error in analysis_result.errors[:2]:
                items.append(FeedbackItem(
                    type='error',
                    icon='❌',
                    title='OPERATION ERROR',
                    message=error[:50] + "..." if len(error) > 50 else error,
                    action='Investigate and correct'
                ))
        
        # Priority 5: Operational status
        if analysis_result.arm_blocking:
            items.append(FeedbackItem(
                type='info',
                icon='🤖',
                title='ROBOT OPERATING',
                message='Analysis will resume when arm clears',
                action='Monitoring paused - normal operation'
            ))
        elif not analysis_result.plate_visible:
            items.append(FeedbackItem(
                type='info',
                icon='👁️',
                title='WAITING FOR CLEAR VIEW',
                message='Cannot analyze well plate contents',
                action='Ensure camera has clear view'
            ))
        
        # Priority 6: Success feedback
        if (analysis_result.status == "NORMAL" and analysis_result.compliance and 
            not analysis_result.failed_wells and analysis_result.plate_visible):
            items.append(FeedbackItem(
                type='success',
                icon='✅',
                title='PROCEDURE ON TRACK',
                message='All operations proceeding correctly',
                action='Continue monitoring'
            ))
        
        return items[:3]  # Show max 3 feedback items
    
    def _draw_glass_panel(self, frame, top_left: Tuple[int, int], 
                         bottom_right: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.3):
        """Draw a glass-like panel with transparency"""
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Thin border
        cv2.rectangle(frame, top_left, bottom_right, color, 1)

class CropUI:
    """Handles crop interface UI"""
    
    def __init__(self):
        self.ui_colors = Config.UI_COLORS
        self.visible = False
        self.position = (50, 100)
        self.buttons = []
        self.sliders = {}
        self.hover_element = None
        self.active_element = None
        
    def toggle_visibility(self):
        """Toggle crop UI visibility"""
        self.visible = not self.visible
        if self.visible:
            self._initialize_elements()
    
    def _initialize_elements(self):
        """Initialize UI elements"""
        ui_x, ui_y = self.position
        
        # Create buttons
        self.buttons = [
            # Preset buttons
            {'id': 'preset_full', 'type': 'button', 'rect': (ui_x + 10, ui_y + 40, 80, 25), 'text': 'Full Frame', 'preset': 'full'},
            {'id': 'preset_center', 'type': 'button', 'rect': (ui_x + 100, ui_y + 40, 80, 25), 'text': 'Center 50%', 'preset': 'center'},
            {'id': 'preset_wellplate', 'type': 'button', 'rect': (ui_x + 10, ui_y + 70, 80, 25), 'text': 'Well Plate', 'preset': 'wellplate'},
            {'id': 'preset_top', 'type': 'button', 'rect': (ui_x + 100, ui_y + 70, 80, 25), 'text': 'Top Half', 'preset': 'top_half'},
            
            # Action buttons
            {'id': 'reset', 'type': 'button', 'rect': (ui_x + 10, ui_y + 140, 60, 25), 'text': 'Reset', 'action': 'reset'},
            {'id': 'undo', 'type': 'button', 'rect': (ui_x + 80, ui_y + 140, 60, 25), 'text': 'Undo', 'action': 'undo'},
            {'id': 'apply', 'type': 'button', 'rect': (ui_x + 150, ui_y + 140, 60, 25), 'text': 'Apply', 'action': 'apply'},
        ]
    
    def draw(self, frame):
        """Draw the crop UI interface"""
        if not self.visible:
            return
        
        ui_x, ui_y = self.position
        panel_width, panel_height = 220, 200
        
        # Draw main panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (ui_x, ui_y), (ui_x + panel_width, ui_y + panel_height), 
                     (25, 35, 45), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Panel border and header
        cv2.rectangle(frame, (ui_x, ui_y), (ui_x + panel_width, ui_y + panel_height), 
                     (100, 200, 255), 2)
        cv2.rectangle(frame, (ui_x, ui_y), (ui_x + panel_width, ui_y + 30), 
                     (100, 200, 255), -1)
        cv2.putText(frame, "CROP CONTROLS", (ui_x + 10, ui_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw buttons
        for button in self.buttons:
            self._draw_button(frame, button)
    
    def _draw_button(self, frame, button):
        """Draw a UI button"""
        bx, by, bw, bh = button['rect']
        
        # Button colors
        is_hovered = self.hover_element == button
        is_active = self.active_element == button
        
        if is_active:
            bg_color = (80, 160, 255)
        elif is_hovered:
            bg_color = (120, 180, 255)
        else:
            bg_color = (60, 70, 80)
        
        # Draw button
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), bg_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)
        
        # Button text
        text_size = cv2.getTextSize(button['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = bx + (bw - text_size[0]) // 2
        text_y = by + (bh + text_size[1]) // 2
        cv2.putText(frame, button['text'], (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_element_at_position(self, x: int, y: int):
        """Get UI element at position"""
        for button in self.buttons:
            bx, by, bw, bh = button['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return button
        return None 