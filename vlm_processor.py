import asyncio
import json
import logging
import time
import base64
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Vertex AI imports
from google.cloud import aiplatform
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage
from vertexai.generative_models import HarmCategory, HarmBlockThreshold, SafetySetting

from config import config

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Structured detection result from VLM"""
    objects: List[str]
    descriptions: List[str]
    confidence_scores: List[float]
    bounding_boxes: List[Tuple[float, float, float, float]]
    processing_time: float
    frame_timestamp: float

class ObjectDetectionAnalyzer:
    """Advanced analyzer for VLM detection results"""
    
    def __init__(self):
        self.detection_history = []
        self.object_tracker = {}
        
    def analyze_detections(self, detection_text: str, confidence_threshold: float = 0.5) -> DetectionResult:
        """Parse and analyze VLM detection results"""
        start_time = time.time()
        
        objects = []
        descriptions = []
        confidence_scores = []
        bounding_boxes = []
        
        try:
            # Parse structured JSON response if available
            if detection_text.strip().startswith('{'):
                data = json.loads(detection_text)
                if 'detections' in data:
                    for detection in data['detections']:
                        objects.append(detection.get('object', 'unknown'))
                        descriptions.append(detection.get('description', ''))
                        confidence_scores.append(detection.get('confidence', 0.0))
                        bbox = detection.get('bounding_box', [0, 0, 0, 0])
                        bounding_boxes.append(tuple(bbox))
            else:
                # Parse natural language response
                objects, descriptions, bboxes = self._parse_natural_language(detection_text)
                confidence_scores = [0.8] * len(objects)  # Default confidence
                bounding_boxes = bboxes
                
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse structured response: {e}")
            # Fallback to simple text parsing
            objects, descriptions, bboxes = self._parse_natural_language(detection_text)
            confidence_scores = [0.7] * len(objects)
            bounding_boxes = bboxes
        
        # Filter by confidence threshold
        filtered_results = [
            (obj, desc, conf, bbox) 
            for obj, desc, conf, bbox in zip(objects, descriptions, confidence_scores, bounding_boxes)
            if conf >= confidence_threshold
        ]
        
        if filtered_results:
            objects, descriptions, confidence_scores, bounding_boxes = zip(*filtered_results)
        else:
            objects = descriptions = confidence_scores = bounding_boxes = []
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            objects=list(objects),
            descriptions=list(descriptions),
            confidence_scores=list(confidence_scores),
            bounding_boxes=list(bounding_boxes),
            processing_time=processing_time,
            frame_timestamp=time.time()
        )
    
    def _parse_natural_language(self, text: str) -> Tuple[List[str], List[str], List[Tuple[float, float, float, float]]]:
        """Parse natural language detection results"""
        objects = []
        descriptions = []
        bounding_boxes = []
        
        # Look for common object detection patterns
        patterns = [
            r"(?i)(person|people|human|man|woman|child)",
            r"(?i)(car|vehicle|truck|bus|motorcycle|bike)",
            r"(?i)(phone|smartphone|mobile|device)",
            r"(?i)(bottle|cup|glass|mug)",
            r"(?i)(book|paper|document|magazine)",
            r"(?i)(chair|table|desk|furniture)",
            r"(?i)(dog|cat|pet|animal)",
            r"(?i)(food|pizza|sandwich|fruit|apple|banana)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match.lower() not in [obj.lower() for obj in objects]:
                    objects.append(match.lower())
                    descriptions.append(f"Detected {match.lower()} in the image")
                    # Default bounding box (normalized coordinates)
                    bounding_boxes.append((0.1, 0.1, 0.9, 0.9))
        
        # If no patterns matched, extract key nouns
        if not objects:
            words = re.findall(r'\b[A-Za-z]+\b', text)
            for word in words[:3]:  # Take first 3 words as potential objects
                if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they', 'have']:
                    objects.append(word.lower())
                    descriptions.append(f"Detected {word.lower()}")
                    bounding_boxes.append((0.1, 0.1, 0.9, 0.9))
        
        return objects, descriptions, bounding_boxes
    
    def calculate_iou(self, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """Calculate Intersection over Union for bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class VertexAIVLMProcessor:
    """Vertex AI-powered Vision Language Model processor for real-time object detection"""
    
    def __init__(self, config_dict=None):
        self.config = config_dict or config
        self.model = None
        self.analyzer = ObjectDetectionAnalyzer()
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0
        self.error_count = 0
        
        # Detection prompts
        self.prompts = {
            "realtime": """Analyze this image and detect all visible objects. 
            Provide a brief list of what you see. Focus on the most prominent objects.
            Be concise for real-time processing.""",
            
            "objects": """List all objects you can identify in this image.
            Format: object1, object2, object3, etc.
            Focus on accuracy and include common items, people, vehicles, etc.""",
            
            "detailed": """Perform detailed object detection on this image.
            Identify and describe:
            1. All people and their activities
            2. All vehicles and their types
            3. All objects and their locations
            4. The general scene and context
            
            Provide specific details about each detected object.""",
            
            "structured": """Analyze this image and return detected objects in this JSON format:
            {
                "detections": [
                    {
                        "object": "object_name",
                        "description": "detailed description",
                        "confidence": 0.0-1.0,
                        "bounding_box": [x_min, y_min, x_max, y_max]
                    }
                ]
            }
            Use normalized coordinates (0.0-1.0) for bounding boxes."""
        }
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
    
    def initialize_model(self) -> bool:
        """Initialize the model (for backward compatibility)"""
        try:
            if self.model is not None:
                logger.info("✅ VLM model already initialized")
                return True
            else:
                self._initialize_vertex_ai()
                return self.model is not None
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
    
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with proper authentication"""
        try:
            project_id = self.config.get('GOOGLE_CLOUD_PROJECT')
            location = self.config.get('VERTEX_AI_LOCATION', 'us-central1')
            
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT must be set for Vertex AI")
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            # Initialize the model
            model_name = self.config.get('VERTEX_AI_MODEL', 'gemini-2.5-pro')
            self.model = GenerativeModel(model_name)
            
            # Configure safety settings
            self.safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
                ),
            ]
            
            logger.info(f"Vertex AI initialized successfully with model: {model_name}")
            logger.info(f"Project: {project_id}, Location: {location}")
            
        except DefaultCredentialsError as e:
            logger.error(f"Vertex AI authentication failed: {e}")
            logger.info("Please run: gcloud auth application-default login")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    async def process_frame_async(self, frame: np.ndarray, prompt_type: str = "realtime") -> Dict[str, Any]:
        """Process frame asynchronously with Vertex AI"""
        start_time = time.time()
        
        try:
            # Convert frame to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Resize image if too large (Vertex AI has size limits)
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to Vertex AI Image format  
            # Save PIL image to bytes buffer
            import io
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create Vertex AI Image from bytes
            vertex_image = VertexImage.from_bytes(img_byte_arr)
            
            # Get the appropriate prompt
            prompt = self.prompts.get(prompt_type, self.prompts["realtime"])
            
            # Generate content
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._generate_content,
                vertex_image,
                prompt
            )
            
            # Process the response
            detection_text = response.text if response else "No response"
            
            # Analyze detections
            detection_result = self.analyzer.analyze_detections(
                detection_text,
                confidence_threshold=self.config.get('DETECTION_THRESHOLD', 0.5)
            )
            
            processing_time = time.time() - start_time
            self.request_count += 1
            self.total_processing_time += processing_time
            
            return {
                "success": True,
                "detection_result": detection_result,
                "raw_response": detection_text,
                "processing_time": processing_time,
                "frame_shape": frame.shape,
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            self.error_count += 1
            processing_time = time.time() - start_time
            logger.error(f"Error processing frame: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
                "detection_result": DetectionResult([], [], [], [], processing_time, time.time())
            }
    
    def _generate_content(self, image: VertexImage, prompt: str):
        """Generate content using Vertex AI (synchronous)"""
        try:
            response = self.model.generate_content(
                [prompt, image],
                safety_settings=self.safety_settings,
                generation_config={
                    "max_output_tokens": 1000,
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            return response
        except Exception as e:
            logger.error(f"Vertex AI generation failed: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray, prompt_type: str = "realtime") -> Dict[str, Any]:
        """Synchronous wrapper for frame processing"""
        return asyncio.run(self.process_frame_async(frame, prompt_type))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        success_rate = (
            (self.request_count - self.error_count) / self.request_count * 100 
            if self.request_count > 0 else 0
        )
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "average_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "requests_per_minute": self.request_count / (self.total_processing_time / 60) if self.total_processing_time > 0 else 0
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.request_count = 0
        self.total_processing_time = 0
        self.error_count = 0

# Backward compatibility alias
GeminiVLMProcessor = VertexAIVLMProcessor 