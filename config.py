import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration management for VLM Real-Time Object Detection"""
    
    # Google Cloud Configuration
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    GOOGLE_CLOUD_PROJECT: str = os.getenv('GOOGLE_CLOUD_PROJECT', '')
    
    # Vertex AI Configuration
    VERTEX_AI_LOCATION: str = os.getenv('VERTEX_AI_LOCATION', 'us-central1')
    VERTEX_AI_MODEL: str = os.getenv('VERTEX_AI_MODEL', 'gemini-2.5-pro')
    
    # Backward compatibility with Gemini API
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
    GEMINI_LIVE_MODEL: str = os.getenv('GEMINI_LIVE_MODEL', 'gemini-2.5-pro')
    
    # Camera Configuration (Optimized for iPhone)
    CAMERA_INDEX: int = int(os.getenv('CAMERA_INDEX', '0'))  # iPhone camera is typically index 0
    CAMERA_WIDTH: int = int(os.getenv('CAMERA_WIDTH', '1920'))  # iPhone native resolution
    CAMERA_HEIGHT: int = int(os.getenv('CAMERA_HEIGHT', '1080'))  # iPhone native resolution
    CAMERA_BACKEND: str = os.getenv('CAMERA_BACKEND', 'avfoundation')  # Best for macOS/iPhone
    CAMERA_RETRY_ATTEMPTS: int = int(os.getenv('CAMERA_RETRY_ATTEMPTS', '3'))
    CAMERA_RETRY_DELAY: float = float(os.getenv('CAMERA_RETRY_DELAY', '1.0'))
    
    # iPhone-specific optimizations
    IPHONE_CAMERA_OPTIMIZATION: bool = os.getenv('IPHONE_CAMERA_OPTIMIZATION', 'true').lower() == 'true'
    CAMERA_BUFFER_SIZE: int = int(os.getenv('CAMERA_BUFFER_SIZE', '1'))  # Minimize latency
    
    # USB iPhone Connection Settings (NEW)
    USB_IPHONE_OPTIMIZATIONS: bool = os.getenv('USB_IPHONE_OPTIMIZATIONS', 'true').lower() == 'true'
    USB_CONNECTION_TIMEOUT: int = int(os.getenv('USB_CONNECTION_TIMEOUT', '30000'))  # 30 seconds timeout
    USB_RECONNECT_ATTEMPTS: int = int(os.getenv('USB_RECONNECT_ATTEMPTS', '3'))
    USB_FRAME_READ_TIMEOUT: float = float(os.getenv('USB_FRAME_READ_TIMEOUT', '5.0'))  # 5 seconds per frame read
    KEEP_ALIVE_INTERVAL: float = float(os.getenv('KEEP_ALIVE_INTERVAL', '1.0'))  # Read frame every 1 second to keep connection alive
    
    # Detection Configuration
    DETECTION_THRESHOLD: float = float(os.getenv('DETECTION_THRESHOLD', '0.5'))
    DETECTION_CONFIDENCE_THRESHOLD: float = float(os.getenv('DETECTION_CONFIDENCE_THRESHOLD', '0.7'))
    FRAME_SKIP_RATIO: int = int(os.getenv('FRAME_SKIP_RATIO', '2'))
    MAX_DETECTIONS_PER_FRAME: int = int(os.getenv('MAX_DETECTIONS_PER_FRAME', '10'))
    
    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '3'))
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    ENABLE_PERFORMANCE_MONITORING: bool = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true'
    PERFORMANCE_STATS_WINDOW: int = int(os.getenv('PERFORMANCE_STATS_WINDOW', '100'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE: str = os.getenv('LOG_FILE', '')
    
    # Streamlit Configuration
    STREAMLIT_PORT: int = int(os.getenv('STREAMLIT_PORT', '8501'))
    STREAMLIT_HOST: str = os.getenv('STREAMLIT_HOST', 'localhost')
    
    # Processing Configuration
    TARGET_FPS: int = int(os.getenv('TARGET_FPS', '30'))
    PROCESSING_INTERVAL: float = float(os.getenv('PROCESSING_INTERVAL', '0.1'))
    
    # Storage Configuration
    ENABLE_FRAME_STORAGE: bool = os.getenv('ENABLE_FRAME_STORAGE', 'false').lower() == 'true'
    STORAGE_DIRECTORY: str = os.getenv('STORAGE_DIRECTORY', 'captures')
    MAX_STORED_FRAMES: int = int(os.getenv('MAX_STORED_FRAMES', '1000'))
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Check required parameters
        if not self.GOOGLE_API_KEY and not self.GOOGLE_CLOUD_PROJECT:
            raise ValueError("Either GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT must be set")
        
        # For Vertex AI, project ID is required
        if not self.GOOGLE_CLOUD_PROJECT:
            logging.warning("GOOGLE_CLOUD_PROJECT not set. Some features may not work.")
        
        # Validate numeric parameters
        if self.DETECTION_THRESHOLD < 0 or self.DETECTION_THRESHOLD > 1:
            raise ValueError("DETECTION_THRESHOLD must be between 0 and 1")
        
        if self.FRAME_SKIP_RATIO < 1:
            raise ValueError("FRAME_SKIP_RATIO must be at least 1")
        
        if self.MAX_CONCURRENT_REQUESTS < 1:
            raise ValueError("MAX_CONCURRENT_REQUESTS must be at least 1")
        
        if self.REQUEST_TIMEOUT < 1:
            raise ValueError("REQUEST_TIMEOUT must be at least 1 second")
        
        # Validate camera parameters
        if self.CAMERA_INDEX < 0:
            raise ValueError("CAMERA_INDEX must be non-negative")
        
        # Create storage directory if needed
        if self.ENABLE_FRAME_STORAGE:
            os.makedirs(self.STORAGE_DIRECTORY, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        
        logging_config = {
            'level': log_level,
            'format': self.LOG_FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        
        if self.LOG_FILE:
            logging_config['filename'] = self.LOG_FILE
            logging_config['filemode'] = 'a'
        
        logging.basicConfig(**logging_config)
        
        # Set specific loggers
        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            key: getattr(self, key) 
            for key in self.__dataclass_fields__.keys()
        }
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logging.warning(f"Unknown configuration key: {key}")
    
    def validate_api_access(self) -> bool:
        """Validate API access and credentials"""
        try:
            if self.GOOGLE_CLOUD_PROJECT:
                # Try to validate Vertex AI access
                from google.auth import default
                from google.cloud import aiplatform
                
                credentials, project = default()
                if project != self.GOOGLE_CLOUD_PROJECT:
                    logging.warning(f"Default project ({project}) differs from configured project ({self.GOOGLE_CLOUD_PROJECT})")
                
                # Initialize Vertex AI to test access
                import vertexai
                vertexai.init(project=self.GOOGLE_CLOUD_PROJECT, location=self.VERTEX_AI_LOCATION)
                
                logging.info("✅ Vertex AI access validated")
                return True
                
            elif self.GOOGLE_API_KEY:
                # Try to validate Google AI Studio access
                import google.generativeai as genai
                genai.configure(api_key=self.GOOGLE_API_KEY)
                
                # Try to list models to validate access
                list(genai.list_models())
                
                logging.info("✅ Google AI Studio access validated")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"❌ API access validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the configured model"""
        if self.GOOGLE_CLOUD_PROJECT:
            return {
                "api_type": "Vertex AI",
                "model": self.VERTEX_AI_MODEL,
                "location": self.VERTEX_AI_LOCATION,
                "project": self.GOOGLE_CLOUD_PROJECT
            }
        else:
            return {
                "api_type": "Google AI Studio",
                "model": self.GEMINI_MODEL,
                "api_key_set": bool(self.GOOGLE_API_KEY)
            }

# Create global config instance
config = Config()

# Validate configuration on import
try:
    config._validate_config()
    logging.info("✅ Configuration validated successfully")
except Exception as e:
    logging.error(f"❌ Configuration validation failed: {e}")
    raise

# Export commonly used values for backward compatibility
GOOGLE_API_KEY = config.GOOGLE_API_KEY
GOOGLE_CLOUD_PROJECT = config.GOOGLE_CLOUD_PROJECT
GEMINI_MODEL = config.GEMINI_MODEL
CAMERA_INDEX = config.CAMERA_INDEX
DETECTION_THRESHOLD = config.DETECTION_THRESHOLD 