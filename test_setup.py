#!/usr/bin/env python3
"""
VLM Setup Validation Script
Tests all components and dependencies for the VLM Real-Time Object Detection system
"""

import logging
import sys
import os
import importlib
from typing import Dict, List, Any

# Configure logging for testing
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def test_package_imports() -> bool:
    """Test importing all required packages"""
    logger.info("🔍 Testing package imports...")
    
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy', 
        'PIL': 'Pillow',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'python-dotenv': {'module': 'dotenv', 'name': 'python-dotenv'},
        'google-cloud-aiplatform': {'module': 'google.cloud.aiplatform', 'name': 'Google Cloud AI Platform'},
        'vertexai': 'Vertex AI',
        'google-generativeai': {'module': 'google.generativeai', 'name': 'Google Generative AI (fallback)'},
    }
    
    failed_imports = []
    
    for package, info in packages.items():
        try:
            if isinstance(info, dict):
                module_name = info['module']
                display_name = info['name']
            else:
                module_name = package
                display_name = info
            
            importlib.import_module(module_name)
            logger.info(f"  ✅ {display_name}")
        except ImportError as e:
            logger.error(f"  ❌ {display_name}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_configuration() -> bool:
    """Test configuration loading and validation"""
    logger.info("🔍 Testing configuration...")
    
    try:
        from config import config
        
        # Test basic configuration loading
        logger.info("  ✅ Configuration loaded")
        
        # Check API configuration
        has_vertex_ai = bool(config.GOOGLE_CLOUD_PROJECT)
        has_api_key = bool(config.GOOGLE_API_KEY)
        
        if has_vertex_ai:
            logger.info(f"  ✅ Vertex AI configured (Project: {config.GOOGLE_CLOUD_PROJECT})")
            logger.info(f"  📍 Location: {config.VERTEX_AI_LOCATION}")
            logger.info(f"  🤖 Model: {config.VERTEX_AI_MODEL}")
        elif has_api_key:
            logger.info("  ✅ Google AI Studio API key is set")
            logger.info(f"  🤖 Model: {config.GEMINI_MODEL}")
        else:
            logger.error("  ❌ No API configuration found")
            logger.info("  Run: ./setup_env.sh to configure authentication")
            return False
        
        # Display other key settings
        logger.info(f"  📷 Camera index: {config.CAMERA_INDEX}")
        logger.info(f"  🎯 Detection threshold: {config.DETECTION_THRESHOLD}")
        logger.info(f"  ⚡ Frame skip ratio: {config.FRAME_SKIP_RATIO}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Configuration error: {e}")
        return False

def test_vertex_ai_authentication() -> bool:
    """Test Vertex AI authentication and access"""
    logger.info("🔍 Testing Vertex AI authentication...")
    
    try:
        from config import config
        if not config.GOOGLE_CLOUD_PROJECT:
            logger.warning("  ⚠️  Vertex AI not configured (no project ID)")
            return True  # Not an error if not using Vertex AI
        
        from google.auth import default
        from google.cloud import aiplatform
        import vertexai
        
        # Test authentication
        credentials, project = default()
        logger.info(f"  ✅ Authentication successful")
        logger.info(f"  📁 Authenticated project: {project}")
        
        if project != config.GOOGLE_CLOUD_PROJECT:
            logger.warning(f"  ⚠️  Project mismatch: authenticated={project}, configured={config.GOOGLE_CLOUD_PROJECT}")
        
        # Test Vertex AI initialization
        vertexai.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.VERTEX_AI_LOCATION)
        logger.info(f"  ✅ Vertex AI initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Vertex AI authentication failed: {e}")
        logger.info("  💡 Try running: gcloud auth application-default login")
        return False

def test_camera_access() -> bool:
    """Test camera access and availability"""
    logger.info("🔍 Testing camera access...")
    
    try:
        import cv2
        from config import config
        
        # Test different camera indices
        camera_found = False
        for i in range(3):  # Test cameras 0, 1, 2
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logger.info(f"  ✅ Camera {i} accessible ({frame.shape[1]}x{frame.shape[0]})")
                        camera_found = True
                        if i == config.CAMERA_INDEX:
                            logger.info(f"  🎯 Using configured camera {i}")
                    cap.release()
                    break
            except Exception as e:
                logger.debug(f"Camera {i} failed: {e}")
                continue
        
        if not camera_found:
            logger.warning("  ⚠️  No cameras detected")
            logger.info("  💡 Check camera permissions and connections")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Camera test failed: {e}")
        return False

def test_vlm_processor() -> bool:
    """Test VLM processor initialization"""
    logger.info("🔍 Testing VLM processor...")
    
    try:
        from config import config
        from vlm_processor import VertexAIVLMProcessor
        
        # Test processor initialization
        processor = VertexAIVLMProcessor(config)
        logger.info(f"  ✅ VLM processor initialized successfully")
        
        # Test model info
        model_info = config.get_model_info()
        logger.info(f"  🤖 API Type: {model_info['api_type']}")
        logger.info(f"  📱 Model: {model_info['model']}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ VLM processor failed: {e}")
        return False

def test_streamlit_components() -> bool:
    """Test Streamlit application components"""
    logger.info("🔍 Testing Streamlit components...")
    
    try:
        # Test streamlit imports
        import streamlit as st
        import plotly
        
        # Test streamlit app import
        import streamlit_app
        logger.info("  ✅ Streamlit app imports successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Streamlit components failed: {e}")
        return False

def test_main_application() -> bool:
    """Test main application module"""
    logger.info("🔍 Testing main application...")
    
    try:
        import main
        logger.info("  ✅ Main application imports successfully")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Main application failed: {e}")
        return False

def run_all_tests() -> None:
    """Run all validation tests"""
    logger.info("🚀 Starting VLM setup validation...")
    logger.info("==================================================")
    
    tests = [
        ("Package Imports", test_package_imports),
        ("Configuration", test_configuration),
        ("Vertex AI Authentication", test_vertex_ai_authentication),
        ("Camera Access", test_camera_access),
        ("VLM Processor", test_vlm_processor),
        ("Streamlit Components", test_streamlit_components),
        ("Main Application", test_main_application),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("\n==================================================")
    logger.info("📊 TEST SUMMARY")
    logger.info("==================================================")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    logger.info("==================================================")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Your VLM system is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python launch.py")
        logger.info("2. Choose your preferred interface (Streamlit or CLI)")
        return True
    else:
        logger.error(f"⚠️  {total - passed} TEST(S) FAILED. Please fix the issues above.")
        logger.info("\nTroubleshooting:")
        logger.info("1. Run: ./setup_env.sh to configure authentication")
        logger.info("2. Check camera permissions in System Settings")
        logger.info("3. Ensure Google Cloud billing is enabled")
        logger.info("4. Run: gcloud auth application-default login")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 