#!/usr/bin/env python3
"""
VLM Application Launcher
Simple menu-driven launcher for the VLM real-time object detection application
"""

import os
import sys
import subprocess
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   VLM Object Detection                       ║
    ║           Real-Time AI Vision with Google Gemini            ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements() -> bool:
    """Check if basic requirements are met"""
    try:
        # Check if we can import basic modules
        import cv2
        import google.generativeai
        import streamlit
        
        # Check if config loads
        from config import config
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def check_api_key() -> bool:
    """Check if Google API key is configured"""
    try:
        from config import config
        if config.GOOGLE_API_KEY:
            logger.info("✅ Google API key is configured")
            return True
        else:
            logger.warning("⚠️  Google API key not set")
            return False
    except Exception:
        return False

def run_command(command: list, description: str) -> bool:
    """Run a command and handle errors"""
    try:
        logger.info(f"🚀 Starting {description}...")
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to run {description}: {e}")
        return False
    except KeyboardInterrupt:
        logger.info(f"\n🛑 {description} interrupted by user")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error running {description}: {e}")
        return False

def launch_streamlit():
    """Launch Streamlit web interface"""
    logger.info("🌐 Launching Streamlit web interface...")
    logger.info("📱 Open http://localhost:8501 in your browser")
    logger.info("🛑 Press Ctrl+C to stop\n")
    
    return run_command(
        ["streamlit", "run", "streamlit_app.py", "--server.headless", "true"],
        "Streamlit web interface"
    )

def launch_command_line():
    """Launch command-line interface"""
    logger.info("💻 Launching command-line interface...")
    logger.info("🛑 Press Ctrl+C to stop\n")
    
    return run_command(
        ["python", "main.py", "--mode", "realtime"],
        "command-line interface"
    )

def launch_camera_preview():
    """Launch camera preview"""
    logger.info("📷 Launching camera preview...")
    logger.info("🛑 Press 'q' to quit, 's' to save snapshot\n")
    
    return run_command(
        ["python", "main.py", "--mode", "preview"],
        "camera preview"
    )

def launch_test_mode():
    """Launch test mode"""
    logger.info("🧪 Running single frame test...")
    
    return run_command(
        ["python", "main.py", "--mode", "test"],
        "test mode"
    )

def run_setup_test():
    """Run setup validation test"""
    logger.info("🔍 Running setup validation...")
    
    return run_command(
        ["python", "test_setup.py"],
        "setup validation"
    )

def setup_api_key():
    """Help user setup API key"""
    print("\n🔑 Google API Key Setup")
    print("=" * 40)
    print("1. Go to: https://aistudio.google.com/")
    print("2. Click 'Get API Key'")
    print("3. Create a new project or select existing")
    print("4. Generate and copy your API key")
    print("5. Set it as environment variable:")
    print("   export GOOGLE_API_KEY='your_api_key_here'")
    print("6. Or create a .env file with:")
    print("   GOOGLE_API_KEY=your_api_key_here")
    print()
    
    api_key = input("Enter your API key now (or press Enter to skip): ").strip()
    
    if api_key:
        # Create .env file
        try:
            with open('.env', 'w') as f:
                f.write(f"GOOGLE_API_KEY={api_key}\n")
            logger.info("✅ API key saved to .env file")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save API key: {e}")
            return False
    else:
        logger.info("Skipped API key setup")
        return False

def show_menu():
    """Show main menu"""
    print("\n📋 Choose an option:")
    print("1. 🌐 Launch Web Interface (Streamlit)")
    print("2. 💻 Launch Command Line Interface")
    print("3. 📷 Camera Preview Only")
    print("4. 🧪 Test Single Frame Detection")
    print("5. 🔍 Run Setup Validation")
    print("6. 🔑 Setup Google API Key")
    print("7. ❓ Show Help")
    print("8. 🚪 Exit")
    print()

def show_help():
    """Show help information"""
    help_text = """
    📚 VLM Object Detection Help
    ═══════════════════════════════════════
    
    🌐 Web Interface: 
       - User-friendly Streamlit interface
       - Real-time video feed with detection results
       - Configuration options in sidebar
       - Analytics and performance tracking
    
    💻 Command Line:
       - Direct command-line execution
       - Configurable via command-line arguments
       - Suitable for automation and scripting
    
    📷 Camera Preview:
       - Test camera connection
       - No AI processing, just video display
       - Press 'q' to quit, 's' for snapshot
    
    🧪 Test Mode:
       - Analyze single frame
       - Test API connection
       - Validate setup
    
    🔍 Setup Validation:
       - Check all dependencies
       - Verify configuration
       - Test camera and API connectivity
    
    Requirements:
    - Python 3.8+
    - Google API key for Gemini
    - Camera access (iPhone via USB or webcam)
    - All dependencies installed (pip install -r requirements.txt)
    """
    print(help_text)

def main():
    """Main launcher function"""
    print_banner()
    
    # Check basic requirements
    if not check_requirements():
        logger.error("❌ Requirements not met. Please install dependencies first.")
        sys.exit(1)
    
    # Check API key
    has_api_key = check_api_key()
    
    if not has_api_key:
        logger.warning("⚠️  Google API key not configured. Some features will not work.")
        logger.info("💡 Use option 6 to setup your API key")
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                launch_streamlit()
            elif choice == '2':
                if not has_api_key:
                    logger.warning("⚠️  API key required for detection. Use option 6 to set it up.")
                    continue
                launch_command_line()
            elif choice == '3':
                launch_camera_preview()
            elif choice == '4':
                if not has_api_key:
                    logger.warning("⚠️  API key required for detection. Use option 6 to set it up.")
                    continue
                launch_test_mode()
            elif choice == '5':
                run_setup_test()
            elif choice == '6':
                if setup_api_key():
                    has_api_key = True
            elif choice == '7':
                show_help()
            elif choice == '8':
                logger.info("👋 Goodbye!")
                sys.exit(0)
            else:
                logger.warning("❌ Invalid choice. Please enter 1-8.")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 