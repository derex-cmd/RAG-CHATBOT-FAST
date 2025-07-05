#!/usr/bin/env python3
"""
SIMPLE STARTUP SCRIPT
File name: start.py
One-click startup for the FAST NUCES AI Assistant
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("🎓 FAST NUCES AI ASSISTANT - QUICK START")
    print("="*60)
    print("🚀 Starting your AI Assistant...")
    print("⏳ Please wait a moment...\n")

def check_files():
    """Check if required files exist"""
    required_files = [
        "streamlit_app.py",
        "rag_system.py", 
        "ui_components.py",
        "config.py",
        "utils.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📋 Please make sure all files are in the current directory.")
        return False
    
    print("✅ All required files found!")
    return True

def check_virtual_env():
    """Check if virtual environment exists and is activated"""
    venv_exists = os.path.exists("venv")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    return venv_exists, in_venv

def setup_and_run():
    """Setup and run the application"""
    
    # Check if dependencies are installed
    try:
        import streamlit
        print("✅ Streamlit is available")
        dependencies_ok = True
    except ImportError:
        print("⚠️ Streamlit not found - will need to install dependencies")
        dependencies_ok = False
    
    # Check virtual environment
    venv_exists, in_venv = check_virtual_env()
    
    if not dependencies_ok:
        print("📦 Installing dependencies...")
        
        if not venv_exists:
            print("🔧 Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Install dependencies
        if os.name == 'nt':  # Windows
            pip_path = "venv\\Scripts\\pip"
            python_path = "venv\\Scripts\\python"
        else:  # macOS/Linux
            pip_path = "venv/bin/pip"
            python_path = "venv/bin/python"
        
        print("⬇️ Installing packages...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        
        # Download NLTK data
        print("📚 Downloading language data...")
        subprocess.run([
            python_path, "-c", 
            "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
        ], check=False)  # Don't fail if NLTK download fails
    
    # Create directories
    print("📁 Setting up directories...")
    directories = ["chroma_db", "exports", "logs", "temp"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Launch the application
    print("🚀 Launching FAST NUCES AI Assistant...")
    print("🌐 Opening browser in 3 seconds...")
    
    # Determine the correct streamlit command
    if venv_exists and not in_venv:
        if os.name == 'nt':  # Windows
            streamlit_cmd = ["venv\\Scripts\\streamlit"]
        else:  # macOS/Linux
            streamlit_cmd = ["venv/bin/streamlit"]
    else:
        streamlit_cmd = ["streamlit"]
    
    # Build full command
    cmd = streamlit_cmd + [
        "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print("\n" + "="*60)
    print("✅ READY! Your AI Assistant is starting...")
    print("🌐 URL: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Run streamlit
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 AI Assistant stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start: {e}")
        print("💡 Try running: python deploy.py")

def main():
    """Main function"""
    print_banner()
    
    # Check if all files exist
    if not check_files():
        print("\n❌ Setup incomplete. Please check the file list above.")
        input("Press Enter to exit...")
        return
    
    try:
        setup_and_run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 For detailed setup, try: python deploy.py")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()