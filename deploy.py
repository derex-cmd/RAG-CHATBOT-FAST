#!/usr/bin/env python3
"""
DEPLOYMENT SCRIPT FOR FAST NUCES AI ASSISTANT
File name: deploy.py
Handles setup, dependency installation, and application launch
"""

import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

def run_command(command, description, ignore_errors=False):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if ignore_errors:
            print(f"⚠️ {description} completed with warnings: {e.stderr}")
            return True, e.stderr
        else:
            print(f"❌ {description} failed:")
            print(f"Error: {e.stderr}")
            return False, e.stderr

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Please use Python 3.8+")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🔧 Setting up virtual environment...")
    
    if not os.path.exists("venv"):
        success, output = run_command(f"{sys.executable} -m venv venv", "Creating virtual environment")
        if not success:
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def get_activation_command():
    """Get the correct activation command for the current OS"""
    if os.name == 'nt':  # Windows
        return "venv\\Scripts\\activate"
    else:  # macOS/Linux
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Get the correct pip path
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # macOS/Linux
        pip_path = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip", ignore_errors=True)
    
    # Install dependencies
    success, output = run_command(
        f"{pip_path} install -r requirements.txt",
        "Installing Python dependencies"
    )
    
    return success

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    # Get the correct python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # macOS/Linux
        python_path = "venv/bin/python"
    
    nltk_commands = [
        f'{python_path} -c "import nltk; nltk.download(\'punkt\', quiet=True)"',
        f'{python_path} -c "import nltk; nltk.download(\'stopwords\', quiet=True)"'
    ]
    
    for command in nltk_commands:
        success, output = run_command(command, "Downloading NLTK data", ignore_errors=True)
        if success:
            break
    
    print("✅ NLTK data download completed")
    return True

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = ["chroma_db", "exports", "logs", "temp"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n🔍 Verifying installation...")
    
    # Get the correct python path
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # macOS/Linux
        python_path = "venv/bin/python"
    
    # Test imports
    test_command = f'''
{python_path} -c "
import sys
required_modules = ['streamlit', 'torch', 'transformers', 'langchain', 'chromadb', 'nltk']
missing_modules = []

for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {{module}}')
    except ImportError:
        missing_modules.append(module)
        print(f'❌ {{module}}')

if missing_modules:
    print(f'Missing modules: {{missing_modules}}')
    sys.exit(1)
else:
    print('All required modules are available!')
"
'''
    
    success, output = run_command(test_command, "Verifying dependencies")
    if success:
        print("✅ All required modules are available")
        return True
    else:
        print("❌ Some modules are missing. Please check the installation.")
        return False

def launch_application(port=8501, host="localhost", open_browser=True):
    """Launch the Streamlit application"""
    print(f"\n🚀 Launching application on {host}:{port}...")
    
    # Get the correct streamlit path
    if os.name == 'nt':  # Windows
        streamlit_path = "venv\\Scripts\\streamlit"
    else:  # macOS/Linux
        streamlit_path = "venv/bin/streamlit"
    
    # Check if main app file exists
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py not found. Please make sure the main application file exists.")
        return False
    
    # Build streamlit command
    command_parts = [
        streamlit_path, "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false"
    ]
    
    if not open_browser:
        command_parts.extend(["--server.headless", "true"])
    
    command = " ".join(command_parts)
    
    print(f"📝 Running command: {command}")
    print(f"🌐 Application will be available at: http://{host}:{port}")
    print("🛑 Press Ctrl+C to stop the application")
    print("\n" + "="*60)
    
    try:
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Application failed to start: {e}")
        return False

def show_manual_instructions():
    """Show manual setup instructions"""
    activation_cmd = get_activation_command()
    
    print("\n📋 Manual Setup Instructions:")
    print("=" * 50)
    print("1. Activate virtual environment:")
    print(f"   {activation_cmd}")
    print("\n2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n3. Launch application:")
    print("   streamlit run streamlit_app.py")
    print("\n4. Open your browser to:")
    print("   http://localhost:8501")
    print("=" * 50)

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy FAST NUCES AI Assistant")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on")
    parser.add_argument("--host", default="localhost", help="Host to run the application on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--setup-only", action="store_true", help="Only setup, don't launch")
    parser.add_argument("--skip-venv", action="store_true", help="Skip virtual environment creation")
    
    args = parser.parse_args()
    
    print("🎓 FAST NUCES AI Assistant Deployment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment (unless skipped)
    if not args.skip_venv:
        if not create_virtual_environment():
            print("❌ Failed to create virtual environment")
            show_manual_instructions()
            sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        print("❌ Failed to setup directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        show_manual_instructions()
        sys.exit(1)
    
    # Download NLTK data
    download_nltk_data()
    
    # Verify installation
    if not verify_installation():
        print("❌ Installation verification failed")
        show_manual_instructions()
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    
    # Launch application (unless setup-only)
    if not args.setup_only:
        launch_application(
            port=args.port, 
            host=args.host, 
            open_browser=not args.no_browser
        )
    else:
        print(f"\n🚀 To launch the application manually:")
        activation_cmd = get_activation_command()
        print(f"1. Activate environment: {activation_cmd}")
        print(f"2. Run: streamlit run streamlit_app.py --server.port {args.port}")

if __name__ == "__main__":
    main()