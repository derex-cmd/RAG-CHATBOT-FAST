#!/usr/bin/env python3
"""
SYSTEM TEST & VERIFICATION SCRIPT
File name: test_system.py
Comprehensive testing of the FAST NUCES AI Assistant
"""

import os
import sys
import subprocess
import importlib
import time
from pathlib import Path

class SystemTester:
    """Comprehensive system testing class"""
    
    def __init__(self):
        self.passed_tests = 0
        self.total_tests = 0
        self.errors = []
    
    def print_header(self):
        """Print test header"""
        print("\n" + "="*60)
        print("🧪 FAST NUCES AI ASSISTANT - SYSTEM TEST")
        print("="*60)
        print("🔍 Running comprehensive system verification...")
        print()
    
    def test(self, test_name, test_func):
        """Run a single test"""
        self.total_tests += 1
        print(f"🔍 Testing: {test_name}...", end=" ")
        
        try:
            result = test_func()
            if result:
                print("✅ PASS")
                self.passed_tests += 1
            else:
                print("❌ FAIL")
                self.errors.append(f"{test_name}: Test function returned False")
        except Exception as e:
            print("❌ ERROR")
            self.errors.append(f"{test_name}: {str(e)}")
    
    def test_python_version(self):
        """Test Python version compatibility"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"    Python {version.major}.{version.minor}.{version.micro}")
            return True
        else:
            self.errors.append(f"Python {version.major}.{version.minor}.{version.micro} is too old. Need 3.8+")
            return False
    
    def test_required_files(self):
        """Test if all required files exist"""
        required_files = [
            "streamlit_app.py",
            "rag_system.py",
            "ui_components.py", 
            "config.py",
            "utils.py",
            "requirements.txt"
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            self.errors.append(f"Missing files: {', '.join(missing)}")
            return False
        
        print(f"    All {len(required_files)} core files present")
        return True
    
    def test_python_imports(self):
        """Test if core Python modules can be imported"""
        modules = ['json', 'os', 'sys', 'datetime', 'typing', 'pathlib']
        
        for module in modules:
            try:
                importlib.import_module(module)
            except ImportError:
                self.errors.append(f"Core Python module '{module}' not available")
                return False
        
        print(f"    All {len(modules)} core Python modules available")
        return True
    
    def test_dependencies(self):
        """Test if required dependencies are installed"""
        required_deps = {
            'streamlit': 'Streamlit web framework',
            'torch': 'PyTorch ML library',
            'transformers': 'Hugging Face transformers',
            'langchain': 'LangChain framework',
            'chromadb': 'ChromaDB vector database',
            'nltk': 'Natural Language Toolkit',
            'pandas': 'Pandas data analysis',
            'numpy': 'NumPy numerical computing'
        }
        
        missing = []
        available = []
        
        for module, description in required_deps.items():
            try:
                importlib.import_module(module)
                available.append(module)
            except ImportError:
                missing.append(f"{module} ({description})")
        
        if missing:
            print(f"    ❌ Missing: {', '.join(missing)}")
            self.errors.append("Run: pip install -r requirements.txt")
            return False
        
        print(f"    All {len(available)} dependencies available")
        return True
    
    def test_gpu_availability(self):
        """Test GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"    GPU Available: {gpu_name}")
            else:
                print("    CPU only (GPU not available)")
            return True
        except ImportError:
            print("    Cannot check GPU (PyTorch not installed)")
            return True  # Don't fail the test for this
    
    def test_directories(self):
        """Test if required directories exist or can be created"""
        directories = ["chroma_db", "exports", "logs", "temp"]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                self.errors.append(f"Cannot create directory '{directory}': {e}")
                return False
        
        print(f"    All {len(directories)} directories ready")
        return True
    
    def test_config_loading(self):
        """Test if configuration can be loaded"""
        try:
            # Add current directory to path to import local modules
            if '.' not in sys.path:
                sys.path.insert(0, '.')
            
            from config import config
            
            # Test a few config values
            assert hasattr(config, 'model')
            assert hasattr(config, 'ui')
            assert hasattr(config, 'database')
            
            print(f"    Configuration loaded successfully")
            return True
        except Exception as e:
            self.errors.append(f"Config loading failed: {e}")
            return False
    
    def test_utils_loading(self):
        """Test if utilities can be loaded"""
        try:
            # Add current directory to path to import local modules
            if '.' not in sys.path:
                sys.path.insert(0, '.')
            
            from utils import init_session_state, get_system_info
            
            # Test a utility function
            info = get_system_info()
            assert isinstance(info, dict)
            
            print(f"    Utilities loaded successfully")
            return True
        except Exception as e:
            self.errors.append(f"Utils loading failed: {e}")
            return False
    
    def test_streamlit_app(self):
        """Test if main Streamlit app can be imported"""
        try:
            # This is a basic import test - we can't fully test Streamlit without running it
            with open('streamlit_app.py', 'r') as f:
                content = f.read()
                
            # Check for key components
            required_components = [
                'import streamlit as st',
                'st.set_page_config',
                'class FastNucesAIApp',
                'def main():'
            ]
            
            for component in required_components:
                if component not in content:
                    self.errors.append(f"Main app missing component: {component}")
                    return False
            
            print(f"    Main application structure valid")
            return True
        except Exception as e:
            self.errors.append(f"Main app validation failed: {e}")
            return False
    
    def test_rag_system(self):
        """Test if RAG system can be imported"""
        try:
            # Add current directory to path to import local modules
            if '.' not in sys.path:
                sys.path.insert(0, '.')
            
            # Import the RAG system class
            from rag_system import UltimateRAGSystem
            
            print(f"    RAG system can be imported")
            return True
        except Exception as e:
            self.errors.append(f"RAG system import failed: {e}")
            return False
    
    def test_nltk_data(self):
        """Test if NLTK data is available"""
        try:
            import nltk
            
            # Try to use NLTK functions that require data
            try:
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                
                # Test with a simple sentence
                test_sentence = "This is a test sentence."
                tokens = word_tokenize(test_sentence)
                stops = set(stopwords.words('english'))
                
                print(f"    NLTK data available and functional")
                return True
            except:
                print(f"    ⚠️ NLTK data missing - will download automatically")
                return True  # Don't fail - the app will download this
                
        except ImportError:
            self.errors.append("NLTK not installed")
            return False
    
    def test_streamlit_command(self):
        """Test if Streamlit command is available"""
        try:
            result = subprocess.run(
                ['streamlit', '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"    Streamlit {version}")
                return True
            else:
                self.errors.append("Streamlit command not working")
                return False
                
        except subprocess.TimeoutExpired:
            self.errors.append("Streamlit command timeout")
            return False
        except FileNotFoundError:
            self.errors.append("Streamlit command not found in PATH")
            return False
        except Exception as e:
            self.errors.append(f"Streamlit command error: {e}")
            return False
    
    def test_port_availability(self, port=8501):
        """Test if the default port is available"""
        import socket
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                print(f"    Port {port} is available")
                return True
        except OSError:
            print(f"    ⚠️ Port {port} is in use - will try alternative")
            return True  # Don't fail - Streamlit can use another port
    
    def run_all_tests(self):
        """Run all tests"""
        self.print_header()
        
        # Core system tests
        self.test("Python Version", self.test_python_version)
        self.test("Required Files", self.test_required_files)
        self.test("Python Imports", self.test_python_imports)
        self.test("Dependencies", self.test_dependencies)
        self.test("GPU Availability", self.test_gpu_availability)
        self.test("Directories", self.test_directories)
        
        # Application tests
        self.test("Configuration", self.test_config_loading)
        self.test("Utilities", self.test_utils_loading)
        self.test("Main Application", self.test_streamlit_app)
        self.test("RAG System", self.test_rag_system)
        self.test("NLTK Data", self.test_nltk_data)
        
        # Runtime tests
        self.test("Streamlit Command", self.test_streamlit_command)
        self.test("Port Availability", self.test_port_availability)
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results"""
        print("\n" + "="*60)
        print("📊 TEST RESULTS")
        print("="*60)
        
        success_rate = (self.passed_tests / self.total_tests) * 100
        
        if success_rate == 100:
            print(f"🎉 ALL TESTS PASSED! ({self.passed_tests}/{self.total_tests})")
            print("✅ Your system is ready to run the AI Assistant!")
            print("\n🚀 To start the application, run:")
            print("   python start.py")
            print("   OR")
            print("   streamlit run streamlit_app.py")
        elif success_rate >= 80:
            print(f"✅ MOSTLY READY! ({self.passed_tests}/{self.total_tests} passed - {success_rate:.1f}%)")
            print("⚠️ Some non-critical issues found:")
            for error in self.errors:
                print(f"   - {error}")
            print("\n🚀 You can still try running the application:")
            print("   python start.py")
        else:
            print(f"❌ NEEDS ATTENTION! ({self.passed_tests}/{self.total_tests} passed - {success_rate:.1f}%)")
            print("🔧 Please fix these issues first:")
            for error in self.errors:
                print(f"   - {error}")
            print("\n💡 Try running the setup script:")
            print("   python deploy.py")
        
        print("="*60)

def main():
    """Main function"""
    tester = SystemTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()