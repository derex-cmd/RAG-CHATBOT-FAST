# ====================================================================
# UTILITIES MODULE
# File name: utils.py
# ====================================================================

import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import base64

def init_session_state():
    """Initialize Streamlit session state variables"""
    
    # Chat history - stores (user_message, assistant_message, timestamp)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Analytics
    if 'total_queries' not in st.session_state:
        st.session_state.total_queries = 0
    
    if 'avg_response_time' not in st.session_state:
        st.session_state.avg_response_time = 0.0
    
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    
    if 'query_topics' not in st.session_state:
        st.session_state.query_topics = {}
    
    # UI state
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # System state
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

def extract_query_topic(query: str) -> str:
    """Extract the main topic from a user query"""
    
    query_lower = query.lower()
    
    topic_keywords = {
        'fyp': 'Final Year Project',
        'final year project': 'Final Year Project',
        'capstone': 'Final Year Project',
        'admission': 'Admissions',
        'admit': 'Admissions',
        'credit': 'Credits/Courses',
        'course': 'Credits/Courses',
        'subject': 'Credits/Courses',
        'fee': 'Fees',
        'tuition': 'Fees',
        'payment': 'Fees',
        'scholarship': 'Scholarships',
        'financial aid': 'Scholarships',
        'registration': 'Registration',
        'enroll': 'Registration',
        'semester': 'Academic Calendar',
        'academic': 'Academic Calendar',
        'exam': 'Examinations',
        'test': 'Examinations',
        'grade': 'Academic Performance',
        'cgpa': 'Academic Performance',
        'gpa': 'Academic Performance',
        'graduation': 'Graduation Requirements',
        'degree': 'Graduation Requirements'
    }
    
    for keyword, topic in topic_keywords.items():
        if keyword in query_lower:
            return topic
    
    return 'General'

def update_analytics(query: str, response_time: float):
    """Update analytics in session state"""
    
    # Update counters
    st.session_state.total_queries += 1
    
    # Update response times
    st.session_state.response_times.append(response_time)
    
    # Calculate average response time
    if st.session_state.response_times:
        st.session_state.avg_response_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
    
    # Update topic tracking
    topic = extract_query_topic(query)
    if topic in st.session_state.query_topics:
        st.session_state.query_topics[topic] += 1
    else:
        st.session_state.query_topics[topic] = 1

def get_analytics_summary() -> Dict[str, Any]:
    """Get comprehensive analytics summary"""
    
    total_queries = st.session_state.total_queries
    avg_time = st.session_state.avg_response_time
    response_times = st.session_state.response_times
    query_topics = st.session_state.query_topics
    
    # Calculate additional metrics
    if response_times:
        min_time = min(response_times)
        max_time = max(response_times)
        recent_avg = sum(response_times[-5:]) / min(len(response_times), 5)
    else:
        min_time = max_time = recent_avg = 0
    
    # Most popular topic
    most_popular_topic = max(query_topics.items(), key=lambda x: x[1]) if query_topics else ("None", 0)
    
    return {
        "total_queries": total_queries,
        "avg_response_time": avg_time,
        "min_response_time": min_time,
        "max_response_time": max_time,
        "recent_avg_time": recent_avg,
        "response_times": response_times,
        "query_topics": query_topics,
        "most_popular_topic": most_popular_topic[0],
        "popular_topic_count": most_popular_topic[1],
        "unique_topics": len(query_topics)
    }

def export_analytics() -> str:
    """Export analytics data as JSON string"""
    
    analytics = get_analytics_summary()
    
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "analytics": analytics,
        "chat_history": st.session_state.chat_history,
        "system_info": get_system_info()
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    
    import platform
    import sys
    
    info = {
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Platform": platform.platform(),
        "Architecture": platform.architecture()[0],
        "Streamlit Version": st.__version__,
    }
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            info["GPU"] = f"CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}"
        else:
            info["GPU"] = "CPU Only"
    except:
        info["GPU"] = "Unknown"
    
    return info

def format_response_time(seconds: float) -> str:
    """Format response time for display"""
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

def save_chat_history(filename: str = None) -> str:
    """Save chat history to a file"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
    
    try:
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        filepath = os.path.join("exports", filename)
        
        chat_data = {
            "export_date": datetime.now().isoformat(),
            "total_conversations": len(st.session_state.chat_history),
            "chat_history": st.session_state.chat_history,
            "analytics": get_analytics_summary()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    except Exception as e:
        raise Exception(f"Failed to save chat history: {str(e)}")

def load_chat_history(filepath: str) -> List:
    """Load chat history from a file"""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('chat_history', [])
    
    except Exception as e:
        raise Exception(f"Failed to load chat history: {str(e)}")

def validate_environment() -> Dict[str, bool]:
    """Validate the environment and dependencies"""
    
    checks = {
        "Python >= 3.8": True,
        "Streamlit installed": True,
        "PyTorch available": False,
        "Transformers available": False,
        "ChromaDB available": False,
        "NLTK available": False
    }
    
    try:
        import torch
        checks["PyTorch available"] = True
    except ImportError:
        pass
    
    try:
        import transformers
        checks["Transformers available"] = True
    except ImportError:
        pass
    
    try:
        import chromadb
        checks["ChromaDB available"] = True
    except ImportError:
        pass
    
    try:
        import nltk
        checks["NLTK available"] = True
    except ImportError:
        pass
    
    return checks

def create_directories():
    """Create necessary directories for the application"""
    
    directories = ["chroma_db", "exports", "logs", "temp"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def handle_error(error: Exception, context: str = "Unknown") -> str:
    """Handle errors gracefully"""
    
    error_msg = f"Error in {context}: {str(error)}"
    print(f"❌ {error_msg}")
    
    return error_msg

def setup_logging():
    """Setup basic logging"""
    
    import logging
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    
    required_modules = [
        'streamlit',
        'torch',
        'transformers',
        'langchain',
        'chromadb',
        'nltk'
    ]
    
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing required modules: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are available")
    return True

def download_nltk_data():
    """Download required NLTK data"""
    
    try:
        import nltk
        print("📚 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"⚠️ NLTK data download failed: {e}")
        return False

def clear_cache():
    """Clear Streamlit cache"""
    
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
        print("🗑️ Cache cleared successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")
        return False

def reset_application():
    """Reset the entire application state"""
    
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != 'rag_system':  # Keep the RAG system initialized
            del st.session_state[key]
    
    # Reinitialize session state
    init_session_state()
    
    # Clear cache
    clear_cache()
    
    print("🔄 Application state reset successfully!")

# Utility functions for file operations
def create_download_link(data: str, filename: str, link_text: str = "Download"):
    """Create a download link for data"""
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def get_file_size(filepath: str) -> str:
    """Get human-readable file size"""
    
    try:
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except:
        return "Unknown"

# Performance monitoring utilities
def measure_time(func):
    """Decorator to measure function execution time"""
    
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️ {func.__name__} executed in {execution_time:.3f} seconds")
        return result
    
    return wrapper

# Example usage and testing
if __name__ == "__main__":
    print("Utils Module Test")
    print("=" * 40)
    
    # Test environment validation
    checks = validate_environment()
    print("Environment validation:")
    for check, status in checks.items():
        print(f"  {check}: {'✅' if status else '❌'}")
    
    # Test directory creation
    if create_directories():
        print("✅ Directory creation successful")
    
    # Test dependency check
    if check_dependencies():
        print("✅ All dependencies available")
    
    # Test NLTK data download
    if download_nltk_data():
        print("✅ NLTK data ready")