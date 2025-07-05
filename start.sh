#!/bin/bash

# ====================================================================
# LINUX/MACOS STARTUP SCRIPT  
# File name: start.sh
# Run this script to start the AI Assistant on Linux/macOS
# Usage: ./start.sh  or  bash start.sh
# ====================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print banner
echo -e "${PURPLE}"
echo "============================================================"
echo "   🎓 FAST NUCES AI ASSISTANT - UNIX LAUNCHER"  
echo "============================================================"
echo -e "${NC}"
echo -e "${CYAN}🚀 Starting your AI Assistant...${NC}"
echo -e "${CYAN}⏳ Please wait a moment...${NC}"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python is not installed${NC}"
        echo -e "${YELLOW}💡 Please install Python 3.8+ first${NC}"
        echo "   - Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
        echo "   - macOS: brew install python3 (or download from python.org)"
        echo "   - CentOS/RHEL: sudo yum install python3 python3-venv python3-pip"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}❌ Python $PYTHON_VERSION is too old${NC}"
    echo -e "${YELLOW}💡 Please upgrade to Python 3.8 or higher${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check if required files exist
required_files=("streamlit_app.py" "rag_system.py" "ui_components.py" "config.py" "utils.py" "requirements.txt")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required files:${NC}"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo -e "${YELLOW}💡 Please make sure all files are in the current directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required files found${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        echo -e "${YELLOW}💡 Try installing python3-venv: sudo apt install python3-venv${NC}"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    echo -e "${YELLOW}   This may take a few minutes on first run...${NC}"
    
    pip install --upgrade pip > /dev/null 2>&1
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        echo -e "${YELLOW}💡 Check your internet connection and try again${NC}"
        exit 1
    fi
    
    # Download NLTK data
    echo -e "${BLUE}📚 Downloading language data...${NC}"
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" > /dev/null 2>&1
fi

echo -e "${GREEN}✅ Dependencies ready${NC}"

# Create directories
echo -e "${BLUE}📁 Setting up directories...${NC}"
mkdir -p chroma_db exports logs temp

# Check if port is available
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Port $PORT is in use, trying 8502...${NC}"
    PORT=8502
fi

# Start the application
echo -e "${GREEN}✅ All ready! Starting AI Assistant...${NC}"
echo -e "${CYAN}🌐 Application will open at: http://localhost:$PORT${NC}"
echo -e "${CYAN}🛑 Press Ctrl+C to stop the application${NC}"
echo
echo -e "${PURPLE}============================================================${NC}"

# Function to open browser (optional)
open_browser() {
    sleep 3
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:$PORT" &> /dev/null &
    elif command -v open &> /dev/null; then
        open "http://localhost:$PORT" &> /dev/null &
    fi
}

# Start browser opener in background
open_browser &

# Start Streamlit
streamlit run streamlit_app.py \
    --server.port $PORT \
    --server.address localhost \
    --browser.gatherUsageStats false

# If we get here, the app has stopped
echo
echo -e "${PURPLE}============================================================${NC}"
echo -e "${GREEN}👋 AI Assistant stopped.${NC}"
echo -e "${YELLOW}💡 To restart, run: ./start.sh${NC}"
echo -e "${PURPLE}============================================================${NC}"
echo