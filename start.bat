@echo off
REM ====================================================================
REM WINDOWS STARTUP SCRIPT
REM File name: start.bat
REM Double-click this file to start the AI Assistant on Windows
REM ====================================================================

title FAST NUCES AI Assistant

echo.
echo ============================================================
echo    🎓 FAST NUCES AI ASSISTANT - WINDOWS LAUNCHER
echo ============================================================
echo    🚀 Starting your AI Assistant...
echo    ⏳ Please wait a moment...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 💡 Please install Python 3.8+ from https://python.org
    echo.
    pause
    exit /b 1
)

REM Check if required files exist
if not exist "streamlit_app.py" (
    echo ❌ streamlit_app.py not found
    echo 💡 Please make sure all files are in the same folder
    echo.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt not found
    echo 💡 Please make sure all files are in the same folder
    echo.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        echo.
        pause
        exit /b 1
    )
)

REM Activate virtual environment and check if Streamlit is installed
call venv\Scripts\activate.bat

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing dependencies...
    echo    This may take a few minutes on first run...
    pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        echo 💡 Check your internet connection and try again
        echo.
        pause
        exit /b 1
    )
    
    REM Download NLTK data
    echo 📚 Downloading language data...
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" >nul 2>&1
)

REM Create directories
if not exist "chroma_db\" mkdir chroma_db
if not exist "exports\" mkdir exports
if not exist "logs\" mkdir logs
if not exist "temp\" mkdir temp

REM Start the application
echo ✅ All ready! Starting AI Assistant...
echo 🌐 Your browser will open in a few seconds...
echo 🌐 Manual URL: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the application
echo.
echo ============================================================

REM Start Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

REM If we get here, the app has stopped
echo.
echo ============================================================
echo 👋 AI Assistant stopped.
echo 💡 To restart, double-click this file again.
echo ============================================================
echo.
pause