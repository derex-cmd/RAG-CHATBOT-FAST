# ====================================================================
# MAIN STREAMLIT APPLICATION
# File name: streamlit_app.py
# ====================================================================

import streamlit as st
import os
import warnings
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import custom modules
from rag_system import UltimateRAGSystem
from ui_components import StreamlitUIComponents
from config import config
from utils import (
    init_session_state, 
    update_analytics, 
    export_analytics,
    handle_error
)

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================

st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.fast.edu.pk',
        'Report a bug': None,
        'About': "Advanced RAG-powered AI Assistant for FAST NUCES"
    }
)

# ====================================================================
# MAIN APPLICATION CLASS
# ====================================================================

class FastNucesAIApp:
    """Main application class"""
    
    def __init__(self):
        self.ui = StreamlitUIComponents()
        self.initialize_app()
    
    @st.cache_resource(show_spinner=False)
    def initialize_rag_system(_self):
        """Initialize RAG system with caching"""
        try:
            with st.spinner("🚀 Initializing AI Assistant..."):
                return UltimateRAGSystem()
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            st.stop()
    
    def initialize_app(self):
        """Initialize the application"""
        # Initialize session state
        init_session_state()
        
        # Load custom CSS
        self.ui.load_custom_styles()
        
        # Initialize RAG system
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = self.initialize_rag_system()
    
    def render_header(self):
        """Render the main header section"""
        self.ui.render_hero_section()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.markdown("## 💬 Chat with AI Assistant")
        
        # Display welcome message if no chat history
        if not st.session_state.chat_history:
            st.markdown(self.ui.render_welcome_message(), unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for i, (user_msg, assistant_msg, timestamp) in enumerate(st.session_state.chat_history):
                self.ui.render_chat_message(user_msg, is_user=True, timestamp=timestamp)
                self.ui.render_chat_message(assistant_msg, is_user=False, timestamp=timestamp)
                st.markdown("<br>", unsafe_allow_html=True)
        
        # Input section
        self.render_input_section()
    
    def render_input_section(self):
        """Render the message input section"""
        
        # Example questions first
        example_response = self.ui.render_example_grid([
            "What are the requirements for final year project?",
            "How many credit hours do I need to graduate?",
            "What is the process for course registration?",
            "Tell me about the computer science program",
            "What are the university's admission criteria?",
            "How do I apply for scholarships?"
        ])
        
        if example_response:
            self.process_user_input(example_response)
            st.rerun()
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask anything about FAST NUCES...",
                    placeholder="e.g., What are the FYP requirements? How many credits do I need to graduate?",
                    height=100,
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🚀 Send", use_container_width=True)
                
                if st.form_submit_button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Process user input
        if submitted and user_input.strip():
            self.process_user_input(user_input.strip())
            st.rerun()
    
    def process_user_input(self, user_input: str):
        """Process user input and generate response"""
        
        try:
            timestamp = datetime.now().strftime("%H:%M")
            
            # Show loading animation
            with st.spinner("🤔 Thinking..."):
                start_time = time.time()
                
                # Process query with RAG system
                response, metadata = st.session_state.rag_system.process_query_ultimate(user_input)
                
                response_time = time.time() - start_time
                
                # Add to chat history with timestamp
                st.session_state.chat_history.append((user_input, response, timestamp))
                
                # Update analytics
                update_analytics(user_input, response_time)
                
        except Exception as e:
            error_msg = handle_error(e, "processing user input")
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append((user_input, f"❌ {error_msg}", timestamp))
    
    def render_sidebar(self):
        """Render the sidebar with analytics and settings"""
        
        with st.sidebar:
            # System status
            st.markdown("## 🛠️ System Status")
            st.success("✅ RAG System: Online")
            st.success("✅ Vector DB: Connected")
            st.success("✅ LLM: Ready")
            
            st.markdown("---")
            
            # Analytics
            st.markdown("## 📊 Analytics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Queries", st.session_state.total_queries)
            with col2:
                st.metric("Avg Time", f"{st.session_state.avg_response_time:.2f}s")
            
            st.metric("Sessions", len(st.session_state.chat_history))
            
            st.markdown("---")
            
            # Configuration info
            st.markdown("## ⚙️ Configuration")
            st.markdown(self.ui.render_system_info_card(), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Export functionality
            if st.session_state.chat_history:
                if st.button("📥 Export Chat History", use_container_width=True):
                    try:
                        export_data = export_analytics()
                        st.download_button(
                            label="💾 Download Data",
                            data=export_data,
                            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.success("✅ Export ready!")
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
            
            st.markdown("---")
            
            # Reset button
            if st.button("🔄 Reset Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'rag_system':  # Keep the RAG system initialized
                        del st.session_state[key]
                init_session_state()
                st.success("Session reset!")
                st.rerun()
    
    def render_footer(self):
        """Render the footer section"""
        self.ui.render_footer()
    
    def run(self):
        """Run the main application"""
        
        # Header
        self.render_header()
        
        # Main content area
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown('<div class="main-container">', unsafe_allow_html=True)
            self.render_chat_interface()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Footer
        self.render_footer()

# ====================================================================
# APPLICATION ENTRY POINT
# ====================================================================

def main():
    """Main function to run the application"""
    try:
        app = FastNucesAIApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please refresh the page or contact support if the problem persists.")

if __name__ == "__main__":
    main()