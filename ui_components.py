# ====================================================================
# UI COMPONENTS MODULE
# File name: ui_components.py
# ====================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any

class StreamlitUIComponents:
    """Custom UI components for Streamlit interface"""
    
    def load_custom_styles(self):
        """Load custom CSS styles for the application"""
        
        css = """
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        /* Main container */
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 25px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Title styling */
        .main-title {
            background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem !important;
            font-weight: 900;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            line-height: 1.2;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            font-size: 1.3rem;
            color: #555;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Feature cards */
        .feature-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 40%, #4facfe 100%);
            color: white;
            padding: 1.2rem;
            border-radius: 20px;
            margin: 0.8rem;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.5);
        }
        
        /* Chat container */
        .chat-container {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* Message bubbles */
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 5px 20px;
            margin: 0.5rem 0;
            margin-left: 20%;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            font-weight: 500;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 20px 20px 20px 5px;
            margin: 0.5rem 0;
            margin-right: 20%;
            box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);
            font-weight: 500;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
            backdrop-filter: blur(10px);
        }
        
        /* Input styling */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border-radius: 15px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.9);
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        }
        
        /* Metrics styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            margin: 0.5rem;
        }
        
        /* Success/Error alerts */
        .success-alert {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-weight: 500;
        }
        
        .error-alert {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            font-weight: 500;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.5rem !important;
            }
            
            .user-message, .assistant-message {
                margin-left: 5%;
                margin-right: 5%;
            }
            
            .feature-card {
                margin: 0.5rem;
                padding: 1rem;
            }
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
        </style>
        """
        
        st.markdown(css, unsafe_allow_html=True)
    
    def render_hero_section(self):
        """Render the main hero section"""
        
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1 class="main-title">🚀 FAST NUCES AI Assistant</h1>
            <p class="subtitle">
                Next-Generation RAG System with Advanced AI Capabilities
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature cards
        cols = st.columns(4)
        features = [
            ("🧠", "Smart Query Processing"),
            ("🔍", "Advanced Retrieval"),
            ("💬", "Conversational Memory"),
            ("📊", "Real-time Analytics")
        ]
        
        for col, (icon, feature) in zip(cols, features):
            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div>{feature}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_chat_message(self, message: str, is_user: bool = True, timestamp: str = None):
        """Render a chat message bubble"""
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M")
        
        if is_user:
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: flex-end;
                margin: 1rem 0;
            ">
                <div class="user-message">
                    <div style="margin-bottom: 0.5rem;">
                        <strong>You</strong>
                        <span style="opacity: 0.7; font-size: 0.8rem; margin-left: 1rem;">{timestamp}</span>
                    </div>
                    <div>{message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: flex-start;
                margin: 1rem 0;
            ">
                <div class="assistant-message">
                    <div style="margin-bottom: 0.5rem;">
                        <strong>🤖 AI Assistant</strong>
                        <span style="opacity: 0.7; font-size: 0.8rem; margin-left: 1rem;">{timestamp}</span>
                    </div>
                    <div>{message}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    def render_example_grid(self, examples: List[str]) -> str:
        """Render a grid of example questions"""
        
        st.markdown("### 💡 Try these example questions:")
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(
                    example,
                    key=f"example_{i}",
                    use_container_width=True,
                    help="Click to use this example question"
                ):
                    return example
        
        return None

    def render_welcome_message(self):
        """Render welcome message for new users"""
        
        return """
        <div style="
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(240, 147, 251, 0.1) 100%);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            border: 1px solid rgba(102, 126, 234, 0.2);
        ">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎓</div>
            <h2 style="color: #333; margin-bottom: 1rem;">Welcome to FAST NUCES AI Assistant!</h2>
            <p style="
                color: #666;
                font-size: 1.1rem;
                line-height: 1.6;
                max-width: 600px;
                margin: 0 auto;
            ">
                I'm here to help you with all your university-related questions. Whether you need information about 
                FYP requirements, course registration, scholarships, or academic policies, I've got you covered!
            </p>
            <div style="
                margin-top: 2rem;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.7);
                border-radius: 10px;
                font-style: italic;
                color: #555;
            ">
                💡 Try asking me: "What are the FYP requirements?" or "How do I register for courses?"
            </div>
        </div>
        """

    def render_system_info_card(self):
        """Render system information card"""
        
        return """
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
        ">
            <h4 style="color: #333; margin-bottom: 1rem; display: flex; align-items: center;">
                <span style="margin-right: 0.5rem;">🛠️</span>
                System Information
            </h4>
            <div style="color: #666; font-size: 0.9rem; line-height: 1.6;">
                <div><strong>🤖 Language Model:</strong> FLAN-T5 Base</div>
                <div><strong>📚 Embeddings:</strong> MPNet Base v2</div>
                <div><strong>🗄️ Vector DB:</strong> ChromaDB</div>
                <div><strong>🔍 Retrieval:</strong> Multi-query with reranking</div>
                <div><strong>💾 Memory:</strong> Conversation context tracking</div>
                <div><strong>⚡ Performance:</strong> GPU-optimized</div>
            </div>
        </div>
        """

    def render_footer(self):
        """Render the application footer"""
        
        st.markdown("""
        <div style="
            text-align: center;
            margin-top: 4rem;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            color: white;
        ">
            <h3 style="margin-bottom: 1rem;">🚀 Ultimate RAG System - Production Ready</h3>
            <p style="
                font-size: 1.1rem;
                margin: 1rem 0;
                font-weight: 500;
            ">
                <strong>Advanced Features:</strong> Query Intelligence • Document Reranking • Conversational Memory • Real-time Analytics
            </p>
            <p style="
                font-size: 0.9rem;
                opacity: 0.9;
                margin-top: 1rem;
            ">
                Built with Latest GenAI Best Practices | Optimized for Client Demonstrations | Ready for Production Deployment
            </p>
            <div style="
                display: flex;
                justify-content: center;
                gap: 2rem;
                margin-top: 1.5rem;
                font-size: 0.9rem;
            ">
                <span>⚡ Streamlit Powered</span>
                <span>🤖 AI Enhanced</span>
                <span>📊 Analytics Ready</span>
                <span>🚀 Production Ready</span>
            </div>
        </div>
        """, unsafe_allow_html=True)