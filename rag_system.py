# ====================================================================
# RAG SYSTEM MODULE
# File name: rag_system.py
# ====================================================================

import os
import warnings
import chromadb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import re
import json
from datetime import datetime
import time

# Core ML/AI imports
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, pipeline, AutoModelForSeq2SeqLM
)
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Import configuration
from config import config

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download required NLTK data safely
def safe_nltk_download():
    """Safely download NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")

safe_nltk_download()

# ====================================================================
# QUERY PROCESSING SYSTEM
# ====================================================================

class QueryProcessor:
    """Advanced query processing with expansion and refinement"""

    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])

    def clean_query(self, query: str) -> str:
        """Clean and normalize the input query"""
        query = ' '.join(query.strip().split())

        # Fix common typos
        typo_fixes = {
            'univercity': 'university',
            'projct': 'project',
            'requirment': 'requirement',
            'programm': 'program',
            'studnt': 'student'
        }

        for typo, correction in typo_fixes.items():
            query = query.replace(typo, correction)

        return query

    def expand_query(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        base_query = self.clean_query(query)
        expansions = [base_query]

        # Add domain-specific expansions
        if 'fyp' in base_query.lower():
            expansions.extend([
                base_query.replace('fyp', 'final year project'),
                base_query.replace('fyp', 'capstone project')
            ])

        if 'requirement' in base_query.lower():
            expansions.extend([
                base_query.replace('requirement', 'criteria'),
                base_query.replace('requirement', 'prerequisite')
            ])

        return list(set(expansions))

# ====================================================================
# DOCUMENT RETRIEVAL SYSTEM
# ====================================================================

class AdvancedRetriever:
    """Enhanced document retrieval with reranking"""

    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor()

    def retrieve_documents(self, query: str, k: int = 6) -> List[Document]:
        """Advanced retrieval with query expansion"""
        
        expanded_queries = self.query_processor.expand_query(query)
        all_docs = []
        doc_scores = {}

        # Retrieve for each expanded query
        for expanded_query in expanded_queries[:2]:  # Limit to top 2 expansions
            try:
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
                docs = retriever.get_relevant_documents(expanded_query)

                for doc in docs:
                    doc_id = hash(doc.page_content[:100])
                    if doc_id not in doc_scores:
                        doc_scores[doc_id] = {'doc': doc, 'score': 0, 'count': 0}
                    doc_scores[doc_id]['score'] += self._calculate_relevance_score(doc, query)
                    doc_scores[doc_id]['count'] += 1

            except Exception as e:
                print(f"Retrieval error: {e}")
                continue

        # Rank and return top documents
        ranked_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'] / max(x['count'], 1),
            reverse=True
        )

        return [item['doc'] for item in ranked_docs[:k]]

    def _calculate_relevance_score(self, doc: Document, query: str) -> float:
        """Calculate relevance score for a document"""
        content = doc.page_content.lower()
        query_lower = query.lower()

        # Simple keyword matching
        query_words = set(query_lower.split())
        content_words = set(content.split())
        common_words = query_words.intersection(content_words)
        score = len(common_words) / max(len(query_words), 1)

        return score

# ====================================================================
# PROMPT ENGINEERING SYSTEM
# ====================================================================

class PromptEngineer:
    """Advanced prompt engineering"""

    @staticmethod
    def create_advanced_prompt() -> PromptTemplate:
        """Create an advanced prompt template"""

        template = """You are an intelligent assistant for FAST NUCES (National University of Computer and Emerging Sciences).

CONTEXT INFORMATION:
{context}

STUDENT QUESTION: {question}

INSTRUCTIONS:
- Provide accurate, helpful answers based on the context
- Be specific and detailed when possible
- If information is limited, acknowledge what you know and don't know
- Use a professional, helpful tone suitable for university students
- Structure your response clearly

RESPONSE FORMAT:
**Answer:** [Main answer to the question]

**Details:** [Specific requirements, deadlines, procedures if applicable]

**Additional Info:** [Any extra helpful context]

Please provide a comprehensive answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    @staticmethod
    def create_fallback_prompt() -> PromptTemplate:
        """Fallback prompt when no context is found"""

        template = """You are a helpful assistant for FAST NUCES.

The student asked: "{question}"

I don't have specific information about this in my knowledge base. However, I recommend:

1. Contact your academic advisor for academic matters
2. Visit the registrar office for course-related questions
3. Check the official FAST NUCES website for current information
4. Contact student services for administrative issues

Is there anything else I can help you with?"""

        return PromptTemplate(
            template=template,
            input_variables=["question"]
        )

# ====================================================================
# MODEL MANAGER
# ====================================================================

class ModelManager:
    """Manages language models with optimization"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_llm(self) -> HuggingFacePipeline:
        """Setup optimized language model"""

        try:
            model_name = config.model.llm_model
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )

            # Create pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.model.max_new_tokens,
                temperature=config.model.temperature,
                repetition_penalty=config.model.repetition_penalty,
                device=0 if torch.cuda.is_available() else -1
            )

            return HuggingFacePipeline(pipeline=pipe)

        except Exception as e:
            print(f"Model loading error: {e}")
            return self._setup_simple_fallback()

    def _setup_simple_fallback(self):
        """Simple fallback model"""
        try:
            from transformers import pipeline
            pipe = pipeline("text-generation", model="gpt2", max_new_tokens=200)
            return HuggingFacePipeline(pipeline=pipe)
        except:
            return None

# ====================================================================
# PERFORMANCE MONITORING
# ====================================================================

class PerformanceMonitor:
    """Monitor system performance"""

    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'avg_response_time': 0,
            'popular_topics': {}
        }
        self.response_times = []

    def log_query(self, query: str, response_time: float, success: bool):
        """Log query metrics"""
        self.metrics['total_queries'] += 1

        if success:
            self.metrics['successful_responses'] += 1
        else:
            self.metrics['failed_responses'] += 1

        self.response_times.append(response_time)
        self.metrics['avg_response_time'] = sum(self.response_times) / len(self.response_times)

    def get_analytics_summary(self) -> str:
        """Generate analytics summary"""
        success_rate = (self.metrics['successful_responses'] / max(self.metrics['total_queries'], 1)) * 100
        
        return f"""
📊 **System Analytics**
- Total Queries: {self.metrics['total_queries']}
- Success Rate: {success_rate:.1f}%
- Avg Response Time: {self.metrics['avg_response_time']:.2f}s
"""

# ====================================================================
# CONVERSATION MEMORY
# ====================================================================

class ConversationMemory:
    """Manage conversation context"""

    def __init__(self, max_history: int = 3):
        self.conversation_history = []
        self.max_history = max_history

    def add_exchange(self, query: str, response: str):
        """Add query-response pair to history"""
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response[:200] + "..." if len(response) > 200 else response
        }

        self.conversation_history.append(exchange)

        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_context_for_query(self, current_query: str) -> str:
        """Get relevant context from conversation history"""
        if not self.conversation_history:
            return ""

        context = "Recent conversation:\n"
        for exchange in self.conversation_history[-2:]:
            context += f"Q: {exchange['query']}\nA: {exchange['response']}\n\n"
        
        return context

# ====================================================================
# ULTIMATE RAG SYSTEM
# ====================================================================

class UltimateRAGSystem:
    """Main RAG system with all enhancements"""

    def __init__(self):
        self.setup_components()

    def setup_components(self):
        """Initialize all system components"""

        print("🔧 Setting up RAG System...")

        # Load embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config.model.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"❌ Embedding model error: {e}")
            raise

        # Connect to vector database
        try:
            # Create directory if it doesn't exist
            os.makedirs(config.database.chroma_path, exist_ok=True)
            
            chroma_client = chromadb.PersistentClient(path=config.database.chroma_path)
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=config.database.collection_name,
                embedding_function=self.embedding_model,
            )
            print("✅ Vector database connected")
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            # Create a dummy vectorstore for demo
            self.vectorstore = self._create_demo_vectorstore()

        # Setup language model
        try:
            model_manager = ModelManager()
            self.llm = model_manager.setup_llm()
            print("✅ Language model loaded")
        except Exception as e:
            print(f"❌ LLM error: {e}")
            self.llm = None

        # Initialize components
        self.retriever = AdvancedRetriever(self.vectorstore, self.embedding_model)
        self.prompt_engineer = PromptEngineer()
        self.performance_monitor = PerformanceMonitor()
        self.conversation_memory = ConversationMemory()

        # Setup prompts
        self.main_prompt = self.prompt_engineer.create_advanced_prompt()
        self.fallback_prompt = self.prompt_engineer.create_fallback_prompt()

        print("🎉 RAG System ready!")

    def _create_demo_vectorstore(self):
        """Create a demo vectorstore with sample data"""
        try:
            # Sample university data
            sample_docs = [
                Document(
                    page_content="FAST NUCES offers undergraduate programs in Computer Science, Software Engineering, Electrical Engineering, and Business Administration. The university has campuses in Karachi, Lahore, Islamabad, and Peshawar.",
                    metadata={"page": "1", "source": "university_info"}
                ),
                Document(
                    page_content="Final Year Project (FYP) requirements include: 1) Completion of 120 credit hours, 2) CGPA above 2.0, 3) Approval from FYP committee, 4) Regular meetings with supervisor, 5) Mid and final presentations.",
                    metadata={"page": "2", "source": "fyp_requirements"}
                ),
                Document(
                    page_content="Course registration process: 1) Log into student portal, 2) Select courses for next semester, 3) Check prerequisites, 4) Submit registration form, 5) Pay semester fee, 6) Confirm enrollment.",
                    metadata={"page": "3", "source": "registration_process"}
                ),
                Document(
                    page_content="Graduation requirements: Students must complete 136 credit hours for BS programs, maintain minimum CGPA of 2.0, complete all core and elective courses, and fulfill internship requirements.",
                    metadata={"page": "4", "source": "graduation_requirements"}
                ),
                Document(
                    page_content="Scholarship opportunities include merit-based scholarships for students with CGPA above 3.5, need-based financial aid, and special scholarships for minorities and underprivileged students.",
                    metadata={"page": "5", "source": "scholarships"}
                )
            ]

            # Create temporary vectorstore
            temp_vectorstore = Chroma.from_documents(
                documents=sample_docs,
                embedding=self.embedding_model,
                persist_directory=config.database.chroma_path
            )
            
            print("✅ Demo vectorstore created")
            return temp_vectorstore
            
        except Exception as e:
            print(f"❌ Demo vectorstore creation failed: {e}")
            return None

    def process_query_ultimate(self, query: str, image_path: str = None) -> Tuple[str, dict]:
        """Ultimate query processing"""

        start_time = time.time()

        try:
            if not query or len(query.strip()) < 3:
                return "❓ Please provide a more detailed question.", {"response_time": 0}

            # Get conversation context
            conversation_context = self.conversation_memory.get_context_for_query(query)
            if conversation_context:
                enhanced_query = f"{conversation_context}\nCurrent question: {query}"
            else:
                enhanced_query = query

            # Retrieve relevant documents
            relevant_docs = []
            if self.vectorstore:
                try:
                    relevant_docs = self.retriever.retrieve_documents(enhanced_query, k=config.retrieval.top_k_retrieve)
                except Exception as e:
                    print(f"Retrieval error: {e}")

            # Generate response
            if relevant_docs and self.llm:
                context = self._prepare_context(relevant_docs)
                prompt_input = self.main_prompt.format(context=context, question=query)
                
                try:
                    response = self.llm.invoke(prompt_input)
                    cleaned_response = self._clean_response(response, relevant_docs)
                except Exception as e:
                    print(f"LLM error: {e}")
                    cleaned_response = self._generate_fallback_response(query, relevant_docs)
            else:
                cleaned_response = self._generate_fallback_response(query, relevant_docs)

            # Calculate metrics
            response_time = time.time() - start_time
            confidence_score = self._calculate_confidence_score(query, cleaned_response)

            # Log metrics
            self.performance_monitor.log_query(query, response_time, True)
            self.conversation_memory.add_exchange(query, cleaned_response)

            # Prepare metadata
            metadata = {
                'response_time': response_time,
                'confidence_score': confidence_score,
                'query_length': len(query),
                'response_length': len(cleaned_response)
            }

            return cleaned_response, metadata

        except Exception as e:
            response_time = time.time() - start_time
            self.performance_monitor.log_query(query, response_time, False)

            error_response = f"⚠️ I encountered an issue processing your question. Please try rephrasing it.\n\nError details: {str(e)}"
            metadata = {'error': str(e), 'response_time': response_time}

            return error_response, metadata

    def _prepare_context(self, docs: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        total_length = 0

        for doc in docs:
            content = doc.page_content.strip()
            
            # Add metadata if available
            metadata_str = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                page = doc.metadata.get('page', 'Unknown')
                metadata_str = f"[Page {page}] "

            part = f"{metadata_str}{content}"

            if total_length + len(part) > config.retrieval.max_context_length:
                break

            context_parts.append(part)
            total_length += len(part)

        return "\n\n".join(context_parts)

    def _clean_response(self, response: str, source_docs: List[Document]) -> str:
        """Clean and enhance the model response"""
        
        if isinstance(response, str):
            cleaned = response
        else:
            cleaned = str(response)

        # Clean up formatting
        cleaned = cleaned.strip()
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Add source information
        if source_docs:
            pages = []
            for doc in source_docs[:3]:
                if hasattr(doc, 'metadata') and doc.metadata:
                    page = doc.metadata.get('page', 'Unknown')
                    if page not in pages:
                        pages.append(str(page))

            if pages:
                source_text = f"\n\n📄 **Sources:** Pages {', '.join(pages)}"
                cleaned += source_text

        return cleaned

    def _generate_fallback_response(self, query: str, relevant_docs: List[Document]) -> str:
        """Generate fallback response when LLM is not available"""
        
        if relevant_docs:
            # Use retrieved documents to create response
            response = "Based on the available information:\n\n"
            for i, doc in enumerate(relevant_docs[:2]):
                response += f"• {doc.page_content[:200]}...\n\n"
            
            response += "💡 *This information is extracted from university documents. For more details, please contact the relevant department.*"
            return response
        else:
            # Use fallback prompt
            return self.fallback_prompt.format(question=query)

    def _calculate_confidence_score(self, query: str, response: str) -> float:
        """Calculate confidence score for the response"""
        score = 0.5  # Base score

        # Response length and detail
        if len(response) > 200:
            score += 0.1
        if len(response) > 500:
            score += 0.1

        # Source attribution
        if "Page" in response or "Sources:" in response:
            score += 0.15

        # Specific information indicators
        if any(word in response.lower() for word in ['specific', 'exactly', 'according to', 'requirements']):
            score += 0.1

        return min(score, 1.0)

    def process_query(self, query: str) -> str:
        """Simple query processing for backward compatibility"""
        response, _ = self.process_query_ultimate(query)
        return response