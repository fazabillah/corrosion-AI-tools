import streamlit as st
import os
import re
import hashlib
import json
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load environment variables
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="SmartMCI", 
    layout="wide",
    page_icon="🛡️"
)

# API Documents configuration
API_INDEXES = {
    "api571": "api571-damage-mechanisms",
    "api970": "api970-corrosion-control", 
    "api584": "api584-integrity-windows"
}

# Configuration data for analysis page
EQUIPMENT_TYPES = [
    "Pressure Vessels", "Piping Systems", "Heat Exchangers", 
    "Storage Tanks", "Reactors", "Columns/Towers"
]

MATERIALS = [
    "Carbon Steel", "Stainless Steel 304/316", "Duplex Stainless Steel", 
    "Super Duplex (2507)", "Inconel 625", "Hastelloy C-276"
]

DAMAGE_MECHANISMS = [
    "Pitting Corrosion", "Crevice Corrosion", "Stress Corrosion Cracking",
    "General Corrosion", "Hydrogen Embrittlement", "Fatigue Cracking",
    "High Temperature Corrosion", "Erosion-Corrosion"
]

ENVIRONMENTS = [
    "Marine/Offshore", "Sour Service (H2S)", "High Temperature", 
    "Chloride Environment", "Caustic Service", "Atmospheric"
]

# Calculator-specific materials (simplified for Phase 1)
CALC_MATERIALS = [
    "Carbon Steel", "Stainless Steel 304/316", "Duplex Stainless Steel", 
    "Aluminum", "Copper", "Other"
]

CALC_ENVIRONMENTS = [
    "General Service", "Marine/Offshore", "Sour Service (H2S)", 
    "High Temperature", "Chloride Environment", "Atmospheric"
]

# Example questions for sidebar
SMART_EXAMPLES = [
    "What causes stress corrosion cracking?",
    "How to prevent pitting corrosion?", 
    "Material selection for offshore platforms",
    "Welding considerations for sour service",
    "Process safety in chemical plants"
]

# API 571 Typical Corrosion Rates (mm/year) - Simplified lookup table
API_571_RATES = {
    "Carbon Steel": {
        "General Service": (0.05, 0.3),
        "Marine/Offshore": (0.2, 1.0),
        "Sour Service (H2S)": (0.3, 2.0),
        "High Temperature": (0.5, 3.0),
        "Chloride Environment": (0.3, 1.5),
        "Atmospheric": (0.02, 0.2)
    },
    "Stainless Steel 304/316": {
        "General Service": (0.001, 0.05),
        "Marine/Offshore": (0.01, 0.3),
        "Sour Service (H2S)": (0.05, 0.5),
        "High Temperature": (0.1, 1.0),
        "Chloride Environment": (0.1, 2.0),
        "Atmospheric": (0.001, 0.01)
    },
    "Duplex Stainless Steel": {
        "General Service": (0.001, 0.02),
        "Marine/Offshore": (0.005, 0.1),
        "Sour Service (H2S)": (0.01, 0.2),
        "High Temperature": (0.05, 0.5),
        "Chloride Environment": (0.01, 0.3),
        "Atmospheric": (0.001, 0.005)
    },
    "Aluminum": {
        "General Service": (0.01, 0.1),
        "Marine/Offshore": (0.1, 0.5),
        "Sour Service (H2S)": (0.05, 0.3),
        "High Temperature": (0.2, 1.0),
        "Chloride Environment": (0.2, 1.5),
        "Atmospheric": (0.005, 0.05)
    },
    "Copper": {
        "General Service": (0.005, 0.05),
        "Marine/Offshore": (0.05, 0.3),
        "Sour Service (H2S)": (0.1, 0.8),
        "High Temperature": (0.1, 0.5),
        "Chloride Environment": (0.1, 0.8),
        "Atmospheric": (0.002, 0.02)
    },
    "Other": {
        "General Service": (0.01, 0.5),
        "Marine/Offshore": (0.05, 1.0),
        "Sour Service (H2S)": (0.1, 2.0),
        "High Temperature": (0.2, 3.0),
        "Chloride Environment": (0.1, 2.0),
        "Atmospheric": (0.005, 0.1)
    }
}

# Simple caching system
class SimpleCache:
    def __init__(self, ttl_hours=24):
        self.ttl_hours = ttl_hours
        if "chat_cache" not in st.session_state:
            st.session_state.chat_cache = {}
    
    def _create_key(self, query: str, context: dict = None) -> str:
        """Create cache key from query and context"""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if context:
            context_str = "|".join([f"{k}:{v}" for k, v in context.items() if v and v != "Not Specified"])
            cache_string = f"{normalized}|{context_str}"
        else:
            cache_string = normalized
            
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, context: dict = None):
        """Get cached response if valid"""
        key = self._create_key(query, context)
        if key in st.session_state.chat_cache:
            cache_data = st.session_state.chat_cache[key]
            cache_time = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - cache_time < timedelta(hours=self.ttl_hours):
                return cache_data["response"]
            else:
                del st.session_state.chat_cache[key]
        return None
    
    def set(self, query: str, response: str, context: dict = None):
        """Cache a response"""
        key = self._create_key(query, context)
        st.session_state.chat_cache[key] = {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }

# Calculator Functions
def calculate_corrosion_rate(initial_thickness: float, current_thickness: float, service_years: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate corrosion rate in mm/year"""
    try:
        if service_years <= 0:
            return None, "Service time must be positive"
        
        if current_thickness > initial_thickness:
            return None, "Current thickness cannot exceed initial thickness"
        
        thickness_loss = initial_thickness - current_thickness
        corrosion_rate = thickness_loss / service_years
        
        return corrosion_rate, None
    except Exception as e:
        return None, f"Calculation error: {str(e)}"

def calculate_remaining_life(current_thickness: float, min_thickness: float, corrosion_rate: float) -> Tuple[Optional[float], Optional[str]]:
    """Calculate remaining service life in years"""
    try:
        if corrosion_rate <= 0:
            return None, "No corrosion detected or negative rate"
        
        if current_thickness <= min_thickness:
            return 0, "Equipment already below minimum thickness"
        
        remaining_thickness = current_thickness - min_thickness
        remaining_life = remaining_thickness / corrosion_rate
        
        return max(0, remaining_life), None
    except Exception as e:
        return None, f"Calculation error: {str(e)}"

def validate_corrosion_rate(rate: float, material: str, environment: str) -> Dict[str, str]:
    """Validate corrosion rate against API 571 standards"""
    validation_result = {
        "status": "unknown",
        "message": "",
        "api_reference": ""
    }
    
    try:
        if material in API_571_RATES and environment in API_571_RATES[material]:
            min_rate, max_rate = API_571_RATES[material][environment]
            
            if rate < min_rate:
                validation_result["status"] = "low"
                validation_result["message"] = f"Rate below typical range ({min_rate}-{max_rate} mm/year)"
                validation_result["api_reference"] = "API 571 - Verify measurements"
            elif rate > max_rate:
                validation_result["status"] = "high" 
                validation_result["message"] = f"Rate above typical range ({min_rate}-{max_rate} mm/year)"
                validation_result["api_reference"] = "API 571 - Review service conditions"
            else:
                validation_result["status"] = "normal"
                validation_result["message"] = f"Rate within expected range ({min_rate}-{max_rate} mm/year)"
                validation_result["api_reference"] = "API 571 - Validated"
        else:
            validation_result["status"] = "unknown"
            validation_result["message"] = "No API reference data available"
            validation_result["api_reference"] = "Manual validation required"
            
    except Exception as e:
        validation_result["status"] = "error"
        validation_result["message"] = f"Validation error: {str(e)}"
        
    return validation_result

def create_thickness_trend_chart(initial: float, current: float, years: float, rate: float) -> go.Figure:
    """Create a simple thickness trend chart"""
    try:
        # Generate trend line
        time_points = [0, years, years + 5, years + 10]  # Current + 5 and 10 years projection
        thickness_points = [
            initial,
            current,
            max(0, current - rate * 5),
            max(0, current - rate * 10)
        ]
        
        fig = go.Figure()
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=time_points,
            y=thickness_points,
            mode='lines+markers',
            name='Thickness Trend',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[years],
            y=[current], 
            mode='markers',
            name='Current Measurement',
            marker=dict(color='red', size=12, symbol='diamond')
        ))
        
        fig.update_layout(
            title="Thickness Trend Analysis",
            xaxis_title="Service Time (years)",
            yaxis_title="Thickness (mm)",
            height=400,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Chart generation error: {str(e)}")
        return None

def export_calculation_results(results: Dict) -> str:
    """Generate text export of calculation results"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    export_text = f"""SMARTMCI CORROSION RATE CALCULATION REPORT
Generated: {timestamp}

EQUIPMENT INFORMATION:
{'='*50}
Equipment ID: {results.get('equipment_id', 'Not specified')}
Equipment Type: {results.get('equipment_type', 'Not specified')}
Material: {results.get('material', 'Not specified')}
Service Environment: {results.get('environment', 'Not specified')}

MEASUREMENTS:
{'='*50}
Initial Thickness: {results.get('initial_thickness', 0):.2f} mm
Current Thickness: {results.get('current_thickness', 0):.2f} mm
Service Time: {results.get('service_years', 0):.1f} years
Minimum Required Thickness: {results.get('min_thickness', 0):.2f} mm

CALCULATION RESULTS:
{'='*50}
Corrosion Rate: {results.get('corrosion_rate', 0):.3f} mm/year
Thickness Loss: {results.get('thickness_loss', 0):.2f} mm
Remaining Life: {results.get('remaining_life', 0):.1f} years

VALIDATION:
{'='*50}
API 571 Status: {results.get('validation_status', 'Unknown')}
Validation Message: {results.get('validation_message', 'No validation performed')}
API Reference: {results.get('api_reference', 'Not available')}

RECOMMENDATIONS:
{'='*50}
{results.get('recommendations', 'No specific recommendations available')}

{'='*50}
⚠️ DISCLAIMER: This calculation is for engineering assessment purposes.
Results must be verified by qualified personnel and should not be the
sole basis for critical decisions without additional analysis.

Generated by SmartMCI - Materials, Corrosion & Integrity Assistant
"""
    
    return export_text

# Initialize components (same as before)
@st.cache_resource
def setup_embeddings():
    """Setup embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource  
def setup_vectorstores():
    """Setup vector stores for API documents"""
    if not os.environ.get("PINECONE_API_KEY"):
        st.error("❌ PINECONE_API_KEY not found. Please check your .env file.")
        return {}, False
    
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    embeddings = setup_embeddings()
    vectorstores = {}
    
    for api_name, index_name in API_INDEXES.items():
        try:
            existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
            if index_name in existing_indexes:
                index = pc.Index(index_name)
                stats = index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
                
                if vector_count > 0:
                    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
                    vectorstores[api_name] = vectorstore
                else:
                    st.warning(f"⚠️ {api_name.upper()} index is empty")
            else:
                st.warning(f"⚠️ {api_name.upper()} index not found")
                
        except Exception as e:
            st.error(f"❌ Error connecting to {api_name.upper()}: {e}")
    
    if vectorstores:
        st.success(f"✅ Connected to: {', '.join([f'{k.upper()}' for k in vectorstores.keys()])}")
    else:
        st.error("❌ No API documents found. Please run ingestion script first.")
    
    return vectorstores, len(vectorstores) > 0

@st.cache_resource
def setup_llm():
    """Setup LLM"""
    if not os.environ.get("GROQ_API_KEY"):
        st.error("❌ GROQ_API_KEY not found. Please check your .env file.")
        return None
    
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

def is_mci_related(query: str) -> bool:
    """Check if query is related to core MCI topics"""
    mci_keywords = [
        # Damage mechanisms
        'corrosion', 'cracking', 'damage', 'degradation', 'failure', 'pitting', 
        'crevice', 'stress', 'fatigue', 'erosion', 'embrittlement', 'oxidation',
        # Materials
        'steel', 'stainless', 'carbon', 'alloy', 'metal', 'material', 'duplex',
        'inconel', 'hastelloy', 'aluminum', 'copper', 'titanium',
        # Equipment
        'pipe', 'vessel', 'tank', 'exchanger', 'reactor', 'equipment', 'piping',
        'pressure', 'storage', 'column', 'tower', 'pipeline',
        # Environment/conditions
        'temperature', 'chloride', 'sour', 'h2s', 'caustic', 'marine', 'offshore',
        'high temp', 'environment', 'service', 'operating', 'process',
        # API standards
        'api 571', 'api 970', 'api 584', 'api571', 'api970', 'api584',
        # Prevention/control
        'mitigation', 'prevention', 'protection', 'inhibitor', 'coating',
        'cathodic', 'inspection', 'monitoring', 'integrity', 'limits'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in mci_keywords)

def is_engineering_related(query: str) -> bool:
    """Check if query is broadly engineering-related (more permissive than strict MCI)"""
    engineering_keywords = [
        # Core MCI terms
        'corrosion', 'cracking', 'damage', 'degradation', 'failure', 'pitting', 
        'crevice', 'stress', 'fatigue', 'erosion', 'embrittlement', 'oxidation',
        'steel', 'stainless', 'carbon', 'alloy', 'metal', 'material', 'duplex',
        'inconel', 'hastelloy', 'aluminum', 'copper', 'titanium',
        'pipe', 'vessel', 'tank', 'exchanger', 'reactor', 'equipment', 'piping',
        'pressure', 'storage', 'column', 'tower', 'pipeline',
        'temperature', 'chloride', 'sour', 'h2s', 'caustic', 'marine', 'offshore',
        'api 571', 'api 970', 'api 584', 'api571', 'api970', 'api584',
        'mitigation', 'prevention', 'protection', 'inhibitor', 'coating',
        'cathodic', 'inspection', 'monitoring', 'integrity', 'limits',
        
        # Broader engineering terms
        'engineering', 'design', 'safety', 'standards', 'specification',
        'welding', 'fabrication', 'construction', 'maintenance', 'testing',
        'quality', 'reliability', 'performance', 'optimization', 'efficiency',
        'process', 'manufacturing', 'industrial', 'chemical', 'mechanical',
        'thermal', 'structural', 'analysis', 'modeling', 'simulation'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in engineering_keywords)

def retrieve_documents(query: str, vectorstores: dict) -> List:
    """Retrieve relevant documents from all API sources"""
    all_docs = []
    
    for api_name, vectorstore in vectorstores.items():
        try:
            retriever = vectorstore.as_retriever(k=3)
            docs = retriever.invoke(query)
            all_docs.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Error retrieving from {api_name.upper()}: {e}")
    
    return all_docs

def format_documents(docs) -> str:
    """Format retrieved documents for the prompt"""
    if not docs:
        return "No relevant API documentation found."
    
    formatted_docs = []
    for i, doc in enumerate(docs):
        api_standard = doc.metadata.get('api_standard', 'Unknown API')
        page_num = doc.metadata.get('page', 'Unknown Page')
        
        formatted_docs.append(
            f"[Source {i+1}] {api_standard} - Page {page_num}\n"
            f"Content: {doc.page_content}\n"
        )
    
    return "\n".join(formatted_docs)

def get_conversation_context(chat_history: List[dict], max_messages: int = 4) -> str:
    """Get recent conversation context"""
    if len(chat_history) <= 1:
        return ""
    
    recent_messages = chat_history[-(max_messages*2):]  # Get recent Q&A pairs
    context_parts = []
    
    for msg in recent_messages:
        if msg["role"] == "user":
            context_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            # Include only first 150 characters of response for context
            summary = msg['content'][:150] + "..." if len(msg['content']) > 150 else msg['content']
            context_parts.append(f"Assistant: {summary}")
    
    return "\n".join(context_parts) if context_parts else ""

def search_web_tavily(query: str, max_results: int = 5, is_mci: bool = True) -> str:
    """Search web using Tavily API for MCI and engineering topics"""
    
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Web search unavailable - TAVILY_API_KEY not configured."
    
    try:
        # Focus search on engineering/MCI topics
        if is_mci:
            focused_query = f"materials corrosion integrity engineering {query}"
        else:
            focused_query = f"engineering industrial {query}"
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": tavily_api_key,
            "query": focused_query,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "max_results": max_results,
            "include_domains": [
                "asme.org", "api.org", "nace.org", "astm.org", 
                "engineeringtoolbox.com", "corrosionpedia.com",
                "sciencedirect.com", "springer.com", "onepetro.org",
                "ieee.org", "aiche.org", "isa.org"
            ]
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            web_content = []
            
            # Add Tavily's answer if available
            if data.get("answer"):
                web_content.append(f"Web Search Summary: {data['answer']}")
            
            # Add search results
            results = data.get("results", [])
            for i, result in enumerate(results[:max_results]):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                
                web_content.append(
                    f"[Web Source {i+1}] {title}\n"
                    f"URL: {url}\n"
                    f"Content: {content[:500]}...\n"
                )
            
            return "\n".join(web_content) if web_content else "No relevant web results found."
            
        else:
            return f"Web search error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Web search failed: {str(e)}"

def assess_content_sufficiency(docs: List, query: str) -> Tuple[bool, str]:
    """Assess if retrieved API documents are sufficient to answer the query"""
    
    if not docs:
        return False, "No relevant API documentation found."
    
    # Simple heuristics to determine if content is sufficient
    total_content_length = sum(len(doc.page_content) for doc in docs)
    
    # Check for key query terms in retrieved content
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    doc_content = " ".join([doc.page_content.lower() for doc in docs])
    doc_terms = set(re.findall(r'\b\w+\b', doc_content))
    
    term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
    
    # Consider content sufficient if:
    # 1. We have substantial content AND good term overlap
    # 2. OR we have very relevant but shorter content
    is_sufficient = (
        (total_content_length > 500 and term_overlap > 0.3) or
        (total_content_length > 200 and term_overlap > 0.5)
    )
    
    if is_sufficient:
        return True, "API documentation appears sufficient."
    else:
        return False, f"API documentation limited (content: {total_content_length} chars, term overlap: {term_overlap:.2f})"

def create_polite_redirect(query: str) -> str:
    """Create a helpful redirect for non-engineering topics"""
    return f"""I'm SmartMCI, specialized in Materials, Corrosion, and Integrity (MCI) engineering. While I'd love to help with "{query}", I'm designed to focus on engineering topics, particularly:

**🛡️ Core Expertise:**
- **Damage Mechanisms** (API 571) - Failure analysis, root causes
- **Corrosion Control** (API 970) - Prevention, mitigation strategies  
- **Integrity Management** (API 584) - Operating limits, safe windows

**🔧 Related Engineering Topics:**
- Materials selection and behavior
- Process safety and reliability
- Equipment design and maintenance
- Industrial standards and codes

**Try asking:**
- "What causes stress corrosion cracking in chloride environments?"
- "How to select materials for high temperature service?"
- "Operating limits for sour gas systems?"
- "Fatigue failure mechanisms in pressure vessels?"

I'm here to help make your engineering projects safer and more reliable! 🚀"""

def create_hybrid_chat_prompt():
    """Create prompt template optimized for MCI focus"""
    return PromptTemplate(
        input_variables=["api_context", "web_context", "conversation_history", "query", "search_used"],
        template="""You are SmartMCI, a specialized assistant for Materials, Corrosion, and Integrity (MCI) engineering with expertise in API 571, API 970, and API 584 standards.

IMPORTANT UNIT REQUIREMENTS:
- ALWAYS use metric SI units
- Convert imperial units if needed: °C = (°F - 32) × 5/9, bar = psi × 0.0689476

API Documentation:
{api_context}

{search_used}Web Search Results:
{web_context}

Conversation History:
{conversation_history}

Current Question: {query}

RESPONSE GUIDELINES:
1. Provide DIRECT, HELPFUL answers without preambles
2. Do NOT mention "based on API documentation" or "web search results" 
3. Start immediately with the information
4. Use metric SI units consistently
5. Reference API standards naturally: "API 571 states..." or "Per API 970..."
6. Maintain conversational flow and acknowledge previous context
7. If you have information from any source, present it confidently
8. Be technically accurate and specific
9. For engineering-related general questions, provide helpful context while steering toward MCI relevance

Response:"""
    )

def create_analysis_prompt():
    """Create structured analysis prompt template"""
    return PromptTemplate(
        input_variables=["context", "equipment_context", "query"],
        template="""You are an expert MCI (Materials, Corrosion, and Integrity) engineering consultant providing structured analysis.

Equipment Context:
{equipment_context}

IMPORTANT UNIT REQUIREMENTS:
- ALWAYS use metric SI units in your responses
- Temperature: Use degrees Celsius (°C) ONLY
- Pressure: Use bar ONLY (not psi, kPa, or MPa)
- If source documents mention Fahrenheit (°F) or psi, convert to °C and bar
- Common conversions: °C = (°F - 32) × 5/9, bar = psi × 0.0689476

API Documentation Available:
{context}

Analysis Request: {query}

Provide a comprehensive structured analysis covering:

## 1. DAMAGE MECHANISMS
- Specific conditions that cause damage
- Environmental factors and thresholds
- Material susceptibility factors
- Critical parameters (temperature in °C, pressure in bar)

## 2. MITIGATION STRATEGIES
- Material selection recommendations
- Environmental control measures
- Protective systems and coatings
- Design modifications
- Process optimization strategies

## 3. OPERATING LIMITS
- Safe operating windows (temperature in °C, pressure in bar)
- Critical control points and alarm settings
- Monitoring requirements
- Inspection frequencies
- Deviation consequences

## 4. SPECIFIC RECOMMENDATIONS
- Context-specific guidance based on equipment and environment
- Risk assessment considerations
- Implementation priorities

Express all temperatures in °C and pressures in bar. Be thorough but concise in each section.

Analysis:"""
    )

def generate_response(query: str, vectorstores: dict, llm, chat_history: List[dict] = None, equipment_context: dict = None) -> str:
    """Generate response with MCI focus and smart boundaries"""
    try:
        # Check if query is MCI or engineering-related
        is_mci = is_mci_related(query)
        is_engineering = is_engineering_related(query)
        
        # Always retrieve API documents for engineering questions
        docs = retrieve_documents(query, vectorstores) if is_engineering else []
        api_context = format_documents(docs)
        
        web_context = ""
        search_used = ""
        
        if is_mci:
            # Core MCI topics - full treatment
            is_sufficient, assessment_msg = assess_content_sufficiency(docs, query)
            if not is_sufficient:
                st.info(f"🔍 Searching for additional MCI information...")
                web_context = search_web_tavily(query, is_mci=True)
                search_used = "✅ "
        elif is_engineering:
            # Engineering-related but not core MCI - web search with MCI context
            st.info("🔍 Searching for engineering information...")
            web_context = search_web_tavily(query, is_mci=True)  # Still use MCI-focused search
            search_used = "✅ "
        else:
            # Non-engineering topics - polite redirect
            return create_polite_redirect(query)
        
        if equipment_context:
            # Structured analysis mode
            context_parts = []
            for key, value in equipment_context.items():
                if value and value != "Not Specified":
                    context_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            context_string = " | ".join(context_parts) if context_parts else "General analysis"
            
            prompt_template = create_analysis_prompt()
            formatted_prompt = prompt_template.format(
                context=api_context,
                equipment_context=context_string,
                query=query
            )
        else:
            # Chat mode
            conversation_context = get_conversation_context(chat_history or [])
            
            prompt_template = create_hybrid_chat_prompt()
            formatted_prompt = prompt_template.format(
                api_context=api_context,
                web_context=web_context,
                conversation_history=conversation_context,
                query=query,
                search_used=search_used
            )
        
        # Generate response
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        
        # Add helpful footer for non-core MCI topics
        if is_engineering and not is_mci:
            result += "\n\n💡 *For specific materials, corrosion, or integrity challenges, feel free to ask about API 571/970/584 standards.*"
        elif web_context and search_used:
            result += "\n\n*📡 Enhanced with web search results*"
        
        return result
            
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again."

def calculator_page():
    """Corrosion Rate Calculator page implementation - Phase 1"""
    st.title("🧮 Corrosion Rate Calculator")
    st.markdown("**Calculate corrosion rates and remaining equipment life**")
    
    # Initialize session state for calculator
    if "calc_results" not in st.session_state:
        st.session_state.calc_results = {}
    
    # Main calculator interface
    st.markdown("### 📏 **Thickness Measurements**")
    
    # Core input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_thickness = st.number_input(
            "Initial Thickness (mm)",
            min_value=0.1,
            max_value=500.0,
            value=10.0,
            step=0.1,
            help="Original design thickness or first measurement"
        )
    
    with col2:
        current_thickness = st.number_input(
            "Current Thickness (mm)",
            min_value=0.1,
            max_value=500.0,
            value=9.5,
            step=0.1,
            help="Latest thickness measurement"
        )
    
    with col3:
        service_years = st.number_input(
            "Service Time (years)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.1,
            help="Time between initial and current measurements"
        )
    
    # Additional equipment information
    st.markdown("### 🏭 **Equipment Information** (Optional)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        equipment_id = st.text_input(
            "Equipment ID",
            placeholder="e.g., V-101, P-205A",
            help="Unique identifier for your equipment"
        )
        
        equipment_type = st.selectbox(
            "Equipment Type",
            ["Not Specified"] + EQUIPMENT_TYPES,
            help="Select equipment type for better validation"
        )
    
    with col2:
        material = st.selectbox(
            "Material",
            ["Not Specified"] + CALC_MATERIALS,
            help="Select material for API cross-reference"
        )
        
        environment = st.selectbox(
            "Service Environment",
            ["Not Specified"] + CALC_ENVIRONMENTS,
            help="Select environment for validation"
        )
    
    # Minimum thickness input
    st.markdown("### ⚖️ **Safety Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_thickness = st.number_input(
            "Minimum Required Thickness (mm)",
            min_value=0.1,
            max_value=initial_thickness if initial_thickness else 500.0,
            value=initial_thickness * 0.7 if initial_thickness else 7.0,
            step=0.1,
            help="Minimum thickness for safe operation"
        )
    
    with col2:
        # Show calculated corrosion allowance
        if initial_thickness and min_thickness:
            corrosion_allowance = initial_thickness - min_thickness
            st.metric("Corrosion Allowance", f"{corrosion_allowance:.2f} mm")
    
    # Real-time validation and warnings
    validation_messages = []
    
    if current_thickness > initial_thickness:
        st.error("❌ Current thickness cannot exceed initial thickness")
        validation_messages.append("Invalid thickness measurements")
    
    if min_thickness and current_thickness <= min_thickness:
        st.error("🚨 CRITICAL: Equipment is below minimum required thickness!")
        validation_messages.append("Below minimum thickness")
    elif min_thickness and current_thickness < min_thickness * 1.1:
        st.warning("⚠️ WARNING: Approaching minimum thickness")
        validation_messages.append("Near minimum thickness")
    
    # Perform calculations
    if initial_thickness and current_thickness and service_years and not validation_messages:
        
        # Calculate corrosion rate
        corrosion_rate, rate_error = calculate_corrosion_rate(initial_thickness, current_thickness, service_years)
        
        # Initialize validation variable
        validation = {
            'status': 'not_validated',
            'message': 'No validation performed',
            'api_reference': 'Not available'
        }
        
        if rate_error:
            st.error(f"Calculation Error: {rate_error}")
        else:
            # Calculate remaining life
            remaining_life, life_error = calculate_remaining_life(current_thickness, min_thickness, corrosion_rate)
            
            # Display main results
            st.markdown("---")
            st.markdown("## 📊 **Calculation Results**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Corrosion Rate", 
                    f"{corrosion_rate:.3f} mm/year",
                    help="Rate of thickness loss per year"
                )
            
            with col2:
                thickness_loss = initial_thickness - current_thickness
                st.metric(
                    "Total Thickness Loss",
                    f"{thickness_loss:.2f} mm",
                    help="Total thickness lost during service time"
                )
            
            with col3:
                if not life_error and remaining_life is not None:
                    if remaining_life > 100:
                        life_display = "> 100 years"
                    else:
                        life_display = f"{remaining_life:.1f} years"
                    
                    st.metric(
                        "Remaining Life",
                        life_display,
                        help="Estimated years until minimum thickness"
                    )
                else:
                    st.metric("Remaining Life", "N/A", help=life_error or "Cannot calculate")
            
            # API validation if material and environment specified
            if material != "Not Specified" and environment != "Not Specified":
                st.markdown("### 📚 **API 571 Validation**")
                
                # Update validation with actual results
                validation = validate_corrosion_rate(corrosion_rate, material, environment)
                
                if validation["status"] == "normal":
                    st.success(f"✅ {validation['message']}")
                elif validation["status"] == "high":
                    st.warning(f"⚠️ {validation['message']}")
                elif validation["status"] == "low":
                    st.info(f"ℹ️ {validation['message']}")
                else:
                    st.info(f"ℹ️ {validation['message']}")
                
                if validation["api_reference"]:
                    st.caption(f"Reference: {validation['api_reference']}")
            
            # Generate chart
            if st.checkbox("📈 Show Thickness Trend Chart"):
                chart = create_thickness_trend_chart(initial_thickness, current_thickness, service_years, corrosion_rate)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 **Recommendations**")
            
            recommendations = []
            
            if corrosion_rate > 1.0:
                recommendations.append("• **High corrosion rate detected** - Review service conditions and consider mitigation")
                recommendations.append("• **Increase inspection frequency** - Monitor more closely")
                recommendations.append("• **Consider material upgrade** - Evaluate more corrosion-resistant options")
            elif corrosion_rate > 0.5:
                recommendations.append("• **Moderate corrosion rate** - Continue regular monitoring")
                recommendations.append("• **Review operating conditions** - Ensure within design limits")
            else:
                recommendations.append("• **Good performance** - Current corrosion rate is acceptable")
                recommendations.append("• **Continue standard inspection intervals**")
            
            if remaining_life is not None and remaining_life < 5:
                recommendations.append("• **Plan for replacement** - Equipment approaching end of life")
                recommendations.append("• **Increase inspection frequency** - Monitor remaining thickness closely")
            elif remaining_life is not None and remaining_life < 10:
                recommendations.append("• **Begin replacement planning** - Consider long-term maintenance strategy")
            
            for rec in recommendations:
                st.markdown(rec)
            
            # Store results for export
            st.session_state.calc_results = {
                'equipment_id': equipment_id or 'Not specified',
                'equipment_type': equipment_type,
                'material': material,
                'environment': environment,
                'initial_thickness': initial_thickness,
                'current_thickness': current_thickness,
                'service_years': service_years,
                'min_thickness': min_thickness,
                'corrosion_rate': corrosion_rate,
                'thickness_loss': thickness_loss,
                'remaining_life': remaining_life if remaining_life is not None else 'N/A',
                'validation_status': validation.get('status', 'not_validated'),
                'validation_message': validation.get('message', 'No validation performed'),
                'api_reference': validation.get('api_reference', 'Not available'),
                'recommendations': '\n'.join([rec.replace('• ', '- ') for rec in recommendations])
            }
            
            # Export options
            st.markdown("---")
            st.markdown("### 📄 **Export Results**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📄 Generate Report", type="primary"):
                    report_text = export_calculation_results(st.session_state.calc_results)
                    st.download_button(
                        "📄 Download Report",
                        report_text,
                        file_name=f"corrosion_calc_{equipment_id or 'equipment'}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                # Quick summary export
                summary = f"""Equipment: {equipment_id or 'Not specified'}
Corrosion Rate: {corrosion_rate:.3f} mm/year
Remaining Life: {remaining_life:.1f} years if remaining_life is not None else 'N/A'
Status: {validation.get('status', 'not_validated').title()}"""
                
                st.download_button(
                    "📋 Quick Summary",
                    summary,
                    file_name=f"summary_{equipment_id or 'equipment'}.txt",
                    mime="text/plain"
                )
            
            with col3:
                # CSV export for data analysis
                if st.button("📊 CSV Data"):
                    csv_data = f"""Parameter,Value,Unit
Equipment ID,{equipment_id or 'Not specified'},
Equipment Type,{equipment_type},
Material,{material},
Environment,{environment},
Initial Thickness,{initial_thickness},mm
Current Thickness,{current_thickness},mm
Service Time,{service_years},years
Minimum Thickness,{min_thickness},mm
Corrosion Rate,{corrosion_rate:.3f},mm/year
Thickness Loss,{thickness_loss:.2f},mm
Remaining Life,{remaining_life if remaining_life is not None else 'N/A'},years
"""
                    st.download_button(
                        "📊 Download CSV",
                        csv_data,
                        file_name=f"corrosion_data_{equipment_id or 'equipment'}.csv",
                        mime="text/csv"
                    )
    
    else:
        # Welcome message when no calculations performed
        if not (initial_thickness and current_thickness and service_years):
            st.markdown("""
            This tool helps you calculate corrosion rates and predict remaining equipment life based on thickness measurements.
            
            **🎯 What you get:**
            - **Corrosion Rate**: mm/year based on your measurements
            - **Remaining Life**: Years until minimum thickness
            - **API 571 Validation**: Cross-check against industry standards
            - **Recommendations**: Actionable guidance for your equipment
            
            **📋 To get started:**
            1. Enter your **thickness measurements** (initial, current, service time)
            2. Optionally specify **equipment details** for better validation
            3. Set **minimum required thickness** for safety assessment
            4. Review **results and recommendations**
            5. **Export** your analysis report
            
            **📏 Units**: All calculations use metric units (mm, years)
            
            **Try it now** - enter your thickness measurements above!
            """)

def chatbot_page():
    """Chatbot page implementation with optimized MCI focus"""
    st.title("🛡️ SmartMCI ChatBot")
    st.markdown("**Expert MCI Engineering Consultant**")
    
    # Initialize session state for chatbot
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing SmartMCI..."):
        vectorstores, available = setup_vectorstores()
        llm = setup_llm()
        
        if not available or not llm:
            st.error("❌ System initialization failed. Please check configuration.")
            st.stop()
        
        # Check Tavily API key
        tavily_available = bool(os.environ.get("TAVILY_API_KEY"))
        if tavily_available:
            st.success("🌐 Web search enabled via Tavily")
        else:
            st.warning("⚠️ Web search disabled - TAVILY_API_KEY not found")
    
    # Sidebar with quick actions
    with st.sidebar:
        st.header("Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        for example in SMART_EXAMPLES:
            if st.button(example, key=f"chat_ex_{hash(example)}", use_container_width=True):
                # Add user message
                st.session_state.chat_messages.append({"role": "user", "content": example})
                
                # Check cache first
                cached_response = cache.get(example)
                if cached_response:
                    response = cached_response + "\n\n*[Cached response]*"
                else:
                    # Generate response with smart boundaries
                    response = generate_response(example, vectorstores, llm, st.session_state.chat_messages)
                    cache.set(example, response)
                
                # Add assistant response
                st.session_state.chat_messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Show cache stats if available
        if st.session_state.get("chat_cache"):
            st.markdown("---")
            st.caption(f"💾 Cached responses: {len(st.session_state.chat_cache)}")
    
    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input with smart boundaries
    if prompt := st.chat_input("Ask about materials, corrosion, integrity, or engineering..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Check cache first
        cached_response = cache.get(prompt)
        if cached_response:
            response = cached_response + "\n\n*[Cached response]*"
        else:
            # Generate response with smart boundaries
            with st.spinner("Analyzing..."):
                response = generate_response(prompt, vectorstores, llm, st.session_state.chat_messages)
                cache.set(prompt, response)
        
        # Add assistant response
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()
    
    # Welcome message for new users
    if not st.session_state.chat_messages:
        st.markdown("""
        I'm your consultant for **Materials, Corrosion & Integrity** engineering with basis from American Petroleum Institute (API):

        **🛡️ Database:**
        - **API 571** - Damage Mechanisms & Failure Analysis
        - **API 970** - Corrosion Control & Prevention  
        - **API 584** - Integrity Operating Windows

        **App Features (Refer on Sidebar):**
        - 💬 **ChatBot**: Conversational AI for MCI engineering questions
        - 🧮 **Calculator**: Corrosion rate calculations (mm/year), Remaining Life, and API validation
        - 🔬 **Analysis**: Comprehensive damage mechanism assessment

        **Try asking:** *"What are the key factors in material selection for offshore platforms?"*
        """)

def analysis_page():
    """Structured analysis page implementation"""
    st.title("🔬 SmartMCI Analysis")
    st.markdown("**Structured Analysis with Equipment Parameters**")
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing analysis tools..."):
        vectorstores, available = setup_vectorstores()
        llm = setup_llm()
        
        if not available or not llm:
            st.error("❌ System initialization failed. Please check configuration.")
            st.stop()
    
    # Check if we should show results or input form
    show_results = st.session_state.get("analysis_result") is not None
    
    if not show_results:
        # Welcome message at the top
        st.markdown("""
        This page provides **structured analysis** based on your specific equipment parameters.
        
        **📋 Analysis covers:**
        - **Damage Mechanisms** (API 571) - Conditions and causes
        - **Mitigation Strategies** (API 970) - Prevention methods
        - **Operating Limits** (API 584) - Safe parameters
        - **Specific Recommendations** - Context-based guidance
        
        **🎯 Configure your equipment parameters below to get started!**
        """)
        
        st.markdown("---")
        
        # Input form on main page
        st.markdown("## 📋 Equipment Parameters")
        st.markdown("Configure your equipment details for comprehensive MCI analysis")
        
        # Equipment Information Section
        with st.expander("🏭 **Equipment Information**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                equipment_type = st.selectbox(
                    "Equipment Type:", 
                    ["Not Specified"] + EQUIPMENT_TYPES,
                    help="Select equipment for specific guidance"
                )
                
                material = st.selectbox(
                    "Material:", 
                    ["Not Specified"] + MATERIALS,
                    help="Select material for specific recommendations"
                )
            
            with col2:
                environment = st.selectbox(
                    "Service Environment:", 
                    ["Not Specified"] + ENVIRONMENTS,
                    help="Select environment for specific analysis"
                )
                
                damage_mechanism = st.selectbox(
                    "Damage Type:", 
                    ["Not Specified"] + DAMAGE_MECHANISMS,
                    help="Select damage type for specific information"
                )
        
        # Operating Conditions Section
        with st.expander("🌡️ **Operating Conditions**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.number_input(
                    "Temperature (°C)", 
                    value=None, 
                    min_value=-50, 
                    max_value=1000, 
                    placeholder="Optional",
                    help="Operating temperature in Celsius"
                )
            
            with col2:
                pressure = st.number_input(
                    "Pressure (bar)", 
                    value=None, 
                    min_value=0, 
                    max_value=500, 
                    placeholder="Optional",
                    help="Operating pressure in bar"
                )
        
        # Analysis Configuration Section
        with st.expander("🔍 **Analysis Configuration**", expanded=True):
            analysis_focus = st.selectbox(
                "Focus Area:",
                [
                    "Comprehensive Analysis",
                    "Damage Mechanisms Only",
                    "Mitigation Strategies Only", 
                    "Operating Limits Only"
                ],
                help="Choose the scope of your analysis"
            )
            
            # Show current parameter summary
            st.markdown("### 📊 **Parameter Summary**")
            
            # Create summary of selected parameters
            selected_params = []
            if equipment_type != "Not Specified":
                selected_params.append(f"**Equipment:** {equipment_type}")
            if material != "Not Specified":
                selected_params.append(f"**Material:** {material}")
            if environment != "Not Specified":
                selected_params.append(f"**Environment:** {environment}")
            if damage_mechanism != "Not Specified":
                selected_params.append(f"**Damage Type:** {damage_mechanism}")
            if temperature is not None:
                selected_params.append(f"**Temperature:** {temperature}°C")
            if pressure is not None:
                selected_params.append(f"**Pressure:** {pressure} bar")
            
            if selected_params:
                for param in selected_params:
                    st.markdown(f"- {param}")
            else:
                st.info("💡 Select parameters above to see summary")
        
        # Action Buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.button("🔍 **Run Analysis**", type="primary", use_container_width=True):
                # Create equipment context
                equipment_context = {
                    'equipment_type': equipment_type,
                    'material': material,
                    'environment': environment,
                    'damage_mechanism': damage_mechanism,
                    'temperature': f"{temperature}°C" if temperature is not None else "Not Specified",
                    'pressure': f"{pressure} bar" if pressure is not None else "Not Specified"
                }
                
                # Create analysis query based on context
                context_parts = []
                if damage_mechanism != "Not Specified":
                    context_parts.append(damage_mechanism)
                if equipment_type != "Not Specified":
                    context_parts.append(f"in {equipment_type}")
                if material != "Not Specified":
                    context_parts.append(f"({material})")
                if environment != "Not Specified":
                    context_parts.append(f"for {environment}")
                
                if context_parts:
                    query = f"{analysis_focus} for {' '.join(context_parts)}"
                else:
                    query = f"{analysis_focus} - general MCI engineering guidance"
                
                # Store in session state for processing
                st.session_state.analysis_query = query
                st.session_state.analysis_context = equipment_context
                st.session_state.run_analysis = True
                st.rerun()
        
        with col2:
            # Example configurations
            if st.button("📝 **Load Example**", use_container_width=True):
                st.info("💡 **Example Configuration Loaded**\n\nTry: Carbon Steel pressure vessel in marine environment with pitting corrosion")
        
        with col3:
            # Help
            if st.button("❓ **Help**", use_container_width=True):
                st.info("""
                **🔧 How to use:**
                1. Select equipment parameters above
                2. Choose analysis focus
                3. Click 'Run Analysis'
                4. Review comprehensive results
                
                **💡 Tips:**
                - More parameters = more specific analysis
                - All parameters are optional
                - Results include API references
                """)
    
    else:
        # Results view with sidebar controls
        with st.sidebar:
            st.header("Analysis Controls")
            
            if st.button("🔙 New Analysis", use_container_width=True, type="primary"):
                # Clear results and return to input form
                if "analysis_result" in st.session_state:
                    del st.session_state.analysis_result
                if "analysis_query" in st.session_state:
                    del st.session_state.analysis_query
                st.rerun()
            
            if st.button("📄 Export Report", use_container_width=True):
                context = st.session_state.get("analysis_context", {})
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                report = f"""MCI Engineering Analysis Report
Generated: {timestamp}

EQUIPMENT PARAMETERS:
"""
                for key, value in context.items():
                    if value != "Not Specified":
                        report += f"- {key.replace('_', ' ').title()}: {value}\n"
                
                report += f"\nANALYSIS RESULTS:\n{'-' * 50}\n"
                report += st.session_state.analysis_result
                report += f"\n\n{'-' * 50}\n⚠️ DISCLAIMER: Results must be verified by qualified engineers."
                
                st.download_button(
                    "📄 Download Report",
                    report,
                    file_name=f"mci_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )
            
            st.markdown("---")
            
            # Show analysis context
            st.markdown("### 📋 Analysis Context")
            context = st.session_state.get("analysis_context", {})
            context_items = []
            
            for key, value in context.items():
                if value != "Not Specified":
                    display_key = key.replace('_', ' ').title()
                    context_items.append(f"**{display_key}:** {value}")
            
            if context_items:
                for item in context_items:
                    st.markdown(item)
            else:
                st.markdown("*General analysis*")
            
            # Cache info
            if st.session_state.get("chat_cache"):
                st.markdown("---")
                st.markdown("### 💾 Cache Status")
                st.caption(f"Stored analyses: {len(st.session_state.chat_cache)}")
        
        # Display results in main area
        st.markdown("## 📊 Analysis Results")
        st.markdown(st.session_state.analysis_result)
    
    # Process analysis if requested
    if st.session_state.get("run_analysis", False):
        query = st.session_state.get("analysis_query", "")
        context = st.session_state.get("analysis_context", {})
        
        # Check cache
        cached_response = cache.get(query, context)
        
        if cached_response:
            st.session_state.analysis_result = cached_response + "\n\n*[Cached response]*"
        else:
            with st.spinner("Generating comprehensive analysis..."):
                response = generate_response(query, vectorstores, llm, equipment_context=context)
                cache.set(query, response, context)
                st.session_state.analysis_result = response
        
        # Clear the run flag
        st.session_state.run_analysis = False
        st.rerun()

def main():
    """Main application with page navigation including calculator"""
    
    # Initialize session state for page selection
    if "current_page" not in st.session_state:
        st.session_state.current_page = "ChatBot"
    
    # Page navigation in sidebar with vertical buttons
    st.sidebar.title("🛡️ SmartMCI App")
    
    st.sidebar.markdown("### 📱 Navigation")
    
    # Vertical navigation with optimal sequence
    if st.sidebar.button(
        "💬 ChatBot", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "ChatBot" else "secondary",
        help="Ask questions about materials, corrosion, and integrity"
    ):
        st.session_state.current_page = "ChatBot"
        st.rerun()
    
    if st.sidebar.button(
        "🧮 Calculator", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "Calculator" else "secondary",
        help="Calculate corrosion rates and remaining life"
    ):
        st.session_state.current_page = "Calculator"
        st.rerun()
    
    if st.sidebar.button(
        "🔬 Analysis", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "Analysis" else "secondary",
        help="Structured analysis with equipment parameters"
    ):
        st.session_state.current_page = "Analysis"
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Route to appropriate page
    if st.session_state.current_page == "ChatBot":
        chatbot_page()
    elif st.session_state.current_page == "Analysis":
        analysis_page()
    elif st.session_state.current_page == "Calculator":
        calculator_page()
    
    # Common footer
    st.markdown("---")
    st.caption("⚠️ **Engineering verification required** | Based on API 571/970/584 standards")

if __name__ == "__main__":
    main()