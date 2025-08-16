import streamlit as st
import os
import re
import hashlib
import json
import time
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Generator
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum

# Load environment variables
load_dotenv(override=True)

# Page configuration
st.set_page_config(
    page_title="SmartMCI", 
    layout="wide",
    page_icon="🛡️"
)

# Pydantic Enums for better type safety
class EquipmentType(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    PRESSURE_VESSELS = "Pressure Vessels"
    PIPING_SYSTEMS = "Piping Systems"
    HEAT_EXCHANGERS = "Heat Exchangers"
    STORAGE_TANKS = "Storage Tanks"
    REACTORS = "Reactors"
    COLUMNS_TOWERS = "Columns/Towers"

class Material(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    CARBON_STEEL = "Carbon Steel"
    STAINLESS_STEEL_304_316 = "Stainless Steel 304/316"
    DUPLEX_STAINLESS_STEEL = "Duplex Stainless Steel"
    SUPER_DUPLEX_2507 = "Super Duplex (2507)"
    INCONEL_625 = "Inconel 625"
    HASTELLOY_C276 = "Hastelloy C-276"

class CalcMaterial(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    CARBON_STEEL = "Carbon Steel"
    STAINLESS_STEEL_304_316 = "Stainless Steel 304/316"
    DUPLEX_STAINLESS_STEEL = "Duplex Stainless Steel"
    ALUMINUM = "Aluminum"
    COPPER = "Copper"
    OTHER = "Other"

class DamageMechanism(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    PITTING_CORROSION = "Pitting Corrosion"
    CREVICE_CORROSION = "Crevice Corrosion"
    STRESS_CORROSION_CRACKING = "Stress Corrosion Cracking"
    GENERAL_CORROSION = "General Corrosion"
    HYDROGEN_EMBRITTLEMENT = "Hydrogen Embrittlement"
    FATIGUE_CRACKING = "Fatigue Cracking"
    HIGH_TEMPERATURE_CORROSION = "High Temperature Corrosion"
    EROSION_CORROSION = "Erosion-Corrosion"

class Environment(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    MARINE_OFFSHORE = "Marine/Offshore"
    SOUR_SERVICE_H2S = "Sour Service (H2S)"
    HIGH_TEMPERATURE = "High Temperature"
    CHLORIDE_ENVIRONMENT = "Chloride Environment"
    CAUSTIC_SERVICE = "Caustic Service"
    ATMOSPHERIC = "Atmospheric"

class CalcEnvironment(str, Enum):
    NOT_SPECIFIED = "Not Specified"
    GENERAL_SERVICE = "General Service"
    MARINE_OFFSHORE = "Marine/Offshore"
    SOUR_SERVICE_H2S = "Sour Service (H2S)"
    HIGH_TEMPERATURE = "High Temperature"
    CHLORIDE_ENVIRONMENT = "Chloride Environment"
    ATMOSPHERIC = "Atmospheric"

class ModelType(str, Enum):
    INSTANT = "instant"
    VERSATILE = "versatile"

class ValidationStatus(str, Enum):
    NORMAL = "normal"
    HIGH = "high"
    LOW = "low"
    ERROR = "error"
    UNKNOWN = "unknown"

# Pydantic Models
class ModelConfig(BaseModel):
    name: str = Field(..., description="Model name identifier")
    cost_per_token: float = Field(..., gt=0, description="Relative cost per token")
    speed: str = Field(..., description="Speed category")
    reasoning: str = Field(..., description="Reasoning capability")
    use_cases: List[str] = Field(..., description="Applicable use cases")

class EquipmentContext(BaseModel):
    equipment_type: EquipmentType = EquipmentType.NOT_SPECIFIED
    material: Material = Material.NOT_SPECIFIED
    environment: Environment = Environment.NOT_SPECIFIED
    damage_mechanism: DamageMechanism = DamageMechanism.NOT_SPECIFIED
    temperature: Optional[float] = Field(None, ge=-50, le=1000, description="Temperature in Celsius")
    pressure: Optional[float] = Field(None, ge=0, le=500, description="Pressure in bar")

    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and (v < -273.15):  # Absolute zero check
            raise ValueError('Temperature cannot be below absolute zero')
        return v

class CalculatorInput(BaseModel):
    initial_thickness: float = Field(..., gt=0, le=500, description="Initial thickness in mm")
    current_thickness: float = Field(..., gt=0, le=500, description="Current thickness in mm")
    service_years: float = Field(..., gt=0, le=50, description="Service time in years")
    min_thickness: float = Field(..., gt=0, le=500, description="Minimum required thickness in mm")
    equipment_id: Optional[str] = Field(None, max_length=50, description="Equipment identifier")
    equipment_type: EquipmentType = EquipmentType.NOT_SPECIFIED
    material: CalcMaterial = CalcMaterial.NOT_SPECIFIED
    environment: CalcEnvironment = CalcEnvironment.NOT_SPECIFIED

    @validator('current_thickness')
    def validate_current_thickness(cls, v, values):
        if 'initial_thickness' in values and v > values['initial_thickness']:
            raise ValueError('Current thickness cannot exceed initial thickness')
        return v

    @validator('min_thickness')
    def validate_min_thickness(cls, v, values):
        if 'current_thickness' in values and v > values['current_thickness']:
            raise ValueError('Minimum thickness should not exceed current thickness')
        return v

class CorrosionCalculationResult(BaseModel):
    corrosion_rate: float = Field(..., description="Corrosion rate in mm/year")
    thickness_loss: float = Field(..., description="Total thickness loss in mm")
    remaining_life: Optional[float] = Field(None, description="Remaining life in years")
    error_message: Optional[str] = Field(None, description="Error message if calculation failed")

class ValidationResult(BaseModel):
    status: ValidationStatus
    message: str
    api_reference: str

class ChatMessage(BaseModel):
    role: str = Field(..., regex="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.now)

class CacheEntry(BaseModel):
    response: str
    timestamp: datetime
    query_hash: str

class SearchResult(BaseModel):
    content: str
    success: bool = True
    error_message: Optional[str] = None

class APIConfig(BaseModel):
    groq_api_key: Optional[str] = Field(None, description="Groq API key")
    pinecone_api_key: Optional[str] = Field(None, description="Pinecone API key")
    tavily_api_key: Optional[str] = Field(None, description="Tavily API key")

    @validator('groq_api_key', 'pinecone_api_key')
    def validate_required_keys(cls, v, field):
        if not v:
            raise ValueError(f'{field.name} is required')
        return v

# Configuration data for analysis page
EQUIPMENT_TYPES = [e.value for e in EquipmentType]
MATERIALS = [m.value for m in Material]
DAMAGE_MECHANISMS = [d.value for d in DamageMechanism]
ENVIRONMENTS = [e.value for e in Environment]

# Calculator-specific materials (simplified for Phase 1)
CALC_MATERIALS = [m.value for m in CalcMaterial]
CALC_ENVIRONMENTS = [e.value for e in CalcEnvironment]

# API Documents configuration
API_INDEXES = {
    "api571": "api571-damage-mechanisms",
    "api970": "api970-corrosion-control", 
    "api584": "api584-integrity-windows"
}

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
    CalcMaterial.CARBON_STEEL.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.05, 0.3),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.2, 1.0),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.3, 2.0),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.5, 3.0),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.3, 1.5),
        CalcEnvironment.ATMOSPHERIC.value: (0.02, 0.2)
    },
    CalcMaterial.STAINLESS_STEEL_304_316.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.001, 0.05),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.01, 0.3),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.05, 0.5),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.1, 1.0),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.1, 2.0),
        CalcEnvironment.ATMOSPHERIC.value: (0.001, 0.01)
    },
    CalcMaterial.DUPLEX_STAINLESS_STEEL.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.001, 0.02),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.005, 0.1),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.01, 0.2),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.05, 0.5),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.01, 0.3),
        CalcEnvironment.ATMOSPHERIC.value: (0.001, 0.005)
    },
    CalcMaterial.ALUMINUM.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.01, 0.1),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.1, 0.5),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.05, 0.3),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.2, 1.0),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.2, 1.5),
        CalcEnvironment.ATMOSPHERIC.value: (0.005, 0.05)
    },
    CalcMaterial.COPPER.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.005, 0.05),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.05, 0.3),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.1, 0.8),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.1, 0.5),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.1, 0.8),
        CalcEnvironment.ATMOSPHERIC.value: (0.002, 0.02)
    },
    CalcMaterial.OTHER.value: {
        CalcEnvironment.GENERAL_SERVICE.value: (0.01, 0.5),
        CalcEnvironment.MARINE_OFFSHORE.value: (0.05, 1.0),
        CalcEnvironment.SOUR_SERVICE_H2S.value: (0.1, 2.0),
        CalcEnvironment.HIGH_TEMPERATURE.value: (0.2, 3.0),
        CalcEnvironment.CHLORIDE_ENVIRONMENT.value: (0.1, 2.0),
        CalcEnvironment.ATMOSPHERIC.value: (0.005, 0.1)
    }
}

# Hybrid LLM Model Configuration
MODEL_CONFIG = {
    ModelType.INSTANT.value: ModelConfig(
        name="llama-3.1-8b-instant",
        cost_per_token=0.05,  # Relative cost (instant = 1x)
        speed="very_fast",
        reasoning="basic",
        use_cases=["simple_queries", "quick_facts", "cached_responses"]
    ),
    ModelType.VERSATILE.value: ModelConfig(
        name="llama-3.3-70b-versatile", 
        cost_per_token=0.8,  # Relative cost (versatile = 16x more expensive)
        speed="fast",
        reasoning="advanced",
        use_cases=["complex_analysis", "engineering_calculations", "structured_reports"]
    )
}

# Simple caching system with Pydantic validation
class SimpleCache:
    def __init__(self, ttl_hours: int = 24):
        self.ttl_hours = ttl_hours
        if "chat_cache" not in st.session_state:
            st.session_state.chat_cache = {}
    
    def _create_key(self, query: str, context: Optional[Dict] = None) -> str:
        """Create cache key from query and context"""
        normalized = re.sub(r'[^\w\s]', '', query.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        if context:
            context_str = "|".join([f"{k}:{v}" for k, v in context.items() if v and v != "Not Specified"])
            cache_string = f"{normalized}|{context_str}"
        else:
            cache_string = normalized
            
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        """Get cached response if valid"""
        try:
            key = self._create_key(query, context)
            if key in st.session_state.chat_cache:
                cache_data = st.session_state.chat_cache[key]
                # Validate cache entry
                cache_entry = CacheEntry(**cache_data)
                if datetime.now() - cache_entry.timestamp < timedelta(hours=self.ttl_hours):
                    return cache_entry.response
                else:
                    del st.session_state.chat_cache[key]
        except (ValidationError, KeyError, ValueError):
            # Invalid cache entry, skip
            pass
        return None
    
    def set(self, query: str, response: str, context: Optional[Dict] = None):
        """Cache a response with validation"""
        try:
            key = self._create_key(query, context)
            cache_entry = CacheEntry(
                response=response,
                timestamp=datetime.now(),
                query_hash=key
            )
            st.session_state.chat_cache[key] = cache_entry.dict()
        except ValidationError as e:
            st.warning(f"Cache validation error: {e}")

# Hybrid LLM Functions
def analyze_query_complexity(query: str, equipment_context: Optional[EquipmentContext] = None, chat_history: Optional[List[ChatMessage]] = None) -> ModelType:
    """
    Analyze query complexity to determine which model to use
    Returns: ModelType enum
    """
    
    # Initialize complexity score
    complexity_score = 0
    
    # 1. QUERY LENGTH ANALYSIS
    word_count = len(query.split())
    if word_count > 20:
        complexity_score += 2
    elif word_count > 10:
        complexity_score += 1
    
    # 2. COMPLEXITY KEYWORDS
    complex_keywords = [
        # Analysis keywords
        'analyze', 'analysis', 'comprehensive', 'detailed', 'evaluate', 'assess', 
        'compare', 'comparison', 'calculate', 'computation', 'design', 'optimize',
        
        # Technical depth keywords  
        'mechanism', 'kinetics', 'thermodynamics', 'stress', 'failure', 'root cause',
        'mitigation', 'strategy', 'selection', 'recommendation', 'specification',
        
        # Report/document keywords
        'report', 'documentation', 'procedure', 'guidelines', 'standards', 'compliance',
        
        # Multi-step keywords
        'step by step', 'procedure', 'process', 'methodology', 'approach',
        
        # Quantitative keywords
        'rate', 'calculation', 'formula', 'equation', 'model', 'simulation'
    ]
    
    simple_keywords = [
        'what is', 'define', 'definition', 'explain', 'tell me', 'quick', 'simple',
        'basic', 'introduction', 'overview', 'summary', 'list', 'examples'
    ]
    
    query_lower = query.lower()
    
    # Count complex keywords
    complex_matches = sum(1 for keyword in complex_keywords if keyword in query_lower)
    simple_matches = sum(1 for keyword in simple_keywords if keyword in query_lower)
    
    complexity_score += complex_matches * 2
    complexity_score -= simple_matches * 1
    
    # 3. EQUIPMENT CONTEXT ANALYSIS
    if equipment_context:
        specified_params = sum(1 for field_name, field_value in equipment_context.__dict__.items()
                             if field_value is not None and str(field_value) != "Not Specified")
        if specified_params >= 4:  # Many parameters specified = complex analysis
            complexity_score += 3
        elif specified_params >= 2:
            complexity_score += 1
    
    # 4. CONVERSATION CONTEXT
    if chat_history and len(chat_history) > 2:
        # Check if this is a follow-up to a complex discussion
        recent_messages = chat_history[-4:]
        for msg in recent_messages:
            if any(keyword in msg.content.lower() for keyword in complex_keywords):
                complexity_score += 1
                break
    
    # 5. SPECIFIC USE CASES
    
    # Force VERSATILE for these scenarios
    if any(pattern in query_lower for pattern in [
        'comprehensive analysis',
        'detailed report', 
        'calculate corrosion',
        'remaining life',
        'material selection',
        'api 571', 'api 970', 'api 584',
        'operating limits',
        'damage mechanism',
        'mitigation strateg'
    ]):
        return ModelType.VERSATILE
    
    # Force INSTANT for these scenarios  
    if any(pattern in query_lower for pattern in [
        'what is',
        'define',
        'quick question',
        'simple',
        'hello', 'hi', 'thank',
        'what does', 'how do you'
    ]):
        return ModelType.INSTANT
    
    # 6. FINAL DECISION BASED ON SCORE
    if complexity_score >= 4:
        return ModelType.VERSATILE
    elif complexity_score <= 1:
        return ModelType.INSTANT
    else:
        # Medium complexity - consider additional factors
        if equipment_context or word_count > 15:
            return ModelType.VERSATILE
        else:
            return ModelType.INSTANT

@st.cache_resource
def setup_hybrid_llm():
    """Setup both LLM models for hybrid use with validation"""
    try:
        # Validate API configuration
        api_config = APIConfig(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY")
        )
    except ValidationError as e:
        st.error(f"❌ API Configuration Error: {e}")
        return None, None
    
    try:
        instant_llm = ChatGroq(
            model=MODEL_CONFIG[ModelType.INSTANT.value].name,
            temperature=0.1,
            streaming=True
        )
        
        versatile_llm = ChatGroq(
            model=MODEL_CONFIG[ModelType.VERSATILE.value].name,
            temperature=0.1,
            streaming=True
        )
        
        return instant_llm, versatile_llm
        
    except Exception as e:
        st.error(f"❌ Error setting up hybrid LLM: {e}")
        return None, None

def get_appropriate_llm(query: str, equipment_context: Optional[EquipmentContext] = None, chat_history: Optional[List[ChatMessage]] = None):
    """Get the appropriate LLM based on query complexity"""
    
    # Get both models
    instant_llm, versatile_llm = setup_hybrid_llm()
    
    if not instant_llm or not versatile_llm:
        return None, ModelType.INSTANT
    
    # Analyze complexity
    model_choice = analyze_query_complexity(query, equipment_context, chat_history)
    
    # Return appropriate model
    if model_choice == ModelType.VERSATILE:
        return versatile_llm, ModelType.VERSATILE
    else:
        return instant_llm, ModelType.INSTANT

# Calculator Functions with Pydantic validation
def calculate_corrosion_rate(calc_input: CalculatorInput) -> CorrosionCalculationResult:
    """Calculate corrosion rate in mm/year with Pydantic validation"""
    try:
        thickness_loss = calc_input.initial_thickness - calc_input.current_thickness
        corrosion_rate = thickness_loss / calc_input.service_years
        
        return CorrosionCalculationResult(
            corrosion_rate=corrosion_rate,
            thickness_loss=thickness_loss,
            remaining_life=None  # Will be calculated separately
        )
    except Exception as e:
        return CorrosionCalculationResult(
            corrosion_rate=0.0,
            thickness_loss=0.0,
            remaining_life=None,
            error_message=f"Calculation error: {str(e)}"
        )

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

def validate_corrosion_rate(rate: float, material: CalcMaterial, environment: CalcEnvironment) -> ValidationResult:
    """Validate corrosion rate against API 571 standards with Pydantic"""
    try:
        if material.value in API_571_RATES and environment.value in API_571_RATES[material.value]:
            min_rate, max_rate = API_571_RATES[material.value][environment.value]
            
            if rate < min_rate:
                return ValidationResult(
                    status=ValidationStatus.LOW,
                    message=f"Rate below typical range ({min_rate}-{max_rate} mm/year)",
                    api_reference="API 571 - Verify measurements"
                )
            elif rate > max_rate:
                return ValidationResult(
                    status=ValidationStatus.HIGH,
                    message=f"Rate above typical range ({min_rate}-{max_rate} mm/year)",
                    api_reference="API 571 - Review service conditions"
                )
            else:
                return ValidationResult(
                    status=ValidationStatus.NORMAL,
                    message=f"Rate within expected range ({min_rate}-{max_rate} mm/year)",
                    api_reference="API 571 - Validated"
                )
        else:
            return ValidationResult(
                status=ValidationStatus.UNKNOWN,
                message="No API reference data available",
                api_reference="Manual validation required"
            )
            
    except Exception as e:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message=f"Validation error: {str(e)}",
            api_reference="Error occurred"
        )

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

# Initialize components
@st.cache_resource
def setup_embeddings():
    """Setup embeddings model"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource  
def setup_vectorstores():
    """Setup vector stores for API documents with validation"""
    try:
        # Validate Pinecone API key
        api_config = APIConfig(
            groq_api_key=os.environ.get("GROQ_API_KEY", "dummy"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY", "dummy")
        )
    except ValidationError:
        st.error("❌ PINECONE_API_KEY not found. Please check your .env file.")
        return {}, False
    
    pc = Pinecone(api_key=api_config.pinecone_api_key)
    embeddings = setup_embeddings()
    vectorstores = {}
    
    for api_name, index_name in API_INDEXES.items():
        try:
            existing_indexes = [index.name for index in pc.list_indexes()]
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
    
    return vectorstores, len(vectorstores) > 0

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

def get_conversation_context(chat_history: List[ChatMessage], max_messages: int = 4) -> str:
    """Get recent conversation context with Pydantic validation"""
    if len(chat_history) <= 1:
        return ""
    
    recent_messages = chat_history[-(max_messages*2):]  # Get recent Q&A pairs
    context_parts = []
    
    for msg in recent_messages:
        try:
            # Validate message structure
            validated_msg = ChatMessage(**msg.dict() if hasattr(msg, 'dict') else msg)
            if validated_msg.role == "user":
                context_parts.append(f"User: {validated_msg.content}")
            elif validated_msg.role == "assistant":
                # Include only first 150 characters of response for context
                summary = validated_msg.content[:150] + "..." if len(validated_msg.content) > 150 else validated_msg.content
                context_parts.append(f"Assistant: {summary}")
        except ValidationError:
            # Skip invalid messages
            continue
    
    return "\n".join(context_parts) if context_parts else ""

def search_web_tavily(query: str, max_results: int = 5, is_mci: bool = True) -> SearchResult:
    """Search web using Tavily API for MCI and engineering topics with Pydantic validation"""
    
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_api_key:
        return SearchResult(
            content="Web search unavailable - TAVILY_API_KEY not configured.",
            success=False,
            error_message="Missing API key"
        )
    
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
            
            content = "\n".join(web_content) if web_content else "No relevant web results found."
            return SearchResult(content=content, success=True)
            
        else:
            return SearchResult(
                content=f"Web search error: HTTP {response.status_code}",
                success=False,
                error_message=f"HTTP {response.status_code}"
            )
            
    except Exception as e:
        return SearchResult(
            content=f"Web search failed: {str(e)}",
            success=False,
            error_message=str(e)
        )

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

def generate_followup_questions(user_query: str, assistant_response: str) -> List[str]:
    """
    Generate technical follow-up questions based on the conversation context
    to guide users toward more in-depth engineering analysis
    """
    
    query_lower = user_query.lower()
    response_lower = assistant_response.lower()
    followup_questions = []
    
    # Corrosion-related follow-ups
    if any(term in query_lower for term in ['corrosion', 'pitting', 'crevice', 'general corrosion']):
        if 'material' in response_lower and 'environment' in response_lower:
            followup_questions.extend([
                "What are the critical operating parameters (temperature, pressure, pH) that could accelerate this corrosion mechanism?",
                "How would you calculate the expected corrosion rate for these specific conditions?",
                "What inspection techniques would be most effective for detecting this type of corrosion?"
            ])
        else:
            followup_questions.extend([
                "What specific materials would you recommend for this corrosive environment?",
                "How do temperature and chloride concentration affect the corrosion kinetics?",
                "What mitigation strategies beyond material selection should be considered?"
            ])
    
    # Stress corrosion cracking follow-ups
    if any(term in query_lower for term in ['stress corrosion', 'scc', 'cracking']):
        followup_questions.extend([
            "What stress levels and environmental conditions create the threshold for SCC initiation?",
            "How would you design a monitoring program to detect early-stage cracking?",
            "What residual stress mitigation techniques would be most effective for this application?"
        ])
    
    # Material selection follow-ups
    if any(term in query_lower for term in ['material selection', 'alloy', 'steel', 'stainless']):
        if 'temperature' in response_lower or 'pressure' in response_lower:
            followup_questions.extend([
                "What are the mechanical property requirements at the operating temperature?",
                "How would thermal cycling affect material performance and life expectancy?",
                "What welding procedures and heat treatment would be required for this material?"
            ])
        else:
            followup_questions.extend([
                "What are the specific PREN (Pitting Resistance Equivalent Number) requirements for this service?",
                "How would you verify material compliance through testing protocols?",
                "What backup material options should be considered if primary selection fails?"
            ])
    
    # High temperature follow-ups
    if any(term in query_lower for term in ['high temperature', 'thermal', 'heat']):
        followup_questions.extend([
            "What creep-rupture data should be evaluated for long-term service reliability?",
            "How would you calculate the remaining life based on time-temperature parameters?",
            "What high-temperature inspection methods would detect degradation mechanisms?"
        ])
    
    # Marine/offshore follow-ups
    if any(term in query_lower for term in ['marine', 'offshore', 'seawater', 'chloride']):
        followup_questions.extend([
            "What cathodic protection current density would be required for this geometry?",
            "How would you design the coating system specification for this marine environment?",
            "What specific API RP 14E requirements apply to this offshore application?"
        ])
    
    # Sour service follow-ups
    if any(term in query_lower for term in ['sour', 'h2s', 'hydrogen sulfide', 'nace']):
        followup_questions.extend([
            "What H2S partial pressure limits apply according to NACE MR0175/ISO 15156?",
            "How would you qualify welding procedures for sour service applications?",
            "What hydrogen charging test requirements should be specified for material validation?"
        ])
    
    # Equipment-specific follow-ups
    if any(term in query_lower for term in ['pressure vessel', 'piping', 'heat exchanger']):
        followup_questions.extend([
            "What fitness-for-service assessment approach would you use per API 579?",
            "How would you establish inspection intervals based on damage mechanism rates?",
            "What operating limit adjustments would maintain integrity throughout service life?"
        ])
    
    # Welding follow-ups
    if any(term in query_lower for term in ['welding', 'weld', 'heat affected zone', 'haz']):
        followup_questions.extend([
            "What preheat and PWHT requirements apply for this material and thickness combination?",
            "How would you prevent hydrogen-induced cracking during welding operations?",
            "What NDE requirements would ensure weld integrity for this service?"
        ])
    
    # Inspection and monitoring follow-ups
    if any(term in query_lower for term in ['inspection', 'monitoring', 'ndt', 'testing']):
        followup_questions.extend([
            "What inspection frequency would be appropriate based on the calculated corrosion rate?",
            "How would you establish statistically valid inspection locations for this damage mechanism?",
            "What online monitoring systems could provide real-time integrity assessment?"
        ])
    
    # Process safety follow-ups
    if any(term in query_lower for term in ['safety', 'hazard', 'risk', 'failure']):
        followup_questions.extend([
            "What would be the consequences of failure and how does this impact inspection strategy?",
            "How would you conduct a quantitative risk assessment for this degradation mechanism?",
            "What emergency response procedures should be in place for this failure mode?"
        ])
    
    # API standards follow-ups
    if any(term in query_lower for term in ['api 571', 'api 970', 'api 584']):
        if 'api 571' in query_lower:
            followup_questions.extend([
                "What environmental conditions would shift this damage mechanism to a higher severity category?",
                "How would you apply the damage mechanism matrix to establish inspection requirements?",
                "What material property changes occur as this damage mechanism progresses?"
            ])
        elif 'api 970' in query_lower:
            followup_questions.extend([
                "What corrosion monitoring program would verify the effectiveness of these control measures?",
                "How would you optimize chemical inhibitor dosing for these specific conditions?",
                "What backup corrosion control methods should be considered?"
            ])
        elif 'api 584' in query_lower:
            followup_questions.extend([
                "How would you establish safe operating limits based on damage mechanism kinetics?",
                "What process upset scenarios could exceed the integrity operating window?",
                "How frequently should operating limits be reassessed based on inspection data?"
            ])
    
    # Design and engineering follow-ups
    if any(term in query_lower for term in ['design', 'specification', 'standard']):
        followup_questions.extend([
            "What design margins should be applied for this damage mechanism and operating severity?",
            "How would you incorporate lessons learned from similar applications in the industry?",
            "What future operating condition changes should be considered in the design?"
        ])
    
    # Mitigation and prevention follow-ups
    if any(term in query_lower for term in ['mitigation', 'prevention', 'protection']):
        followup_questions.extend([
            "What is the economic trade-off between prevention cost and expected damage cost?",
            "How would you validate the effectiveness of these mitigation measures in service?",
            "What contingency plans should be in place if primary mitigation fails?"
        ])
    
    # If response contains calculations or rates
    if any(term in response_lower for term in ['rate', 'calculation', 'mm/year', 'years']):
        followup_questions.extend([
            "How would you validate these calculated rates against field inspection data?",
            "What safety factors should be applied to these calculations for critical equipment?",
            "How would process condition changes affect these calculated rates?"
        ])
    
    # General technical depth follow-ups if no specific matches
    if not followup_questions:
        if len(assistant_response) > 500:  # Detailed response
            followup_questions.extend([
                "What specific technical parameters would you need to quantify for a detailed engineering assessment?",
                "How would you verify these recommendations through testing or field experience?",
                "What are the critical success factors for implementing these recommendations?"
            ])
        else:  # Shorter response
            followup_questions.extend([
                "What additional technical details would help optimize this for your specific application?",
                "How would you quantify the engineering requirements for this scenario?",
                "What are the next steps for a detailed technical evaluation?"
            ])
    
    # Remove duplicates and limit to 3 most relevant questions
    unique_questions = list(dict.fromkeys(followup_questions))
    return unique_questions[:3]

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

# Hybrid streaming response function with Pydantic validation
def generate_response_stream_hybrid(query: str, vectorstores: dict, chat_history: Optional[List[ChatMessage]] = None, equipment_context: Optional[EquipmentContext] = None):
    """Generate streaming response with hybrid LLM selection and Pydantic validation"""
    try:
        # Validate inputs
        if chat_history:
            validated_chat_history = []
            for msg in chat_history:
                try:
                    if isinstance(msg, dict):
                        validated_msg = ChatMessage(**msg)
                    else:
                        validated_msg = msg
                    validated_chat_history.append(validated_msg)
                except ValidationError:
                    # Skip invalid messages
                    continue
            chat_history = validated_chat_history
        
        # Get appropriate LLM
        llm, model_used = get_appropriate_llm(query, equipment_context, chat_history)
        
        if not llm:
            yield "❌ Error: Could not initialize LLM models."
            return
        
        # Check if query is MCI or engineering-related
        is_mci = is_mci_related(query)
        is_engineering = is_engineering_related(query)
        
        # Non-engineering topics - quick redirect
        if not is_engineering:
            yield create_polite_redirect(query)
            return
        
        # Document retrieval
        docs = retrieve_documents(query, vectorstores)
        api_context = format_documents(docs)
        
        web_context = ""
        search_used = ""
        
        # Smarter web searching based on model and complexity
        should_search = False
        
        if model_used == ModelType.VERSATILE:
            # For complex queries, be more thorough with search
            if is_mci and len(docs) < 3:
                should_search = True
            elif is_engineering and len(docs) < 2:
                should_search = True
        else:
            # For simple queries, only search if no docs found
            if len(docs) == 0:
                should_search = True
        
        if should_search:
            yield "🔍 Searching...\n\n"
            search_result = search_web_tavily(query, is_mci=True)
            if search_result.success:
                web_context = search_result.content
                search_used = "✅ "
            else:
                web_context = search_result.content
                search_used = "⚠️ "
        
        # Prepare prompt
        if equipment_context:
            context_parts = []
            for field_name, field_value in equipment_context.__dict__.items():
                if field_value is not None and str(field_value) != "Not Specified":
                    display_name = field_name.replace('_', ' ').title()
                    context_parts.append(f"{display_name}: {field_value}")
            
            context_string = " | ".join(context_parts) if context_parts else "General analysis"
            
            prompt_template = create_analysis_prompt()
            formatted_prompt = prompt_template.format(
                context=api_context,
                equipment_context=context_string,
                query=query
            )
        else:
            # Adjust conversation context based on model
            max_messages = 3 if model_used == ModelType.VERSATILE else 2
            conversation_context = get_conversation_context(chat_history or [], max_messages=max_messages)
            
            prompt_template = create_hybrid_chat_prompt()
            formatted_prompt = prompt_template.format(
                api_context=api_context,
                web_context=web_context,
                conversation_history=conversation_context,
                query=query,
                search_used=search_used
            )
        
        # Stream response
        response_stream = llm.stream(formatted_prompt)
        
        for chunk in response_stream:
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield content
        
        # Add footer
        if is_engineering and not is_mci:
            yield "\n\n💡 *For specific MCI questions, ask about API 571/970/584 standards.*"
        elif web_context and search_used:
            yield "\n\n*📡 Enhanced with web search*"
            
    except Exception as e:
        yield f"\n\n❌ Error: {str(e)}. Please try again."

def generate_response(query: str, vectorstores: dict, llm=None, chat_history: Optional[List[ChatMessage]] = None, equipment_context: Optional[EquipmentContext] = None) -> str:
    """Generate non-streaming response for compatibility"""
    try:
        # Collect all streaming chunks
        full_response = ""
        for chunk in generate_response_stream_hybrid(query, vectorstores, chat_history, equipment_context):
            full_response += chunk
        
        return full_response
            
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again."

# Calculator page with Pydantic validation
def calculator_page():
    """Corrosion Rate Calculator page implementation with Pydantic validation"""
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
            EQUIPMENT_TYPES,
            help="Select equipment type for better validation"
        )
    
    with col2:
        material = st.selectbox(
            "Material",
            CALC_MATERIALS,
            help="Select material for API cross-reference"
        )
        
        environment = st.selectbox(
            "Service Environment",
            CALC_ENVIRONMENTS,
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
    
    # Validate inputs using Pydantic
    try:
        calc_input = CalculatorInput(
            initial_thickness=initial_thickness,
            current_thickness=current_thickness,
            service_years=service_years,
            min_thickness=min_thickness,
            equipment_id=equipment_id if equipment_id else None,
            equipment_type=EquipmentType(equipment_type),
            material=CalcMaterial(material),
            environment=CalcEnvironment(environment)
        )
        
        # If validation passes, perform calculations
        calc_result = calculate_corrosion_rate(calc_input)
        
        if calc_result.error_message:
            st.error(f"Calculation Error: {calc_result.error_message}")
        else:
            # Calculate remaining life
            remaining_life, life_error = calculate_remaining_life(
                calc_input.current_thickness, 
                calc_input.min_thickness, 
                calc_result.corrosion_rate
            )
            
            # Display main results
            st.markdown("---")
            st.markdown("## 📊 **Calculation Results**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Corrosion Rate", 
                    f"{calc_result.corrosion_rate:.3f} mm/year",
                    help="Rate of thickness loss per year"
                )
            
            with col2:
                st.metric(
                    "Total Thickness Loss",
                    f"{calc_result.thickness_loss:.2f} mm",
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
            if calc_input.material != CalcMaterial.NOT_SPECIFIED and calc_input.environment != CalcEnvironment.NOT_SPECIFIED:
                st.markdown("### 📚 **API 571 Validation**")
                
                validation = validate_corrosion_rate(calc_result.corrosion_rate, calc_input.material, calc_input.environment)
                
                if validation.status == ValidationStatus.NORMAL:
                    st.success(f"✅ {validation.message}")
                elif validation.status == ValidationStatus.HIGH:
                    st.warning(f"⚠️ {validation.message}")
                elif validation.status == ValidationStatus.LOW:
                    st.info(f"ℹ️ {validation.message}")
                else:
                    st.info(f"ℹ️ {validation.message}")
                
                if validation.api_reference:
                    st.caption(f"Reference: {validation.api_reference}")
            
            # Generate chart
            if st.checkbox("📈 Show Thickness Trend Chart"):
                chart = create_thickness_trend_chart(
                    calc_input.initial_thickness, 
                    calc_input.current_thickness, 
                    calc_input.service_years, 
                    calc_result.corrosion_rate
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 **Recommendations**")
            
            recommendations = []
            
            if calc_result.corrosion_rate > 1.0:
                recommendations.append("• **High corrosion rate detected** - Review service conditions and consider mitigation")
                recommendations.append("• **Increase inspection frequency** - Monitor more closely")
                recommendations.append("• **Consider material upgrade** - Evaluate more corrosion-resistant options")
            elif calc_result.corrosion_rate > 0.5:
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
                'equipment_id': calc_input.equipment_id or 'Not specified',
                'equipment_type': calc_input.equipment_type.value,
                'material': calc_input.material.value,
                'environment': calc_input.environment.value,
                'initial_thickness': calc_input.initial_thickness,
                'current_thickness': calc_input.current_thickness,
                'service_years': calc_input.service_years,
                'min_thickness': calc_input.min_thickness,
                'corrosion_rate': calc_result.corrosion_rate,
                'thickness_loss': calc_result.thickness_loss,
                'remaining_life': remaining_life if remaining_life is not None else 'N/A',
                'validation_status': validation.status.value if 'validation' in locals() else 'not_validated',
                'validation_message': validation.message if 'validation' in locals() else 'No validation performed',
                'api_reference': validation.api_reference if 'validation' in locals() else 'Not available',
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
                        file_name=f"corrosion_calc_{calc_input.equipment_id or 'equipment'}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                # Quick summary export
                summary = f"""Equipment: {calc_input.equipment_id or 'Not specified'}
Corrosion Rate: {calc_result.corrosion_rate:.3f} mm/year
Remaining Life: {remaining_life:.1f} years if remaining_life is not None else 'N/A'
Status: {validation.status.value.title() if 'validation' in locals() else 'not_validated'}"""
                
                st.download_button(
                    "📋 Quick Summary",
                    summary,
                    file_name=f"summary_{calc_input.equipment_id or 'equipment'}.txt",
                    mime="text/plain"
                )
            
            with col3:
                # CSV export for data analysis
                if st.button("📊 CSV Data"):
                    csv_data = f"""Parameter,Value,Unit
Equipment ID,{calc_input.equipment_id or 'Not specified'},
Equipment Type,{calc_input.equipment_type.value},
Material,{calc_input.material.value},
Environment,{calc_input.environment.value},
Initial Thickness,{calc_input.initial_thickness},mm
Current Thickness,{calc_input.current_thickness},mm
Service Time,{calc_input.service_years},years
Minimum Thickness,{calc_input.min_thickness},mm
Corrosion Rate,{calc_result.corrosion_rate:.3f},mm/year
Thickness Loss,{calc_result.thickness_loss:.2f},mm
Remaining Life,{remaining_life if remaining_life is not None else 'N/A'},years
"""
                    st.download_button(
                        "📊 Download CSV",
                        csv_data,
                        file_name=f"corrosion_data_{calc_input.equipment_id or 'equipment'}.csv",
                        mime="text/csv"
                    )
    
    except ValidationError as e:
        # Handle Pydantic validation errors
        st.error("❌ **Input Validation Error**")
        for error in e.errors():
            field = error['loc'][0] if error['loc'] else 'unknown'
            message = error['msg']
            st.error(f"**{field.replace('_', ' ').title()}**: {message}")
    
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

# Hybrid chatbot page with Pydantic validation
def chatbot_page():
    """Updated chatbot page with hybrid LLM and follow-up questions with Pydantic validation"""
    st.title("🛡️ MCI AI-Assistance")
    st.markdown("**MCI Engineering Consultant**")
    
    # Initialize session state for chatbot with validation
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Validate existing messages
    validated_messages = []
    for msg in st.session_state.chat_messages:
        try:
            if isinstance(msg, dict):
                validated_msg = ChatMessage(**msg)
            else:
                validated_msg = msg
            validated_messages.append(validated_msg)
        except ValidationError:
            # Skip invalid messages
            continue
    st.session_state.chat_messages = validated_messages
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing SmartMCI System..."):
        vectorstores, available = setup_vectorstores()
        instant_llm, versatile_llm = setup_hybrid_llm()
        
        if not available or not instant_llm or not versatile_llm:
            st.error("❌ System initialization failed. Please check configuration.")
            st.stop()
        
        # Check Tavily API key
        tavily_available = bool(os.environ.get("TAVILY_API_KEY"))
        if tavily_available:
            st.success("🌐 Web search enabled via Tavily")
        else:
            st.warning("⚠️ Web search disabled - TAVILY_API_KEY not found")
    
    # Sidebar with quick actions and model info
    with st.sidebar:
        st.header("Quick Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        
        for example in SMART_EXAMPLES:
            if st.button(example, key=f"chat_ex_{hash(example)}", use_container_width=True):
                # Check if this example is already being processed
                if st.session_state.get("processing_example") == example:
                    continue
                    
                # Add user message only once with validation
                try:
                    user_msg = ChatMessage(role="user", content=example)
                    st.session_state.chat_messages.append(user_msg)
                    
                    # Check cache first
                    cached_response = cache.get(example)
                    if cached_response:
                        response = cached_response + "\n\n*[Cached response]*"
                        assistant_msg = ChatMessage(role="assistant", content=response)
                        st.session_state.chat_messages.append(assistant_msg)
                    else:
                        # Set flag for streaming response and mark as processing
                        st.session_state.pending_example = example
                        st.session_state.processing_example = example
                    
                    st.rerun()
                except ValidationError as e:
                    st.error(f"Message validation error: {e}")
        
        # Show cache stats if available
        if st.session_state.get("chat_cache"):
            st.markdown("---")
            st.caption(f"💾 Cached responses: {len(st.session_state.chat_cache)}")
    
    # Display chat history with follow-up questions
    for i, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message.role):
            st.markdown(message.content)
        
        # Add follow-up questions after each assistant response
        if message.role == "assistant" and i > 0:
            # Get the user query that prompted this response
            user_query = st.session_state.chat_messages[i-1].content if i > 0 else ""
            assistant_response = message.content
            
            # Generate follow-up questions
            try:
                followup_questions = generate_followup_questions(user_query, assistant_response)
                
                if followup_questions:
                    st.markdown("---")
                    st.markdown("**🔍 Related Questions:**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for j, question in enumerate(followup_questions[:3]):
                        with [col1, col2, col3][j]:
                            if st.button(
                                question[:80] + "..." if len(question) > 80 else question,
                                key=f"followup_{i}_{j}_{hash(question)}",
                                help=question,
                                use_container_width=True
                            ):
                                try:
                                    new_msg = ChatMessage(role="user", content=question)
                                    st.session_state.chat_messages.append(new_msg)
                                    st.rerun()
                                except ValidationError as e:
                                    st.error(f"Message validation error: {e}")
            except Exception as e:
                pass  # Silently continue if follow-up generation fails
    
    # Handle pending example streaming
    if st.session_state.get("pending_example"):
        example = st.session_state.pending_example
        
        # Clear the pending flag immediately to prevent re-execution
        del st.session_state.pending_example
        
        # Show streaming response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                for chunk in generate_response_stream_hybrid(example, vectorstores, st.session_state.chat_messages):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Cache and store the response with validation
                clean_response = full_response.replace("\n\n*📡 Enhanced with web search*", "")
                cache.set(example, clean_response)
                
                try:
                    assistant_msg = ChatMessage(role="assistant", content=full_response)
                    st.session_state.chat_messages.append(assistant_msg)
                except ValidationError as e:
                    st.error(f"Response validation error: {e}")
                
                # Generate and show follow-up questions immediately after streaming
                try:
                    followup_questions = generate_followup_questions(example, full_response)
                    if followup_questions:
                        st.markdown("---")
                        st.markdown("**🔍 Technical Follow-up Questions:**")
                        
                        col1, col2, col3 = st.columns(3)
                        for j, question in enumerate(followup_questions[:3]):
                            with [col1, col2, col3][j]:
                                if st.button(
                                    question[:80] + "..." if len(question) > 80 else question,
                                    key=f"example_followup_{j}_{hash(question)}",
                                    help=question,
                                    use_container_width=True
                                ):
                                    try:
                                        new_msg = ChatMessage(role="user", content=question)
                                        st.session_state.chat_messages.append(new_msg)
                                        st.rerun()
                                    except ValidationError as e:
                                        st.error(f"Message validation error: {e}")
                except Exception as e:
                    pass  # Silently continue if follow-up generation fails
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                message_placeholder.markdown(error_msg)
                try:
                    error_message = ChatMessage(role="assistant", content=error_msg)
                    st.session_state.chat_messages.append(error_message)
                except ValidationError:
                    pass
            
            finally:
                # Clear processing flag
                if "processing_example" in st.session_state:
                    del st.session_state.processing_example
    
    # Check if we need to auto-respond to a follow-up question
    if (len(st.session_state.chat_messages) > 0 and 
        st.session_state.chat_messages[-1].role == "user" and
        not st.session_state.get("processing_prompt") and
        not st.session_state.get("pending_example")):
        
        # This means a follow-up question was just added, auto-generate response
        latest_prompt = st.session_state.chat_messages[-1].content
        
        # Check cache first
        cached_response = cache.get(latest_prompt)
        if cached_response:
            response = cached_response + "\n\n*[Cached response]*"
            with st.chat_message("assistant"):
                st.markdown(response)
                
                # Generate and show follow-up questions for cached responses
                try:
                    followup_questions = generate_followup_questions(latest_prompt, response)
                    if followup_questions:
                        st.markdown("---")
                        st.markdown("**🔍 Technical Follow-up Questions:**")
                        
                        col1, col2, col3 = st.columns(3)
                        for j, question in enumerate(followup_questions[:3]):
                            with [col1, col2, col3][j]:
                                if st.button(
                                    question[:80] + "..." if len(question) > 80 else question,
                                    key=f"auto_cached_followup_{j}_{hash(question)}_{len(st.session_state.chat_messages)}",
                                    help=question,
                                    use_container_width=True
                                ):
                                    try:
                                        new_msg = ChatMessage(role="user", content=question)
                                        st.session_state.chat_messages.append(new_msg)
                                        st.rerun()
                                    except ValidationError as e:
                                        st.error(f"Message validation error: {e}")
                except Exception as e:
                    pass
                    
            try:
                assistant_msg = ChatMessage(role="assistant", content=response)
                st.session_state.chat_messages.append(assistant_msg)
            except ValidationError as e:
                st.error(f"Response validation error: {e}")
        else:
            # Generate streaming response automatically
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in generate_response_stream_hybrid(latest_prompt, vectorstores, st.session_state.chat_messages[:-1]):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Cache and store the response with validation
                    clean_response = full_response.replace("\n\n*📡 Enhanced with web search*", "")
                    cache.set(latest_prompt, clean_response)
                    
                    try:
                        assistant_msg = ChatMessage(role="assistant", content=full_response)
                        st.session_state.chat_messages.append(assistant_msg)
                    except ValidationError as e:
                        st.error(f"Response validation error: {e}")
                    
                    # Generate and show follow-up questions immediately after streaming
                    try:
                        followup_questions = generate_followup_questions(latest_prompt, full_response)
                        if followup_questions:
                            st.markdown("---")
                            st.markdown("**🔍 Technical Follow-up Questions:**")
                            
                            col1, col2, col3 = st.columns(3)
                            for j, question in enumerate(followup_questions[:3]):
                                with [col1, col2, col3][j]:
                                    if st.button(
                                        question[:80] + "..." if len(question) > 80 else question,
                                        key=f"auto_stream_followup_{j}_{hash(question)}_{len(st.session_state.chat_messages)}",
                                        help=question,
                                        use_container_width=True
                                    ):
                                        try:
                                            new_msg = ChatMessage(role="user", content=question)
                                            st.session_state.chat_messages.append(new_msg)
                                            st.rerun()
                                        except ValidationError as e:
                                            st.error(f"Message validation error: {e}")
                    except Exception as e:
                        pass
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    try:
                        error_message = ChatMessage(role="assistant", content=error_msg)
                        st.session_state.chat_messages.append(error_message)
                    except ValidationError:
                        pass

    # Chat input with hybrid streaming and validation
    if prompt := st.chat_input("Ask about materials, corrosion, integrity, or engineering..."):
        # Check if we're already processing this prompt
        if st.session_state.get("processing_prompt") == prompt:
            return
            
        # Add user message to chat history with validation
        try:
            user_msg = ChatMessage(role="user", content=prompt)
            st.session_state.chat_messages.append(user_msg)
        except ValidationError as e:
            st.error(f"Input validation error: {e}")
            return
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check cache first
        cached_response = cache.get(prompt)
        if cached_response:
            response = cached_response + "\n\n*[Cached response]*"
            with st.chat_message("assistant"):
                st.markdown(response)
                
                # Generate and show follow-up questions for cached responses
                try:
                    followup_questions = generate_followup_questions(prompt, response)
                    if followup_questions:
                        st.markdown("---")
                        st.markdown("**🔍 Technical Follow-up Questions:**")
                        
                        col1, col2, col3 = st.columns(3)
                        for j, question in enumerate(followup_questions[:3]):
                            with [col1, col2, col3][j]:
                                if st.button(
                                    question[:80] + "..." if len(question) > 80 else question,
                                    key=f"cached_followup_{j}_{hash(question)}",
                                    help=question,
                                    use_container_width=True
                                ):
                                    try:
                                        new_msg = ChatMessage(role="user", content=question)
                                        st.session_state.chat_messages.append(new_msg)
                                        st.rerun()
                                    except ValidationError as e:
                                        st.error(f"Message validation error: {e}")
                except Exception as e:
                    pass  # Silently continue if follow-up generation fails
                    
            try:
                assistant_msg = ChatMessage(role="assistant", content=response)
                st.session_state.chat_messages.append(assistant_msg)
            except ValidationError as e:
                st.error(f"Response validation error: {e}")
        else:
            # Mark as processing
            st.session_state.processing_prompt = prompt
            
            # Generate streaming response with hybrid model selection
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in generate_response_stream_hybrid(prompt, vectorstores, st.session_state.chat_messages):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Cache and store the response with validation
                    clean_response = full_response.replace("\n\n*📡 Enhanced with web search*", "")
                    cache.set(prompt, clean_response)
                    
                    try:
                        assistant_msg = ChatMessage(role="assistant", content=full_response)
                        st.session_state.chat_messages.append(assistant_msg)
                    except ValidationError as e:
                        st.error(f"Response validation error: {e}")
                    
                    # Generate and show follow-up questions immediately after streaming
                    try:
                        followup_questions = generate_followup_questions(prompt, full_response)
                        if followup_questions:
                            st.markdown("---")
                            st.markdown("**🔍 Technical Follow-up Questions:**")
                            
                            col1, col2, col3 = st.columns(3)
                            for j, question in enumerate(followup_questions[:3]):
                                with [col1, col2, col3][j]:
                                    if st.button(
                                        question[:80] + "..." if len(question) > 80 else question,
                                        key=f"stream_followup_{j}_{hash(question)}",
                                        help=question,
                                        use_container_width=True
                                    ):
                                        try:
                                            new_msg = ChatMessage(role="user", content=question)
                                            st.session_state.chat_messages.append(new_msg)
                                            st.rerun()
                                        except ValidationError as e:
                                            st.error(f"Message validation error: {e}")
                    except Exception as e:
                        pass  # Silently continue if follow-up generation fails
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    try:
                        error_message = ChatMessage(role="assistant", content=error_msg)
                        st.session_state.chat_messages.append(error_message)
                    except ValidationError:
                        pass
                
                finally:
                    # Clear processing flag
                    if "processing_prompt" in st.session_state:
                        del st.session_state.processing_prompt
    
    # Welcome message for new users
    if not st.session_state.chat_messages:
        st.markdown("""
        I'm your consultant for **Materials, Corrosion & Integrity (MCI)** based on American Petroleum Institute (API) standards:

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

# Analysis page with hybrid LLM and Pydantic validation
def analysis_page():
    """Structured analysis page implementation with hybrid streaming and Pydantic validation"""
    st.title("🔬 SmartMCI Analysis")
    st.markdown("**Structured Analysis with Equipment Parameters**")
    
    # Initialize cache
    cache = SimpleCache()
    
    # Setup components
    with st.spinner("Initializing analysis tools..."):
        vectorstores, available = setup_vectorstores()
        instant_llm, versatile_llm = setup_hybrid_llm()
        
        if not available or not instant_llm or not versatile_llm:
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
        
        # Input form on main page with Pydantic validation
        st.markdown("## 📋 Equipment Parameters")
        st.markdown("Configure your equipment details for comprehensive MCI analysis")
        
        # Equipment Information Section
        with st.expander("🏭 **Equipment Information**", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                equipment_type = st.selectbox(
                    "Equipment Type:", 
                    EQUIPMENT_TYPES,
                    help="Select equipment for specific guidance"
                )
                
                material = st.selectbox(
                    "Material:", 
                    MATERIALS,
                    help="Select material for specific recommendations"
                )
            
            with col2:
                environment = st.selectbox(
                    "Service Environment:", 
                    ENVIRONMENTS,
                    help="Select environment for specific analysis"
                )
                
                damage_mechanism = st.selectbox(
                    "Damage Type:", 
                    DAMAGE_MECHANISMS,
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
        
        # Validate equipment context
        try:
            equipment_context = EquipmentContext(
                equipment_type=EquipmentType(equipment_type),
                material=Material(material),
                environment=Environment(environment),
                damage_mechanism=DamageMechanism(damage_mechanism),
                temperature=temperature,
                pressure=pressure
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
                if equipment_context.equipment_type != EquipmentType.NOT_SPECIFIED:
                    selected_params.append(f"**Equipment:** {equipment_context.equipment_type.value}")
                if equipment_context.material != Material.NOT_SPECIFIED:
                    selected_params.append(f"**Material:** {equipment_context.material.value}")
                if equipment_context.environment != Environment.NOT_SPECIFIED:
                    selected_params.append(f"**Environment:** {equipment_context.environment.value}")
                if equipment_context.damage_mechanism != DamageMechanism.NOT_SPECIFIED:
                    selected_params.append(f"**Damage Type:** {equipment_context.damage_mechanism.value}")
                if equipment_context.temperature is not None:
                    selected_params.append(f"**Temperature:** {equipment_context.temperature}°C")
                if equipment_context.pressure is not None:
                    selected_params.append(f"**Pressure:** {equipment_context.pressure} bar")
                
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
                    # Create analysis query based on context
                    context_parts = []
                    if equipment_context.damage_mechanism != DamageMechanism.NOT_SPECIFIED:
                        context_parts.append(equipment_context.damage_mechanism.value)
                    if equipment_context.equipment_type != EquipmentType.NOT_SPECIFIED:
                        context_parts.append(f"in {equipment_context.equipment_type.value}")
                    if equipment_context.material != Material.NOT_SPECIFIED:
                        context_parts.append(f"({equipment_context.material.value})")
                    if equipment_context.environment != Environment.NOT_SPECIFIED:
                        context_parts.append(f"for {equipment_context.environment.value}")
                    
                    if context_parts:
                        query = f"{analysis_focus} for {' '.join(context_parts)}"
                    else:
                        query = f"{analysis_focus} - general MCI engineering guidance"
                    
                    # Store in session state for streaming processing
                    st.session_state.analysis_query = query
                    st.session_state.analysis_context = equipment_context
                    st.session_state.stream_analysis = True
                    st.rerun()
            
            with col2:
                # Example configurations
                if st.button("📋 **Load Example**", use_container_width=True):
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
        
        except ValidationError as e:
            st.error("❌ **Parameter Validation Error**")
            for error in e.errors():
                field = error['loc'][0] if error['loc'] else 'unknown'
                message = error['msg']
                st.error(f"**{field.replace('_', ' ').title()}**: {message}")
    
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
                context = st.session_state.get("analysis_context")
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                report = f"""MCI Engineering Analysis Report
Generated: {timestamp}

EQUIPMENT PARAMETERS:
"""
                if context:
                    try:
                        validated_context = EquipmentContext(**context.dict() if hasattr(context, 'dict') else context)
                        for field_name, field_value in validated_context.__dict__.items():
                            if field_value is not None and str(field_value) != "Not Specified":
                                display_name = field_name.replace('_', ' ').title()
                                report += f"- {display_name}: {field_value}\n"
                    except (ValidationError, AttributeError):
                        report += "- Context validation error\n"
                
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
            context = st.session_state.get("analysis_context")
            context_items = []
            
            if context:
                try:
                    validated_context = EquipmentContext(**context.dict() if hasattr(context, 'dict') else context)
                    for field_name, field_value in validated_context.__dict__.items():
                        if field_value is not None and str(field_value) != "Not Specified":
                            display_name = field_name.replace('_', ' ').title()
                            context_items.append(f"**{display_name}:** {field_value}")
                except (ValidationError, AttributeError):
                    context_items.append("*Context validation error*")
            
            if context_items:
                for item in context_items:
                    st.markdown(item)
            else:
                st.markdown("*General analysis*")
            
            # Cache info
            if st.session_state.get("chat_cache"):
                st.markdown("---")
                st.caption(f"💾 Cached analyses: {len(st.session_state.chat_cache)}")
        
        # Display results in main area
        st.markdown("## 📊 Analysis Results")
        st.markdown(st.session_state.analysis_result)
    
    # Process streaming analysis if requested
    if st.session_state.get("stream_analysis", False):
        query = st.session_state.get("analysis_query", "")
        context = st.session_state.get("analysis_context")
        
        # Validate context
        try:
            if context:
                validated_context = EquipmentContext(**context.dict() if hasattr(context, 'dict') else context)
            else:
                validated_context = None
        except ValidationError:
            validated_context = None
            st.warning("⚠️ Context validation failed, proceeding with general analysis")
        
        # Check cache
        context_dict = validated_context.dict() if validated_context else {}
        cached_response = cache.get(query, context_dict)
        
        if cached_response:
            st.session_state.analysis_result = cached_response + "\n\n*[Cached response]*"
        else:
            # Create placeholder for streaming
            st.markdown("## 📊 Analysis Results")
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in generate_response_stream_hybrid(query, vectorstores, equipment_context=validated_context):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Cache and store the response
            cache.set(query, full_response, context_dict)
            st.session_state.analysis_result = full_response
        
        # Clear the stream flag
        st.session_state.stream_analysis = False
        st.rerun()

# Main application function with Pydantic validation
def main():
    """Main application with hybrid LLM and Pydantic validation"""
    
    # Initialize session state for page selection
    if "current_page" not in st.session_state:
        st.session_state.current_page = "ChatBot"
    
    # Page navigation in sidebar with vertical buttons
    st.sidebar.title("🛡️ SmartMCI App")
    
    st.sidebar.markdown("### 📱 Navigation")
    
    # Vertical navigation
    if st.sidebar.button(
        "💬 ChatBot", 
        use_container_width=True,
        type="primary" if st.session_state.current_page == "ChatBot" else "secondary",
        help="Ask questions with smart model selection"
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
    
    # Show Pydantic validation status
    st.sidebar.markdown("### 🔧 System Status")
    try:
        # Test API configuration validation
        api_config = APIConfig(
            groq_api_key=os.environ.get("GROQ_API_KEY", "test"),
            pinecone_api_key=os.environ.get("PINECONE_API_KEY", "test"),
            tavily_api_key=os.environ.get("TAVILY_API_KEY", "test")
        )
        st.sidebar.success("✅ Pydantic Validation Active")
    except ValidationError:
        st.sidebar.warning("⚠️ Some API keys missing")
    
    # Route to appropriate page
    if st.session_state.current_page == "ChatBot":
        chatbot_page()
    elif st.session_state.current_page == "Analysis":
        analysis_page()
    elif st.session_state.current_page == "Calculator":
        calculator_page()
    
    # Common footer
    st.markdown("---")
    st.caption("⚠️ **Engineering verification required** | Based on API 571/970/584 standards | 🔧 **Pydantic validation enabled**")

if __name__ == "__main__":
    main()