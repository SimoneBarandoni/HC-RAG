"""
Configuration for Property Graph Builder
=======================================

Customize these settings to control how your graph is built without editing the main code.
"""

import os
from typing import Optional, List

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Path to your data - can be a directory or single file
DATA_PATH = "data"

# File extensions to include (None = all supported files)
# Examples: ['.csv', '.json', '.txt'], ['.pdf', '.md'], None
FILE_EXTENSIONS = None

# =============================================================================
# GRAPH EXTRACTION CONFIGURATION
# =============================================================================

# Extraction mode: how to discover entities and relationships
# Options: "auto", "simple", "dynamic", "implicit"
# - "auto": Uses both simple and dynamic extraction
# - "simple": Basic LLM-based extraction
# - "dynamic": Discovers schema automatically
# - "implicit": Uses embeddings for implicit relationships
EXTRACTION_MODE = "auto"

# =============================================================================
# BACKEND CONFIGURATION
# =============================================================================

# Whether to use Neo4j as the graph backend
USE_NEO4J = False

# Neo4j connection settings (only used if USE_NEO4J=True)
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# OpenAI API settings
OPENAI_API_KEY = "ScrYTIOnaDERtiEntioNfEyBOMpL"
LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LLM_TEMPERATURE = 0.1
LLM_URL = "https://proxy.rationalai.gkops.net/v1"

# Embedding model - using local HuggingFace model instead of OpenAI
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free local model
EMBEDDING_TYPE = "huggingface"  # Options: "huggingface", "openai"

# =============================================================================
# EXTRACTION PARAMETERS
# =============================================================================

# Maximum number of entity/relationship paths to extract per text chunk
MAX_PATHS_PER_CHUNK = 10

# Number of worker threads for extraction
NUM_WORKERS = 1

# Text chunking parameters
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# =============================================================================
# QUERY CONFIGURATION
# =============================================================================

# Default similarity top-k for vector queries
SIMILARITY_TOP_K = 10

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Where to save the graph (used for non-Neo4j backends)
OUTPUT_PATH = "graph_output"

# Whether to show progress bars during processing
SHOW_PROGRESS = True

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# CSV processing settings
CSV_SEPARATORS = [',', ';', '\t', '|']
CSV_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

# Maximum number of sample items to show in graph exploration
MAX_SAMPLE_ITEMS = 5

# Whether to include text content in graph nodes
INCLUDE_TEXT_IN_NODES = True

# =============================================================================
# VALIDATION AND HELPER FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if not os.path.exists(DATA_PATH):
        errors.append(f"DATA_PATH '{DATA_PATH}' does not exist")
    
    if EXTRACTION_MODE not in ["auto", "simple", "dynamic", "implicit"]:
        errors.append(f"EXTRACTION_MODE '{EXTRACTION_MODE}' is not valid")
    
    if USE_NEO4J and not all([NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD]):
        errors.append("Neo4j settings are incomplete")
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Create a .env file with your API key.")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    print("✅ Configuration is valid")
    return True

def get_file_extensions() -> Optional[List[str]]:
    """Get file extensions, ensuring they start with a dot."""
    if FILE_EXTENSIONS is None:
        return None
    
    extensions = []
    for ext in FILE_EXTENSIONS:
        if not ext.startswith('.'):
            ext = '.' + ext
        extensions.append(ext.lower())
    
    return extensions

def print_config():
    """Print current configuration."""
    print("🔧 Current Configuration:")
    print(f"  • Data Path: {DATA_PATH}")
    print(f"  • File Extensions: {get_file_extensions()}")
    print(f"  • Extraction Mode: {EXTRACTION_MODE}")
    print(f"  • Use Neo4j: {USE_NEO4J}")
    print(f"  • LLM Model: {LLM_MODEL}")
    print(f"  • Embedding Model: {EMBEDDING_MODEL} ({EMBEDDING_TYPE})")
    print(f"  • Output Path: {OUTPUT_PATH}")

if __name__ == "__main__":
    print_config()
    validate_config() 