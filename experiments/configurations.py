from openai import OpenAI

# --- 1. CONFIGURATION ---
# Configuration for gemma3:1b via Ollama
OLLAMA_BASE_URL = "https://proxy.rationalai.gkops.net/v1"#"http://localhost:11434/v1"
OLLAMA_KEY = "ScrYTIOnaDERtiEntioNfEyBOMpL"
OLLAMA_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"

# OpenAI client configured for Ollama
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_KEY,
)

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"