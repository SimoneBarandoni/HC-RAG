from IPython.display import Markdown, display
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, List, Mapping, Any

from llama_index.llms.openai_like import OpenAILike
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import PropertyGraphIndex

# LLM settings
LLM_KEY = "ScrYTIOnaDERtiEntioNfEyBOMpL"
LLM_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
LLM_TEMPERATURE = 0.1
LLM_URL = "https://proxy.rationalai.gkops.net/v1"

# Embedding model - using local HuggingFace model instead of OpenAI
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free local model
EMBEDDING_TYPE = "huggingface"  # Options: "huggingface", "openai"

llm = OpenAILike(
    api_key=LLM_KEY,
    api_base=LLM_URL,
    model=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
    is_chat_model=True,  # Set to True for chat-based models
)

# You can now set this LLM as the global default for LlamaIndex
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)

# Load your data
documents = SimpleDirectoryReader("data/IngestedDocuments").load_data()


index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=Settings.embed_model,
    show_progress=True,
)

index.property_graph_store.save_networkx_graph(name="./kg.html")

retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve("Do you have any vintage mountain bikes in stock?")

for node in nodes:
    print(node.text)

query_engine = index.as_query_engine(
    include_text=True
)

response = query_engine.query("Do you have any vintage mountain bikes in stock?")

print(f"{response.response}")