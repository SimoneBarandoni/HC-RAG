"""
Property Graph Query Interface
=============================

A flexible query interface for LlamaIndex property graphs that works with any
data structure without hardcoding entity types or relationships.
"""

import os
import pickle
from typing import List, Dict, Any, Optional, Union
from llama_index.core import PropertyGraphIndex
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.query_engine import PropertyGraphQueryEngine
from llama_index.core.retrievers import (
    VectorPropertyGraphRetriever,
    TextToCypherRetriever,
    KnowledgeGraphRAGRetriever
)
from llama_index.core.retrievers.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever
)
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
import requests
from typing import Any
from dotenv import load_dotenv

load_dotenv()


class CustomLLMClient(CustomLLM):
    """Custom LLM client for your deployed model endpoint."""
    
    context_window: int = 32768
    num_output: int = 2048
    model_name: str = "custom"
    api_url: str = ""
    api_key: str = ""
    
    def __init__(self, api_url: str, model_name: str, api_key: str = "", **kwargs):
        super().__init__(
            api_url=api_url,
            model_name=model_name,
            api_key=api_key,
            **kwargs
        )
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Send completion request to your custom endpoint."""
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # OpenAI-compatible API format
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            return CompletionResponse(text=text)
            
        except Exception as e:
            print(f"Error calling custom LLM: {e}")
            return CompletionResponse(text=f"Error: {str(e)}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion from your custom endpoint."""
        # For now, just use the complete method and yield the full response
        # You can implement proper streaming later if your endpoint supports it
        response = self.complete(prompt, **kwargs)
        yield response


class QueryInterface:
    """Query interface for property graphs."""
    
    def __init__(self, index: Optional[PropertyGraphIndex] = None, 
                 graph_store_path: Optional[str] = None):
        """Initialize the query interface.
        
        Args:
            index: Pre-built PropertyGraphIndex
            graph_store_path: Path to saved graph store pickle file
        """
        # Import configuration
        import config
        
        # Configure settings - use custom LLM
        Settings.llm = CustomLLMClient(
            api_url=config.LLM_URL,
            model_name=config.LLM_MODEL,
            api_key=config.OPENAI_API_KEY
        )
        
        # Configure embedding model based on type
        if config.EMBEDDING_TYPE == "huggingface":
            Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        else:
            Settings.embed_model = OpenAIEmbedding(model=config.EMBEDDING_MODEL)
        
        if index:
            self.index = index
        elif graph_store_path:
            self.index = self._load_graph(graph_store_path)
        else:
            raise ValueError("Either index or graph_store_path must be provided")
        
        # Create different query engines
        self.query_engines = self._create_query_engines()
    
    def _load_graph(self, graph_store_path: str) -> PropertyGraphIndex:
        """Load a saved graph."""
        try:
            with open(graph_store_path, "rb") as f:
                graph_store = pickle.load(f)
            
            # Reconstruct the index
            index = PropertyGraphIndex.from_existing(
                property_graph_store=graph_store,
                embed_model=Settings.embed_model
            )
            return index
        except Exception as e:
            raise ValueError(f"Failed to load graph from {graph_store_path}: {e}")
    
    def _create_query_engines(self) -> Dict[str, PropertyGraphQueryEngine]:
        """Create different types of query engines."""
        engines = {}
        
        # 1. Vector-based retrieval
        try:
            vector_retriever = VectorPropertyGraphRetriever(
                self.index.property_graph_store,
                embed_model=Settings.embed_model,
                similarity_top_k=10
            )
            engines["vector"] = PropertyGraphQueryEngine(
                retriever=vector_retriever,
                llm=Settings.llm
            )
        except Exception as e:
            print(f"Warning: Could not create vector query engine: {e}")
        
        # 2. Keyword/synonym-based retrieval
        try:
            synonym_retriever = LLMSynonymRetriever(
                self.index.property_graph_store,
                llm=Settings.llm,
                include_text=True
            )
            engines["keyword"] = PropertyGraphQueryEngine(
                retriever=synonym_retriever,
                llm=Settings.llm
            )
        except Exception as e:
            print(f"Warning: Could not create keyword query engine: {e}")
        
        # 3. Context-aware retrieval
        try:
            context_retriever = VectorContextRetriever(
                self.index.property_graph_store,
                embed_model=Settings.embed_model,
                similarity_top_k=10
            )
            engines["context"] = PropertyGraphQueryEngine(
                retriever=context_retriever,
                llm=Settings.llm
            )
        except Exception as e:
            print(f"Warning: Could not create context query engine: {e}")
        
        # 4. Default engine from the index
        try:
            engines["default"] = self.index.as_query_engine()
        except Exception as e:
            print(f"Warning: Could not create default query engine: {e}")
        
        if not engines:
            raise ValueError("No query engines could be created")
        
        return engines
    
    def query(self, question: str, mode: str = "auto") -> Dict[str, Any]:
        """Query the graph with automatic mode selection.
        
        Args:
            question: The question to ask
            mode: Query mode ('auto', 'vector', 'keyword', 'context', 'default')
        
        Returns:
            Dictionary with response and metadata
        """
        print(f"🔍 Querying: {question}")
        print(f"📊 Mode: {mode}")
        
        # Auto mode: try different engines based on query type
        if mode == "auto":
            mode = self._select_best_mode(question)
            print(f"🎯 Auto-selected mode: {mode}")
        
        # Get the appropriate engine
        engine = self.query_engines.get(mode)
        if not engine:
            print(f"⚠️  Mode '{mode}' not available, using default")
            engine = self.query_engines.get("default")
            if not engine:
                engine = list(self.query_engines.values())[0]
        
        try:
            response = engine.query(question)
            
            result = {
                "question": question,
                "answer": str(response),
                "mode": mode,
                "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0,
                "metadata": response.metadata if hasattr(response, 'metadata') else {}
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return {
                "question": question,
                "answer": f"Query failed: {e}",
                "mode": mode,
                "source_nodes": 0,
                "metadata": {}
            }
    
    def _select_best_mode(self, question: str) -> str:
        """Select the best query mode based on question type."""
        question_lower = question.lower()
        
        # Vector mode for semantic/similarity questions
        if any(word in question_lower for word in 
               ["similar", "like", "related", "compare", "difference", "semantic"]):
            return "vector"
        
        # Keyword mode for specific entity searches
        if any(word in question_lower for word in 
               ["find", "search", "locate", "show", "list", "what is"]):
            return "keyword"
        
        # Context mode for complex analytical questions
        if any(word in question_lower for word in 
               ["analyze", "explain", "why", "how", "relationship", "pattern"]):
            return "context"
        
        # Default for general questions
        return "default"
    
    def explore_graph(self) -> Dict[str, Any]:
        """Explore the graph structure to understand what's available."""
        print("🔍 Exploring graph structure...")
        
        try:
            # Get graph store
            graph_store = self.index.property_graph_store
            
            # Get basic statistics
            stats = {
                "total_nodes": 0,
                "total_relationships": 0,
                "node_types": set(),
                "relationship_types": set(),
                "sample_nodes": [],
                "sample_relationships": []
            }
            
            # Try to get nodes and relationships
            try:
                # For SimplePropertyGraphStore
                if hasattr(graph_store, 'graph'):
                    nodes = graph_store.graph.nodes(data=True)
                    edges = graph_store.graph.edges(data=True)
                    
                    stats["total_nodes"] = len(nodes)
                    stats["total_relationships"] = len(edges)
                    
                    # Sample nodes
                    for i, (node_id, node_data) in enumerate(nodes):
                        if i < 5:  # Get first 5 nodes
                            stats["sample_nodes"].append({
                                "id": node_id,
                                "type": node_data.get("type", "unknown"),
                                "properties": {k: v for k, v in node_data.items() if k != "type"}
                            })
                        
                        # Collect node types
                        if "type" in node_data:
                            stats["node_types"].add(node_data["type"])
                    
                    # Sample relationships
                    for i, (src, dst, edge_data) in enumerate(edges):
                        if i < 5:  # Get first 5 relationships
                            stats["sample_relationships"].append({
                                "source": src,
                                "target": dst,
                                "type": edge_data.get("type", "unknown"),
                                "properties": {k: v for k, v in edge_data.items() if k != "type"}
                            })
                        
                        # Collect relationship types
                        if "type" in edge_data:
                            stats["relationship_types"].add(edge_data["type"])
            
            except Exception as e:
                print(f"Could not access graph structure directly: {e}")
                # Try alternative approaches
                stats["error"] = str(e)
            
            # Convert sets to lists for JSON serialization
            stats["node_types"] = list(stats["node_types"])
            stats["relationship_types"] = list(stats["relationship_types"])
            
            return stats
            
        except Exception as e:
            print(f"❌ Error exploring graph: {e}")
            return {"error": str(e)}
    
    def suggest_queries(self, graph_stats: Optional[Dict] = None) -> List[str]:
        """Suggest example queries based on graph structure."""
        if not graph_stats:
            graph_stats = self.explore_graph()
        
        suggestions = []
        
        # Generic suggestions that work with any graph
        suggestions.extend([
            "What are the main entities in this data?",
            "What relationships exist between different entities?",
            "Show me the most important connections in the graph",
            "What patterns can you identify in the data?",
            "Summarize the key information in this dataset"
        ])
        
        # Suggestions based on node types
        node_types = graph_stats.get("node_types", [])
        if node_types:
            for node_type in node_types[:3]:  # Top 3 node types
                suggestions.append(f"Tell me about {node_type} entities")
                suggestions.append(f"What are the properties of {node_type}?")
        
        # Suggestions based on relationship types
        rel_types = graph_stats.get("relationship_types", [])
        if rel_types:
            for rel_type in rel_types[:3]:  # Top 3 relationship types
                suggestions.append(f"Show me examples of {rel_type} relationships")
        
        return suggestions
    
    def interactive_query(self):
        """Start an interactive query session."""
        print("🚀 Property Graph Query Interface")
        print("=" * 50)
        
        # First, explore the graph
        stats = self.explore_graph()
        print(f"\n📊 Graph Statistics:")
        print(f"  • Nodes: {stats.get('total_nodes', 'unknown')}")
        print(f"  • Relationships: {stats.get('total_relationships', 'unknown')}")
        print(f"  • Node Types: {', '.join(stats.get('node_types', ['unknown']))}")
        print(f"  • Relationship Types: {', '.join(stats.get('relationship_types', ['unknown']))}")
        
        # Show suggested queries
        suggestions = self.suggest_queries(stats)
        print(f"\n💡 Suggested Queries:")
        for i, suggestion in enumerate(suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        print(f"\n🎯 Available Modes: {', '.join(self.query_engines.keys())}")
        print("📝 Type your questions (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                # Check if user specified a mode
                mode = "auto"
                if question.startswith("mode:"):
                    parts = question.split(":", 1)
                    if len(parts) == 2:
                        mode = parts[0].replace("mode", "").strip()
                        question = parts[1].strip()
                
                # Query the graph
                result = self.query(question, mode)
                
                print(f"\n✅ Answer ({result['mode']} mode):")
                print(f"   {result['answer']}")
                print(f"   📎 Sources: {result['source_nodes']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Demo the query interface."""
    
    # Import configuration
    import config
    
    # Try to load a saved graph
    GRAPH_PATH = os.path.join(config.OUTPUT_PATH, "graph_store.pkl")
    
    if os.path.exists(GRAPH_PATH):
        print("📂 Loading saved graph...")
        query_interface = QueryInterface(graph_store_path=GRAPH_PATH)
    else:
        print("⚠️  No saved graph found. Please run graph_builder.py first.")
        print(f"Expected path: {GRAPH_PATH}")
        return
    
    # Start interactive session
    query_interface.interactive_query()

if __name__ == "__main__":
    main() 