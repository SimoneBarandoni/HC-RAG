import random
import logging
from typing import List, Dict, TypedDict, Optional, Annotated, Literal
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode
from langgraph.constants import Send
import operator
import asyncio
import json
from datetime import datetime

# Import the relevance scoring system
from isRelevant import (
    isRelevant,
    batch_isRelevant,
    QueryInput,
    NodeInput,
    ScorerType,
    QueryIntent,
    CompositeWeights,
    DEFAULT_COMPOSITE_WEIGHTS,
)
import numpy as np
from openai import OpenAI
from pydantic import BaseModel as PydanticBaseModel, Field

# import configurations
from configurations import (
    OLLAMA_BASE_URL,
    OLLAMA_KEY,
    OLLAMA_MODEL,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_PASSWORD,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- 1. ENHANCED CONFIGURATION WITH LANGGRAPH FEATURES ---

# Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# OpenAI client configured for Ollama
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_KEY,
)

# LangGraph memory and persistence setup
memory_saver = MemorySaver()
long_term_store = InMemoryStore(
    index={
        "dims": 384,  # Embedding dimensions
        "embed": lambda texts: [np.random.rand(384).tolist() for _ in texts],  # Replace with real embeddings
    }
)

# Global configuration
CURRENT_SCORER_TYPE = ScorerType.COMPOSITE
CURRENT_COMPOSITE_WEIGHTS = DEFAULT_COMPOSITE_WEIGHTS
RANDOM_SEED = None
BATCH_SIZE = 10
MAX_RETRIES = 3
STREAM_ENABLED = True
HUMAN_IN_LOOP_ENABLED = True

# --- 2. ENHANCED STATE DEFINITION WITH MEMORY MANAGEMENT ---

class RAGState(TypedDict):
    # Core query processing
    question: str
    user_id: str
    session_id: str
    query_input: QueryInput
    
    # Node processing results
    sampled_nodes: List[Dict]
    semantic_scored_nodes: List[NodeInput]
    expanded_nodes: List[Dict]
    expanded_scored_nodes: List[NodeInput]
    final_relevant_nodes: List[NodeInput]
    expanded_subgraph: List[Dict]
    
    # Enhanced state management
    conversation_history: Annotated[List[Dict], operator.add]
    user_preferences: Dict  # Long-term memory
    context_summary: str  # Compressed context for long conversations
    
    # Flow control
    decision: Literal["sufficient", "revision", "human_review", "retry"]
    confidence_score: float
    retry_count: int
    
    # Streaming and feedback
    partial_results: List[str]
    human_feedback: Optional[str]
    
    # Final outputs
    final_answer: str
    metadata: Dict
    
    # Memory and optimization
    token_usage: Dict[str, int]
    processing_time: float
    memory_summary: str

# --- 3. MEMORY MANAGEMENT FUNCTIONS ---

def load_user_preferences(user_id: str, store: InMemoryStore) -> Dict:
    """Load user preferences from long-term memory."""
    try:
        namespace = ("user_preferences", user_id)
        items = store.search(namespace, query="preferences")
        if items:
            return items[0].value
        return {}
    except Exception as e:
        logger.warning(f"Failed to load user preferences: {e}")
        return {}

def save_user_preferences(user_id: str, preferences: Dict, store: InMemoryStore):
    """Save user preferences to long-term memory."""
    try:
        namespace = ("user_preferences", user_id)
        store.put(namespace, "preferences", preferences)
    except Exception as e:
        logger.warning(f"Failed to save user preferences: {e}")

def manage_conversation_memory(state: RAGState) -> Dict:
    """Manage conversation memory to avoid token limits."""
    history = state.get("conversation_history", [])
    
    # If conversation is getting long, summarize older messages
    if len(history) > 10:
        # Summarize everything except the last 5 messages
        messages_to_summarize = history[:-5]
        recent_messages = history[-5:]
        
        # Create summary (simplified - use LLM in production)
        summary = f"Previous conversation summary: {len(messages_to_summarize)} messages about {state['question']}"
        
        return {
            "context_summary": summary,
            "conversation_history": recent_messages,
            "memory_summary": f"Compressed {len(messages_to_summarize)} messages"
        }
    
    return {"memory_summary": "No compression needed"}

# --- 4. ENHANCED LLM FUNCTIONS WITH STREAMING ---

async def call_ollama_llm_streaming(
    system_prompt: str, 
    user_prompt: str, 
    response_format=None,
    timeout=30,
    stream_callback=None
) -> str:
    """Enhanced LLM call with streaming support."""
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_KEY,
            timeout=timeout,
        )

        if STREAM_ENABLED and stream_callback:
            # Streaming mode
            stream = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                timeout=timeout,
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    if stream_callback:
                        stream_callback(content)
            
            return full_response
        else:
            # Regular mode
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=timeout,
            )
            return response.choices[0].message.content

    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return "Error: LLM timeout or failure"

# --- 5. ENHANCED GRAPH NODES WITH OPTIMIZATION ---

def initialize_session(state: RAGState) -> Dict:
    """Initialize session with user preferences and memory."""
    user_id = state.get("user_id", "anonymous")
    session_id = state.get("session_id", f"session_{datetime.now().isoformat()}")
    
    # Load user preferences from long-term memory
    preferences = load_user_preferences(user_id, long_term_store)
    
    # Initialize conversation history
    conversation_history = [{
        "role": "user",
        "content": state["question"],
        "timestamp": datetime.now().isoformat()
    }]
    
    return {
        "user_id": user_id,
        "session_id": session_id,
        "user_preferences": preferences,
        "conversation_history": conversation_history,
        "retry_count": 0,
        "confidence_score": 0.0,
        "token_usage": {"input": 0, "output": 0},
        "processing_time": 0.0,
        "metadata": {
            "session_start": datetime.now().isoformat(),
            "user_preferences_loaded": bool(preferences)
        }
    }

def analyze_query_with_context(state: RAGState) -> Dict:
    """Enhanced query analysis with context and user preferences."""
    question = state["question"]
    user_preferences = state.get("user_preferences", {})
    context_summary = state.get("context_summary", "")
    
    # Incorporate user preferences into query analysis
    enhanced_question = question
    if user_preferences:
        pref_context = f"User preferences: {json.dumps(user_preferences)}"
        enhanced_question = f"{question}\n\nContext: {pref_context}"
    
    if context_summary:
        enhanced_question += f"\n\nConversation context: {context_summary}"
    
    query_input = create_query_input(enhanced_question)
    
    # Manage conversation memory
    memory_update = manage_conversation_memory(state)
    
    result = {"query_input": query_input}
    result.update(memory_update)
    
    return result

def sample_neo4j_nodes_with_retry(state: RAGState) -> Dict:
    """Sample nodes with retry logic and error recovery."""
    retry_count = state.get("retry_count", 0)
    
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    try:
        sampled_nodes = sample_random_nodes_from_neo4j(limit=20)
        
        if not sampled_nodes and retry_count < MAX_RETRIES:
            logger.warning(f"No nodes sampled, retry {retry_count + 1}/{MAX_RETRIES}")
            return {
                "decision": "retry",
                "retry_count": retry_count + 1,
                "sampled_nodes": []
            }
        
        return {
            "sampled_nodes": sampled_nodes,
            "decision": "sufficient" if sampled_nodes else "revision"
        }
        
    except Exception as e:
        logger.error(f"Error sampling nodes: {e}")
        if retry_count < MAX_RETRIES:
            return {
                "decision": "retry",
                "retry_count": retry_count + 1,
                "sampled_nodes": []
            }
        else:
            return {
                "decision": "revision",
                "sampled_nodes": [],
                "metadata": {"error": str(e)}
            }

def score_semantic_similarity_optimized(state: RAGState) -> Dict:
    """Optimized semantic similarity scoring with batch processing."""
    sampled_nodes = state["sampled_nodes"]
    query_input = state["query_input"]
    
    if not sampled_nodes:
        return {"semantic_scored_nodes": [], "confidence_score": 0.0}
    
    # Convert nodes to NodeInput format
    candidate_nodes = []
    for neo4j_node in sampled_nodes:
        node_input = convert_neo4j_node_to_node_input(neo4j_node)
        candidate_nodes.append(node_input)
    
    try:
        # Batch processing for efficiency
        from isRelevant import batch_semantic_similarity
        
        similarity_scores = batch_semantic_similarity(query_input, candidate_nodes)
        
        # Assign scores and calculate confidence
        scored_nodes = []
        for node, score in zip(candidate_nodes, similarity_scores):
            node.score = score
            scored_nodes.append(node)
        
        # Sort and filter by threshold
        sorted_nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)
        semantic_scored_nodes = [node for node in sorted_nodes if node.score >= 0.60]
        
        # Calculate confidence score
        confidence = np.mean(similarity_scores) if similarity_scores else 0.0
        
        logger.info(f"Scored {len(semantic_scored_nodes)} nodes with confidence {confidence:.3f}")
        
        return {
            "semantic_scored_nodes": semantic_scored_nodes,
            "confidence_score": confidence
        }
        
    except Exception as e:
        logger.error(f"Semantic similarity scoring failed: {e}")
        return {
            "semantic_scored_nodes": [],
            "confidence_score": 0.0,
            "decision": "retry" if state.get("retry_count", 0) < MAX_RETRIES else "revision"
        }

def expand_subgraph_parallel(state: RAGState) -> Dict:
    """Parallel subgraph expansion for better performance."""
    semantic_scored_nodes = state["semantic_scored_nodes"]
    
    # Use parallel processing for subgraph expansion
    expanded_subgraph = []
    expanded_nodes = []
    
    try:
        with neo4j_driver.session() as session:
            for node in semantic_scored_nodes:
                # Expand each node's subgraph
                node_expansion = expand_single_node(session, node)
                expanded_subgraph.extend(node_expansion["relations"])
                expanded_nodes.extend(node_expansion["nodes"])
        
        return {
            "expanded_subgraph": expanded_subgraph,
            "expanded_nodes": expanded_nodes
        }
        
    except Exception as e:
        logger.error(f"Subgraph expansion failed: {e}")
        return {
            "expanded_subgraph": [],
            "expanded_nodes": [],
            "decision": "retry" if state.get("retry_count", 0) < MAX_RETRIES else "revision"
        }

def evaluate_context_intelligent(state: RAGState) -> Dict:
    """Intelligent context evaluation with confidence scoring."""
    final_relevant_nodes = state.get("final_relevant_nodes", [])
    confidence_score = state.get("confidence_score", 0.0)
    retry_count = state.get("retry_count", 0)
    user_preferences = state.get("user_preferences", {})
    
    # Prevent infinite loops
    if retry_count >= MAX_RETRIES:
        return {"decision": "sufficient"}
    
    # Check if human review is needed based on confidence
    if HUMAN_IN_LOOP_ENABLED and confidence_score < 0.3:
        return {"decision": "human_review"}
    
    # Evaluate based on multiple factors
    high_relevance_nodes = [n for n in final_relevant_nodes if getattr(n, "score", 0) > 0.7]
    
    # Consider user preferences for evaluation
    quality_threshold = user_preferences.get("quality_threshold", 0.6)
    
    if len(high_relevance_nodes) >= 1 and confidence_score >= quality_threshold:
        return {"decision": "sufficient"}
    else:
        return {"decision": "revision"}

def human_review_node(state: RAGState) -> Dict:
    """Human-in-the-loop review node."""
    logger.info("Human review requested due to low confidence")
    
    # In a real implementation, this would pause execution
    # and wait for human input via UI or API
    
    # For demo purposes, we'll simulate human feedback
    human_feedback = "Please try with different search terms"
    
    return {
        "human_feedback": human_feedback,
        "decision": "revision"
    }

def generate_answer_streaming(state: RAGState) -> Dict:
    """Generate answer with streaming support."""
    final_relevant_nodes = state.get("final_relevant_nodes", [])
    query_input = state["query_input"]
    user_preferences = state.get("user_preferences", {})
    context_summary = state.get("context_summary", "")
    
    # Prepare context
    context_text = "\n\n".join([
        f"Resource {i + 1} (relevance: {getattr(node, 'score', 0):.3f}):\n{node.text}"
        for i, node in enumerate(final_relevant_nodes)
    ])
    
    # Include user preferences in system prompt
    user_context = ""
    if user_preferences:
        user_context = f"\n\nUser preferences: {json.dumps(user_preferences)}"
    
    if context_summary:
        user_context += f"\n\nConversation context: {context_summary}"
    
    system_prompt = f"""You are an AI assistant that uses a knowledge graph to answer questions. 
You specialize in {query_input.intent.value}.

{user_context}

Instructions:
1. Use only information from the provided context
2. Personalize responses based on user preferences
3. Indicate confidence level in your response
4. Structure the response clearly and helpfully
5. Mention that information comes from a knowledge graph"""
    
    user_prompt = f"""Question: {state["question"]}
Intent: {query_input.intent.value}
Relevant entities: {", ".join(query_input.entities)}

--- CONTEXT FROM KNOWLEDGE GRAPH ---
{context_text}
--- END CONTEXT ---

Provide a complete and accurate answer based on the knowledge graph:"""
    
    # Collect streaming tokens
    partial_results = []
    
    def stream_callback(token: str):
        partial_results.append(token)
        # In a real implementation, you'd send these to the UI
        logger.debug(f"Streaming token: {token}")
    
    try:
        if STREAM_ENABLED:
            answer = asyncio.run(call_ollama_llm_streaming(
                system_prompt, user_prompt, stream_callback=stream_callback
            ))
        else:
            answer = call_ollama_llm(system_prompt, user_prompt)
        
        # Update conversation history
        conversation_update = {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().isoformat(),
            "confidence": state.get("confidence_score", 0.0)
        }
        
        # Save user interaction patterns for future optimization
        interaction_data = {
            "question": state["question"],
            "answer": answer,
            "confidence": state.get("confidence_score", 0.0),
            "nodes_used": len(final_relevant_nodes),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update user preferences based on interaction
        updated_preferences = update_user_preferences(
            state.get("user_id", "anonymous"),
            interaction_data,
            user_preferences
        )
        
        return {
            "final_answer": answer,
            "partial_results": partial_results,
            "conversation_history": [conversation_update],
            "user_preferences": updated_preferences,
            "metadata": {
                "streaming_enabled": STREAM_ENABLED,
                "tokens_streamed": len(partial_results),
                "answer_length": len(answer)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {
            "final_answer": f"I apologize, but I encountered an error: {str(e)}",
            "partial_results": [],
            "decision": "retry" if state.get("retry_count", 0) < MAX_RETRIES else "sufficient"
        }

def update_user_preferences(user_id: str, interaction_data: Dict, current_preferences: Dict) -> Dict:
    """Update user preferences based on interaction patterns."""
    updated_preferences = current_preferences.copy()
    
    # Update quality threshold based on user interactions
    if interaction_data["confidence"] > 0.8:
        updated_preferences["quality_threshold"] = min(
            updated_preferences.get("quality_threshold", 0.6) + 0.01, 0.9
        )
    elif interaction_data["confidence"] < 0.3:
        updated_preferences["quality_threshold"] = max(
            updated_preferences.get("quality_threshold", 0.6) - 0.01, 0.1
        )
    
    # Track interaction patterns
    updated_preferences["last_interaction"] = interaction_data["timestamp"]
    updated_preferences["total_interactions"] = updated_preferences.get("total_interactions", 0) + 1
    
    # Save to long-term memory
    save_user_preferences(user_id, updated_preferences, long_term_store)
    
    return updated_preferences

# --- 6. ENHANCED HELPER FUNCTIONS ---

def expand_single_node(session, node: NodeInput) -> Dict:
    """Expand a single node's subgraph."""
    relations = []
    nodes = []
    
    # Implementation similar to original but with better error handling
    try:
        node_id = None
        if "product_id" in node.graph_relations:
            node_id = node.graph_relations["product_id"]
            query = """
            MATCH (p:Product {product_id: $node_id})-[r]-(connected)
            RETURN p, type(r) as relation_type, connected, labels(connected) as connected_labels
            LIMIT 3
            """
        elif "_id" in node.graph_relations:
            node_id = node.graph_relations["_id"]
            query = """
            MATCH (n)-[r]-(connected)
            WHERE id(n) = $node_id
            RETURN n, type(r) as relation_type, connected, labels(connected) as connected_labels
            LIMIT 3
            """
        
        if node_id:
            results = session.run(query, node_id=node_id)
            for record in results:
                relations.append(dict(record))
                connected_node = dict(record["connected"])
                connected_node["labels"] = record["connected_labels"]
                nodes.append(connected_node)
    
    except Exception as e:
        logger.debug(f"Error expanding node: {e}")
    
    return {"relations": relations, "nodes": nodes}

# Helper functions from original code (simplified)
def create_query_input(question: str) -> QueryInput:
    """Create structured QueryInput from user question."""
    from isRelevant import QueryIntent
    
    intent = QueryIntent.PRODUCT_SEARCH  # Simplified
    entities = []  # Simplified
    embeddings = np.random.rand(384)  # Replace with real embeddings
    
    return QueryInput(
        text=question,
        embeddings=embeddings,
        entities=entities,
        intent=intent
    )

def sample_random_nodes_from_neo4j(limit: int = 20) -> List[Dict]:
    """Sample random nodes from Neo4j graph."""
    try:
        with neo4j_driver.session() as session:
            query = """
            MATCH (n)
            RETURN n, labels(n) as node_labels, id(n) as node_id
            ORDER BY rand()
            LIMIT $limit
            """
            results = session.run(query, limit=limit)
            nodes = []
            for record in results:
                node = dict(record["n"])
                node["_labels"] = record["node_labels"]
                node["_id"] = record["node_id"]
                nodes.append(node)
            return nodes
    except Exception as e:
        logger.error(f"Error sampling nodes: {e}")
        return []

def convert_neo4j_node_to_node_input(neo4j_node: Dict) -> NodeInput:
    """Convert a Neo4j node to NodeInput for isRelevant."""
    # Simplified version of the original function
    text = str(neo4j_node.get("name", "Unknown"))
    embeddings = np.random.rand(384)
    labels = neo4j_node.get("_labels", ["Unknown"])
    node_type = labels[0].lower() if labels else "unknown"
    
    node = NodeInput(
        text=text,
        embeddings=embeddings,
        graph_relations=neo4j_node,
        node_type=node_type,
        entities=[]
    )
    node.score = 0.0
    return node

def call_ollama_llm(system_prompt: str, user_prompt: str, response_format=None, timeout=30) -> str:
    """Synchronous LLM call for compatibility."""
    try:
        response = ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return "Error: LLM call failed"

# --- 7. ENHANCED WORKFLOW CONSTRUCTION ---

def build_optimized_workflow():
    """Build the optimized LangGraph workflow."""
    workflow = StateGraph(RAGState)
    
    # Add nodes with enhanced capabilities
    workflow.add_node("initialize_session", initialize_session)
    workflow.add_node("analyze_query", analyze_query_with_context)
    workflow.add_node("sample_nodes", sample_neo4j_nodes_with_retry)
    workflow.add_node("score_semantic", score_semantic_similarity_optimized)
    workflow.add_node("expand_subgraph", expand_subgraph_parallel)
    workflow.add_node("score_all_nodes", score_expanded_nodes_with_isrelevant)
    workflow.add_node("evaluate_context", evaluate_context_intelligent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("revise_question", revise_question)
    workflow.add_node("generate_answer", generate_answer_streaming)
    
    # Enhanced flow with conditional routing
    workflow.set_entry_point("initialize_session")
    workflow.add_edge("initialize_session", "analyze_query")
    workflow.add_edge("analyze_query", "sample_nodes")
    
    # Conditional edge for retry logic
    workflow.add_conditional_edges(
        "sample_nodes",
        lambda x: x.get("decision", "sufficient"),
        {
            "sufficient": "score_semantic",
            "retry": "sample_nodes",
            "revision": "revise_question"
        }
    )
    
    workflow.add_edge("score_semantic", "expand_subgraph")
    workflow.add_edge("expand_subgraph", "score_all_nodes")
    workflow.add_edge("score_all_nodes", "evaluate_context")
    
    # Enhanced conditional routing
    workflow.add_conditional_edges(
        "evaluate_context",
        lambda x: x.get("decision", "sufficient"),
        {
            "sufficient": "generate_answer",
            "revision": "revise_question",
            "human_review": "human_review",
            "retry": "sample_nodes"
        }
    )
    
    workflow.add_edge("human_review", "revise_question")
    workflow.add_edge("revise_question", "analyze_query")
    workflow.add_edge("generate_answer", END)
    
    return workflow

# Original functions needed for compatibility
def score_expanded_nodes_with_isrelevant(state: RAGState) -> Dict:
    """Score expanded nodes with isRelevant."""
    # Simplified version of original function
    semantic_scored_nodes = state.get("semantic_scored_nodes", [])
    expanded_nodes = state.get("expanded_nodes", [])
    
    # Convert and combine nodes
    all_nodes = semantic_scored_nodes.copy()
    for node in expanded_nodes:
        all_nodes.append(convert_neo4j_node_to_node_input(node))
    
    # Sort by existing scores and take top 15
    final_relevant_nodes = sorted(all_nodes, key=lambda x: getattr(x, "score", 0), reverse=True)[:15]
    
    return {
        "expanded_scored_nodes": [convert_neo4j_node_to_node_input(n) for n in expanded_nodes],
        "final_relevant_nodes": final_relevant_nodes
    }

def revise_question(state: RAGState) -> Dict:
    """Revise question based on previous attempts."""
    current_question = state["question"]
    retry_count = state.get("retry_count", 0)
    human_feedback = state.get("human_feedback", "")
    
    # Simple question revision
    if human_feedback:
        revised_question = f"{current_question} (considering: {human_feedback})"
    else:
        revised_question = f"Please provide information about: {current_question}"
    
    return {
        "question": revised_question,
        "retry_count": retry_count + 1,
        "decision": "sufficient"
    }

# --- 8. MAIN EXECUTION WITH ENHANCED FEATURES ---

def main():
    """Main execution with enhanced LangGraph features."""
    
    # Build the optimized workflow
    workflow = build_optimized_workflow()
    
    # Compile with memory and checkpointing
    app = workflow.compile(
        checkpointer=memory_saver,
        store=long_term_store
    )
    
    # Example usage with enhanced features
    user_question = "What mountain bikes do you have under $500?"
    user_id = "user_123"
    
    # Configuration for this session
    config = {
        "configurable": {
            "thread_id": f"thread_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "checkpoint_ns": f"user_{user_id}",
        }
    }
    
    inputs = {
        "question": user_question,
        "user_id": user_id,
        "session_id": f"session_{datetime.now().isoformat()}",
        "conversation_history": [],
        "user_preferences": {},
        "retry_count": 0,
        "confidence_score": 0.0,
        "sampled_nodes": [],
        "semantic_scored_nodes": [],
        "expanded_nodes": [],
        "final_relevant_nodes": [],
        "expanded_subgraph": [],
        "partial_results": [],
        "token_usage": {"input": 0, "output": 0},
        "processing_time": 0.0,
        "metadata": {}
    }
    
    try:
        print("🚀 Starting Enhanced RAG System with LangGraph...")
        print(f"User: {user_id}")
        print(f"Question: {user_question}")
        print("-" * 50)
        
        # Execute with streaming support
        if STREAM_ENABLED:
            print("📡 Streaming mode enabled")
            for event in app.stream(inputs, config=config):
                for node, output in event.items():
                    print(f"🔄 Node: {node}")
                    if "partial_results" in output and output["partial_results"]:
                        print(f"📝 Partial output: {''.join(output['partial_results'][-3:])}")
        else:
            final_state = app.invoke(inputs, config=config)
            
            print("\n✅ FINAL ANSWER:")
            print("-" * 40)
            print(final_state.get("final_answer", "No answer generated"))
            
            # Show enhanced statistics
            print("\n📊 ENHANCED STATISTICS:")
            print(f"• Confidence Score: {final_state.get('confidence_score', 0):.3f}")
            print(f"• Retry Count: {final_state.get('retry_count', 0)}")
            print(f"• Memory Summary: {final_state.get('memory_summary', 'N/A')}")
            print(f"• User Preferences Updated: {bool(final_state.get('user_preferences'))}")
            print(f"• Nodes Processed: {len(final_state.get('final_relevant_nodes', []))}")
            print(f"• Human Review Needed: {final_state.get('decision') == 'human_review'}")
            
            # Show conversation history
            history = final_state.get("conversation_history", [])
            if history:
                print(f"• Conversation Length: {len(history)} messages")
            
            # Show metadata
            metadata = final_state.get("metadata", {})
            if metadata:
                print(f"• Additional Metadata: {json.dumps(metadata, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.exception("Full traceback:")
    
    finally:
        # Cleanup
        neo4j_driver.close()
        print("\n🔚 Session ended")

if __name__ == "__main__":
    main() 