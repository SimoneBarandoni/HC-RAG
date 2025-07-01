import random
import logging
from typing import List, Dict, TypedDict, Optional
from neo4j import GraphDatabase
from langgraph.graph import StateGraph, END

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

# --- 1. CONFIGURATION ---

# Neo4j driver
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# OpenAI client configured for Ollama
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_KEY,
)

# Global scorer configuration
CURRENT_SCORER_TYPE = ScorerType.COMPOSITE

# Global composite weights configuration
CURRENT_COMPOSITE_WEIGHTS = DEFAULT_COMPOSITE_WEIGHTS

# Global random seed for consistent sampling
RANDOM_SEED = None

# Global batch processing configuration
BATCH_SIZE = 10

# Utility functions to configure global settings
def set_scorer_type(scorer_type: ScorerType):
    """Set the global scorer type for isRelevant evaluation."""
    global CURRENT_SCORER_TYPE
    CURRENT_SCORER_TYPE = scorer_type


def set_composite_weights(weights: CompositeWeights):
    """Set the global composite weights for COMPOSITE scorer type."""
    global CURRENT_COMPOSITE_WEIGHTS
    CURRENT_COMPOSITE_WEIGHTS = weights


def set_random_seed(seed: int):
    """Set the global random seed for consistent node sampling."""
    global RANDOM_SEED
    RANDOM_SEED = seed


def set_batch_size(batch_size: int = 10):
    """Configure batch size for processing. Always uses batch processing."""
    global BATCH_SIZE
    BATCH_SIZE = batch_size


def reset_global_config():
    """Reset global configuration to defaults."""
    global CURRENT_SCORER_TYPE, CURRENT_COMPOSITE_WEIGHTS, RANDOM_SEED, BATCH_SIZE
    CURRENT_SCORER_TYPE = ScorerType.COMPOSITE
    CURRENT_COMPOSITE_WEIGHTS = DEFAULT_COMPOSITE_WEIGHTS
    RANDOM_SEED = None
    BATCH_SIZE = 10


# Helper function for LLM calls
def call_ollama_llm(
    system_prompt: str, user_prompt: str, response_format=None, timeout=30
) -> str:
    """
    Call Ollama LLM with timeout and error handling
    """
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,  # "http://localhost:11434/v1",
            api_key=OLLAMA_KEY,
            timeout=timeout,  # Add timeout
        )

        if response_format:
            response = client.beta.chat.completions.parse(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
                timeout=timeout,
            )
            return response.choices[0].message.parsed
        else:
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
        # Return a fallback response instead of crashing
        if response_format:
            # Try to create a minimal valid response for structured formats
            try:
                if (
                    hasattr(response_format, "__name__")
                    and response_format.__name__ == "QueryIntentResponse"
                ):
                    from openai import BaseModel

                    class FallbackResponse(BaseModel):
                        intent: str = "product_search"
                        confidence: float = 0.5
                        reasoning: str = "Fallback due to LLM timeout"
                        entities: list = []

                    return FallbackResponse()
                else:
                    return "Error: LLM timeout"
            except:
                return "Error: LLM timeout"
        else:
            return "I apologize, but I'm having trouble processing your request due to a technical issue. Please try again."


# --- 2. AGENT STATE DEFINITION ---
class RetrievalState(TypedDict):
    question: str
    query_input: QueryInput  # Structured query for relevance scoring
    sampled_nodes: List[Dict]  # 10 random nodes from Neo4j graph
    semantic_scored_nodes: List[
        NodeInput
    ]  # Top 5 nodes selected by semantic similarity
    expanded_nodes: List[Dict]  # Nodes found through subgraph expansion
    expanded_scored_nodes: List[NodeInput]  # Expanded nodes scored with isRelevant
    final_relevant_nodes: List[NodeInput]  # Final combined list of relevant nodes
    expanded_subgraph: List[Dict]  # Expanded subgraph relationships
    decision: str
    final_answer: str
    revision_history: List[str]  # Track revision attempts


# --- 3. HELPER FUNCTIONS FOR RELEVANCE SYSTEM ---


class QueryIntentResponse(PydanticBaseModel):
    """Structured response for query intent analysis."""

    intent: str = Field(
        description="Query intent: product_search, document_request, technical_support, comparison_request, or specification_inquiry"
    )
    confidence: float = Field(description="Confidence level in detected intent (0-1)")
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


def analyze_query_intent(question: str) -> QueryIntent:
    """Analyze user query intent via LLM."""

    system_prompt = """You are an expert in user intent analysis. Your task is to classify user questions into one of the following categories:

1. **product_search**: User is looking for specific products, often with criteria like price, color, features
   - Examples: "Find red mountain bikes under $1000", "Search for affordable mountain bikes"

2. **document_request**: User wants documents, manuals, guides, instructions
   - Examples: "Show me the manual", "I want the documentation", "Where can I find instructions?"

3. **technical_support**: User has technical problems, seeks help, troubleshooting
   - Examples: "My bike is broken", "How do I fix this problem?", "The bike doesn't work"

4. **comparison_request**: User wants to compare products or options
   - Examples: "Compare bike A vs bike B", "Differences between X and Y", "Which is better?"

5. **specification_inquiry**: User seeks technical specifications, detailed features
   - Examples: "Technical specs of X", "Technical characteristics", "Product specifications"

Analyze the question and return the most appropriate intent with confidence and reasoning."""

    user_prompt = f"Analyze this question and determine the intent: '{question}'"

    try:
        response = call_ollama_llm(system_prompt, user_prompt, QueryIntentResponse)

        # Convert intent string to enum
        intent_mapping = {
            "product_search": QueryIntent.PRODUCT_SEARCH,
            "document_request": QueryIntent.DOCUMENT_REQUEST,
            "technical_support": QueryIntent.TECHNICAL_SUPPORT,
            "comparison_request": QueryIntent.COMPARISON_REQUEST,
            "specification_inquiry": QueryIntent.SPECIFICATION_INQUIRY,
        }

        intent_str = response.intent.lower()
        detected_intent = intent_mapping.get(intent_str, QueryIntent.PRODUCT_SEARCH)

        return detected_intent

    except Exception as e:
        logger.error(f"Error in intent analysis: {e}")
        return QueryIntent.PRODUCT_SEARCH


def extract_entities_from_query(question: str) -> List[str]:
    """Extract entities from query (simplified version)."""
    # Simplified implementation - use more sophisticated NER in production
    import re

    # Extract common product keywords
    entities = []

    # Colors
    colors = re.findall(
        r"\b(red|blue|green|black|white|rosso|blu|verde|nero|bianco)\b",
        question.lower(),
    )
    entities.extend(colors)

    # Product types
    products = re.findall(
        r"\b(mountain bike|bike|bicycle|handlebar|brake|frame|helmet|jersey|bici|bicicletta|manubrio|freno)\b",
        question.lower(),
    )
    entities.extend(products)

    # Remove duplicates and return
    return list(set(entities))


def create_query_input(question: str) -> QueryInput:
    """Create structured QueryInput from user question."""
    intent = analyze_query_intent(question)
    entities = extract_entities_from_query(question)

    # Generate embeddings for query (mock for now)
    embeddings = np.random.rand(384)  # Use real embeddings in production

    return QueryInput(
        text=question, embeddings=embeddings, entities=entities, intent=intent
    )


def sample_random_nodes_from_neo4j(limit: int = 5) -> List[Dict]:
    """Sample random nodes from Neo4j graph."""

    try:
        with neo4j_driver.session() as session:
            # Get total node count for random sampling
            count_query = "MATCH (n) RETURN count(n) as total"
            total_nodes = session.run(count_query).single()["total"]

            if total_nodes == 0:
                logger.warning("No nodes found in graph!")
                return []

            # Generate random numbers for sampling
            skip_values = [
                random.randint(0, max(0, total_nodes - 1)) for _ in range(limit)
            ]

            sampled_nodes = []
            for skip_val in skip_values:
                # Query to get a random node
                query = f"""
                MATCH (n)
                SKIP {skip_val}
                LIMIT 1
                RETURN n, labels(n) as node_labels, id(n) as node_id
                """

                result = session.run(query).single()
                if result:
                    node = dict(result["n"])
                    node["_labels"] = result["node_labels"]
                    node["_id"] = result["node_id"]
                    sampled_nodes.append(node)

            return sampled_nodes

    except Exception as e:
        logger.error(f"Error sampling nodes: {e}")
        return []


def convert_neo4j_node_to_node_input(neo4j_node: Dict) -> NodeInput:
    """Convert a Neo4j node to NodeInput for isRelevant."""

    # Extract descriptive text from node
    text_fields = ["name", "document_name", "filename", "content"]
    text_parts = []

    for field in text_fields:
        if field in neo4j_node and neo4j_node[field]:
            text_parts.append(str(neo4j_node[field]))

    # Add other relevant properties
    if "category_name" in neo4j_node and neo4j_node["category_name"]:
        text_parts.append(f"Category: {neo4j_node['category_name']}")

    if "list_price" in neo4j_node and neo4j_node["list_price"]:
        text_parts.append(f"Price: ${neo4j_node['list_price']}")

    if "color" in neo4j_node and neo4j_node["color"]:
        text_parts.append(f"Color: {neo4j_node['color']}")

    text = " | ".join(text_parts) if text_parts else "Unknown content"

    # Determine node type
    labels = neo4j_node.get("_labels", ["Unknown"])
    node_type = labels[0].lower() if labels else "unknown"

    # Extract entities (simplified version)
    entities = extract_entities_from_query(text)

    # Generate embeddings (mock for now)
    embeddings = np.random.rand(384)

    # Graph relations
    graph_relations = {k: v for k, v in neo4j_node.items() if not k.startswith("_")}

    node = NodeInput(
        text=text,
        embeddings=embeddings,
        graph_relations=graph_relations,
        node_type=node_type,
        entities=entities,
    )

    # Add score attribute (initially 0)
    node.score = 0.0

    return node


# --- 4. EXECUTION GRAPH NODE DEFINITIONS ---


def analyze_query(state: RetrievalState) -> Dict:
    """Node to analyze query and create QueryInput structure."""
    question = state["question"]

    query_input = create_query_input(question)

    return {"query_input": query_input}


def sample_neo4j_nodes(state: RetrievalState) -> Dict:

    global RANDOM_SEED
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    sampled_nodes = sample_random_nodes_from_neo4j(limit=20)

    return {"sampled_nodes": sampled_nodes}


def score_semantic_similarity(state: RetrievalState) -> Dict:
    """Node to score sampled nodes using semantic similarity and select top 5."""
    global BATCH_SIZE
    
    logger.debug(f"NODE: Semantic Similarity Scoring (batch_size={BATCH_SIZE})")

    sampled_nodes = state["sampled_nodes"]
    query_input = state["query_input"]

    # Convert nodes to NodeInput format for semantic scoring
    candidate_nodes = []
    for neo4j_node in sampled_nodes:
        node_input = convert_neo4j_node_to_node_input(neo4j_node)
        candidate_nodes.append(node_input)

    if not candidate_nodes:
        logger.warning("No candidate nodes to score")
        return {"semantic_scored_nodes": []}

    try:
        # Always use batch semantic similarity processing
        from isRelevant import batch_semantic_similarity
        
        similarity_scores = batch_semantic_similarity(query_input, candidate_nodes)

        # Assign scores to nodes
        for node, score in zip(candidate_nodes, similarity_scores):
            node.score = score

        scored_nodes = candidate_nodes

        # Sort by similarity and apply threshold
        sorted_nodes = sorted(scored_nodes, key=lambda x: x.score, reverse=True)

        # Take top 5 with minimum threshold of 0.50
        semantic_scored_nodes = [node for node in sorted_nodes[:5] if node.score >= 0.50]

        logger.info(
            f"Selected {len(semantic_scored_nodes)} nodes with semantic similarity ≥ 0.50"
        )

        return {"semantic_scored_nodes": semantic_scored_nodes}

    except Exception as e:
        logger.error(f"Batch semantic similarity processing failed: {e}")
        logger.exception("Full traceback:")
        
        # Return empty results rather than falling back
        return {"semantic_scored_nodes": []}


def expand_subgraph(state: RetrievalState) -> Dict:
    """Node to expand subgraph around semantic scored nodes."""

    semantic_scored_nodes = state["semantic_scored_nodes"]
    expanded_subgraph = []
    expanded_nodes = []
    expanded_node_ids = set()  # Track to avoid duplicates

    try:
        with neo4j_driver.session() as session:
            for node in semantic_scored_nodes:
                # Try to extract usable ID for expansion
                node_id = None

                # Try different ID fields
                if "product_id" in node.graph_relations:
                    node_id = node.graph_relations["product_id"]
                    expansion_query = """
                    MATCH (p:Product {product_id: $node_id})-[r]-(connected)
                    RETURN p, type(r) as relation_type, connected, labels(connected) as connected_labels
                    LIMIT 3
                    """
                elif "_id" in node.graph_relations:
                    node_id = node.graph_relations["_id"]
                    expansion_query = """
                    MATCH (n)-[r]-(connected)
                    WHERE id(n) = $node_id
                    RETURN n, type(r) as relation_type, connected, labels(connected) as connected_labels
                    LIMIT 3
                    """

                if node_id is not None:
                    try:
                        expansion_results = session.run(
                            expansion_query, node_id=node_id
                        )
                        for record in expansion_results:
                            expanded_subgraph.append(dict(record))

                            # Extract connected node and add to expanded_nodes if not already present
                            connected_node = dict(record["connected"])
                            connected_node["labels"] = record["connected_labels"]

                            # Create unique identifier for deduplication
                            if "product_id" in connected_node:
                                unique_id = f"product_{connected_node['product_id']}"
                            elif hasattr(record["connected"], "id"):
                                unique_id = f"id_{record['connected'].id}"
                            else:
                                unique_id = f"node_{hash(str(connected_node))}"

                            if unique_id not in expanded_node_ids:
                                expanded_node_ids.add(unique_id)
                                expanded_nodes.append(connected_node)

                    except Exception as e:
                        logger.debug(f"Error expanding node {node_id}: {e}")

    except Exception as e:
        logger.error(f"General error in subgraph expansion: {e}")

    return {"expanded_subgraph": expanded_subgraph, "expanded_nodes": expanded_nodes}


def score_expanded_nodes_with_isrelevant(state: RetrievalState) -> Dict:
    """Score both semantic nodes and expanded nodes using isRelevant, then combine."""
    global CURRENT_SCORER_TYPE, CURRENT_COMPOSITE_WEIGHTS, BATCH_SIZE
    scorer_type = CURRENT_SCORER_TYPE

    logger.debug(
        f"NODE: Score All Nodes with isRelevant ({scorer_type.value.upper()}, batch_size={BATCH_SIZE})"
    )

    semantic_scored_nodes = state["semantic_scored_nodes"]
    expanded_nodes = state.get("expanded_nodes", [])
    query_input = state["query_input"]

    # Convert expanded nodes to NodeInput format first
    expanded_node_inputs = []
    for neo4j_node in expanded_nodes:
        try:
            node_input = convert_neo4j_node_to_node_input(neo4j_node)
            expanded_node_inputs.append(node_input)
        except Exception as e:
            logger.warning(f"Error converting expanded node: {e}")
            continue

    # Combine all nodes for processing
    all_nodes = semantic_scored_nodes + expanded_node_inputs

    if not all_nodes:
        logger.warning("No nodes to score")
        return {
            "expanded_scored_nodes": [],
            "final_relevant_nodes": [],
        }

    try:
        # Always use batch processing - much more efficient
        all_scores = batch_isRelevant(
            query_input, all_nodes, scorer_type, BATCH_SIZE, CURRENT_COMPOSITE_WEIGHTS
        )

        # Split scores back to semantic and expanded
        semantic_scores = all_scores[: len(semantic_scored_nodes)]
        expanded_scores = all_scores[len(semantic_scored_nodes) :]

        # Update semantic nodes with new scores
        rescored_semantic_nodes = []
        for i, node in enumerate(semantic_scored_nodes):
            if i < len(semantic_scores):
                node.score = semantic_scores[i]
                rescored_semantic_nodes.append(node)

        # Update expanded nodes with scores
        expanded_scored_nodes = []
        for i, node in enumerate(expanded_node_inputs):
            if i < len(expanded_scores):
                node.score = expanded_scores[i]
                expanded_scored_nodes.append(node)

        # Combine all nodes and select top ones based on isRelevant scores
        all_scored_nodes = rescored_semantic_nodes + expanded_scored_nodes

        # Sort all nodes by isRelevant score and take top 15
        final_relevant_nodes = sorted(
            all_scored_nodes, key=lambda x: x.score, reverse=True
        )[:15]

        logger.info(f"Successfully scored {len(all_nodes)} nodes using batch processing")

        return {
            "expanded_scored_nodes": expanded_scored_nodes,
            "final_relevant_nodes": final_relevant_nodes,
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        logger.exception("Full traceback:")
        
        # Return empty results rather than falling back to individual processing
        return {
            "expanded_scored_nodes": [],
            "final_relevant_nodes": semantic_scored_nodes[:5],  # Return top semantic nodes as fallback
        }


def evaluate_context(state: RetrievalState) -> Dict:
    """Node to evaluate if collected context is sufficient."""

    class Decision(PydanticBaseModel):
        """Decision whether context is sufficient or needs revision."""

        decision: str = Field(description="'sufficient' or 'revision'")
        reasoning: str = Field(description="Brief explanation of the decision")

    final_relevant_nodes = state.get("final_relevant_nodes", [])
    query_input = state["query_input"]
    revision_history = state.get("revision_history", [])

    # Prevent infinite loops: if we've already revised 2+ times, force sufficient
    if len(revision_history) >= 2:
        return {"decision": "sufficient"}

    # Check if we have enough relevant nodes with high score
    high_relevance_nodes = [
        n for n in final_relevant_nodes if getattr(n, "score", 0) > 0.7
    ]

    # If we have at least one high-relevance node, consider sufficient
    if len(high_relevance_nodes) >= 1:
        return {"decision": "sufficient"}

    context_summary = f"""
    Total relevant nodes: {len(final_relevant_nodes)}
    High relevance nodes (>0.7): {len(high_relevance_nodes)}
    Query intent: {query_input.intent.value}
    """

    top_nodes_text = "\n".join(
        [
            f"- {node.text[:100]}... (score: {getattr(node, 'score', 0):.3f})"
            for node in final_relevant_nodes[:5]
        ]
    )

    system_prompt = """You are a supervisor of a knowledge graph-based RAG system. Evaluate whether the collected context is sufficient to answer the user's question.

If the context seems complete and relevant for the intent, respond 'sufficient'.
If the context is poor or irrelevant, respond 'revision'.

IMPORTANT: Bias towards 'sufficient' unless the context is completely irrelevant."""

    user_prompt = f"""Question: {state["question"]}
Intent detected: {query_input.intent.value}
Revision history: {revision_history}

Context analysis from knowledge graph:
{context_summary}

Top 5 relevant nodes:
{top_nodes_text}

Evaluate whether the context is sufficient to answer the question."""

    try:
        decision = call_ollama_llm(system_prompt, user_prompt, Decision, timeout=15)
        return {"decision": decision.decision}
    except Exception as e:
        logger.warning(f"Error in context evaluation: {e}")
        # Fallback: always consider sufficient to prevent loops
        return {"decision": "sufficient"}


def revise_question(state: RetrievalState) -> Dict:
    """Node to reformulate question if context is insufficient."""

    current_query = state["query_input"]

    system_prompt = """You are an expert in knowledge graph search. The previous question did not produce sufficient results from the knowledge graph.
Reformulate the question for a different approach.

Suggestions:
- Use different synonyms
- Change the search focus
- Be more or less specific
- Consider a different intent

Generate only the reformulated question."""

    user_prompt = f"""Original Question: {state["question"]}
Current intent: {current_query.intent.value}
Entities found: {current_query.entities}
Revision history: {state.get("revision_history", [])}

Reformulate the question to get better results from the knowledge graph:"""

    try:
        new_question = call_ollama_llm(system_prompt, user_prompt)

        # Update history
        history = state.get("revision_history", [])
        history.append(state["question"])

        return {"question": new_question.strip(), "revision_history": history}

    except Exception as e:
        logger.warning(f"Error in question revision: {e}")
        # Fallback: small modification to original question
        original = state["question"]
        fallback_question = f"Show me information about: {original}"

        history = state.get("revision_history", [])
        history.append(state["question"])

        return {"question": fallback_question, "revision_history": history}


def generate_answer(state: RetrievalState) -> Dict:
    """Node to generate final answer."""

    final_relevant_nodes = state.get("final_relevant_nodes", [])
    query_input = state["query_input"]
    expanded_subgraph = state.get("expanded_subgraph", [])

    # Prepare context from final relevant nodes (which includes both sampled and expanded nodes)
    context_text = "\n\n".join(
        [
            f"Resource {i + 1} (relevance: {getattr(node, 'score', 0):.3f}):\n{node.text}"
            for i, node in enumerate(final_relevant_nodes)
        ]
    )

    # Add information from expanded subgraph if available
    if expanded_subgraph:
        context_text += (
            f"\n\nKnowledge graph connections: {len(expanded_subgraph)} relations found"
        )

    system_prompt = f"""You are an AI assistant that uses a knowledge graph to answer questions. 
You specialize in {query_input.intent.value}. 

Answer the user's question based on the provided context, which has been extracted from a knowledge graph and evaluated for relevance.

Instructions:
1. Use only information from the provided context
2. Indicate confidence level in your response
3. If context is insufficient, be honest about it
4. Structure the response clearly and helpfully
5. Mention that information comes from a knowledge graph"""

    user_prompt = f"""Question: {state["question"]}
Intent: {query_input.intent.value}
Relevant entities: {", ".join(query_input.entities)}

--- CONTEXT FROM KNOWLEDGE GRAPH ---
{context_text}
--- END CONTEXT ---

Provide a complete and accurate answer based on the knowledge graph:"""

    try:
        answer = call_ollama_llm(system_prompt, user_prompt)
        final_result = {
            "final_answer": answer.strip() if answer else "No answer generated"
        }
        return final_result

    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        logger.exception("Full traceback:")

        # Fallback: basic response
        fallback_answer = f"""I'm sorry, I encountered an error generating a response from the knowledge graph. 

Available context from graph:
{context_text[:500]}...

For the question '{state["question"]}', I suggest reformulating the request or checking the knowledge graph connection."""

        return {"final_answer": fallback_answer}


# --- 5. EXECUTION GRAPH CONSTRUCTION WITH LANGGRAPH ---

workflow = StateGraph(RetrievalState)

# Add nodes
workflow.add_node("analyze_query", analyze_query)
workflow.add_node("sample_neo4j_nodes", sample_neo4j_nodes)
workflow.add_node("score_semantic_similarity", score_semantic_similarity)
workflow.add_node("expand_subgraph", expand_subgraph)
workflow.add_node(
    "score_all_nodes_with_isrelevant", score_expanded_nodes_with_isrelevant
)
workflow.add_node("evaluate_context", evaluate_context)
workflow.add_node("revise_question", revise_question)
workflow.add_node("generate_answer", generate_answer)

# Define edge logic
workflow.set_entry_point("analyze_query")
workflow.add_edge("analyze_query", "sample_neo4j_nodes")
workflow.add_edge("sample_neo4j_nodes", "score_semantic_similarity")
workflow.add_edge("score_semantic_similarity", "expand_subgraph")
workflow.add_edge("expand_subgraph", "score_all_nodes_with_isrelevant")
workflow.add_edge("score_all_nodes_with_isrelevant", "evaluate_context")

# Conditional edge: after evaluation, either generate answer or revise question
workflow.add_conditional_edges(
    "evaluate_context",
    lambda x: x["decision"],
    {
        "sufficient": "generate_answer",
        "revision": "revise_question",
    },
)

# Loop: after revising question, return to analysis (which creates new QueryInput)
workflow.add_edge("revise_question", "analyze_query")
workflow.add_edge("generate_answer", END)

# Compile graph
app = workflow.compile()

# --- 6. RAG SYSTEM EXECUTION WITH KNOWLEDGE GRAPH ---

if __name__ == "__main__":
    # Example questions suitable for knowledge graph
    example_questions = [
        "What mountain bikes do you have?",
        "Show me products under $500",
        "What documents are available about bikes?",
    ]

    # Use first question as example
    user_question = example_questions[1]

    inputs = {
        "question": user_question,
        "revision_history": [],
        "sampled_nodes": [],
        "semantic_scored_nodes": [],
        "expanded_nodes": [],
        "expanded_scored_nodes": [],
        "final_relevant_nodes": [],
        "expanded_subgraph": [],
    }

    try:
        # Test Neo4j connection
        with neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as total_nodes").single()
            total_nodes = result["total_nodes"]

        # Instead of streaming, use invoke to get complete final state
        final_state = app.invoke(inputs, {"recursion_limit": 15})

        # Show workflow steps summary
        if "query_input" in final_state:
            qi = final_state["query_input"]

        if "semantic_scored_nodes" in final_state:
            semantic_nodes = final_state["semantic_scored_nodes"]

        if "final_relevant_nodes" in final_state:
            final_relevant_nodes = final_state["final_relevant_nodes"]

        if "final_answer" in final_state and final_state["final_answer"]:
            print("\nFINAL ANSWER FROM KNOWLEDGE GRAPH:")
            print("-" * 40)
            print(final_state["final_answer"])

            # Show statistics
            if "final_relevant_nodes" in final_state:
                final_relevant_nodes = final_state["final_relevant_nodes"]
                semantic_nodes = final_state.get("semantic_scored_nodes", [])
                expanded_nodes = final_state.get("expanded_nodes", [])

                if semantic_nodes:
                    avg_semantic = sum(
                        getattr(n, "score", 0) for n in semantic_nodes
                    ) / len(semantic_nodes)

                if final_relevant_nodes:
                    avg_final = sum(
                        getattr(n, "score", 0) for n in final_relevant_nodes
                    ) / len(final_relevant_nodes)

        else:
            logger.debug(
                f"Available keys in final_state: {list(final_state.keys()) if final_state else 'None'}"
            )
            if final_state and "final_answer" in final_state:
                logger.debug(f"Final answer content: '{final_state['final_answer']}'")

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.exception("Full traceback:")

    finally:
        # Close Neo4j connection
        neo4j_driver.close()
