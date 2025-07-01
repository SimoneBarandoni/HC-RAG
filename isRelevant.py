from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time
from configurations import OLLAMA_BASE_URL, OLLAMA_KEY, OLLAMA_MODEL


class QueryIntent(Enum):
    PRODUCT_SEARCH = "product_search"
    DOCUMENT_REQUEST = "document_request"
    TECHNICAL_SUPPORT = "technical_support"
    COMPARISON_REQUEST = "comparison_request"
    SPECIFICATION_INQUIRY = "specification_inquiry"

@dataclass
class QueryInput():
    text: str
    embeddings: np.ndarray
    entities: List[str]
    intent: QueryIntent

@dataclass
class NodeInput():
    text: str
    embeddings: np.ndarray
    graph_relations: Dict[str, Any]
    node_type: str
    entities: List[str]

class ScorerType(Enum):
    COMPOSITE = "composite"
    PARALLEL = "parallel"
    ROUTER = "router"
    ROUTER_ALL = "router_all"
    ROUTER_TWO_SEM_LLM = "router_two_sem_llm"
    ROUTER_TWO_ENT_TYPE = "router_two_ent_type"
    ROUTER_SINGLE_SEM = "router_single_sem"
    ROUTER_SINGLE_LLM = "router_single_llm"
    ROUTER_SINGLE_ENT = "router_single_ent"
    ROUTER_SINGLE_TYPE = "router_single_type"

class RelevanceScore(BaseModel):
    score: float

class BatchRelevanceScore(BaseModel):
    scores: List[float] = Field(description="List of relevance scores for each node in the batch")

priority_matrix = {
    QueryIntent.PRODUCT_SEARCH: {
        'product': 1.0,
        'category': 0.8,
        'specification': 0.6,
        'document': 0.3,
        'annotation': 0.2,
        'unknown': 0.1
    },
    QueryIntent.DOCUMENT_REQUEST: {
        'document': 1.0,
        'specification': 0.7,
        'annotation': 0.6,
        'product': 0.4,
        'category': 0.2,
        'unknown': 0.1
    },
    QueryIntent.TECHNICAL_SUPPORT: {
        'document': 1.0,
        'specification': 0.9,
        'annotation': 0.7,
        'product': 0.6,
        'category': 0.3,
        'unknown': 0.1
    },
    QueryIntent.COMPARISON_REQUEST: {
        'product': 1.0,
        'specification': 0.8,
        'category': 0.6,
        'document': 0.4,
        'annotation': 0.3,
        'unknown': 0.1
    },
    QueryIntent.SPECIFICATION_INQUIRY: {
        'specification': 1.0,
        'product': 0.7,
        'annotation': 0.6,
        'document': 0.5,
        'category': 0.3,
        'unknown': 0.1
    }
}

def input_query() -> QueryInput:
    text = "Find red mountain bikes under $1000"
    entities = ["red mountain bike"]
    intent = QueryIntent.PRODUCT_SEARCH
    embeddings = np.random.rand(384)
    return QueryInput(text=text, entities=entities, intent=intent, embeddings=embeddings)

def input_node() -> NodeInput:
    text = "Red mountain bike description"
    node_type = "document"
    entities = ["red mountain bike", "handlebar", "brakes", "pedals"]
    embeddings = np.random.rand(384)
    graph_relations = {"type": "product", "id": "123"}
    return NodeInput(text=text, embeddings=embeddings, graph_relations=graph_relations, node_type=node_type, entities=entities)


def semantic_similarity(query: QueryInput, node: NodeInput) -> float:
    query_emb = query.embeddings.reshape(1, -1)
    node_emb = node.embeddings.reshape(1, -1)
    similarity = cosine_similarity(query_emb, node_emb)[0][0]
    return (similarity + 1) / 2


def llm_judge(query: QueryInput, node: NodeInput) -> float:
    prompt = f"""
            User Query: {query.text}
            
            Content: {node.text}
            
            """
    
    system_prompt = """You are an expert relevance evaluator for a knowledge graph system. Your task is to assess how relevant a piece of content is to a user's query.
                    Scoring Guidelines:
                    - 0.9-1.0: Perfect match - directly answers the query or provides exactly what's requested
                    - 0.8-0.9: Highly relevant - very useful for answering the query, contains key information
                    - 0.6-0.7: Moderately relevant - somewhat useful, related but not central to the query
                    - 0.4-0.5: Marginally relevant - tangentially related, might provide context
                    - 0.2-0.3: Low relevance - weakly related, unlikely to be useful
                    - 0.0-0.1: Not relevant - completely unrelated to the query

                    Consider these factors:
                    1. Direct topic alignment (does the content address the query topic?)
                    2. Specificity match (does it match specific criteria like price, color, features?)
                    3. Content type appropriateness (product info for product queries, docs for technical questions)
                    4. Completeness (does it provide comprehensive information?)

                    Examples:

                    Query: "Find red mountain bikes under $1000"
                    Content: "Premium Red Mountain Bike - Trail Blazer X1 with advanced suspension and lightweight frame, perfect for off-road adventures under $900"
                    Score: 0.95 (Perfect match - red mountain bike under $1000)

                    Query: "Find red mountain bikes under $1000"  
                    Content: "Blue Mountain Bike - Rugged terrain specialist with 21-speed gear system, priced at $750"
                    Score: 0.7 (Good price and category match, but wrong color)

                    Now evaluate the given query and content, considering semantic relevance, topic alignment, and potential usefulness:"""
    
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,#"http://localhost:11434/v1",
        api_key=OLLAMA_KEY,
    )
    response = client.beta.chat.completions.parse(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": prompt},
        ],
        response_format=RelevanceScore,
    )
    return response.choices[0].message.parsed.score

def batch_llm_judge(query: QueryInput, nodes: List[NodeInput]) -> List[float]:
    """Batch LLM judge for multiple nodes at once - much more efficient than individual calls."""
    
    if not nodes:
        return []
    
    # Prepare batch content
    nodes_text = []
    for i, node in enumerate(nodes, 1):
        nodes_text.append(f"Content {i}: {node.text}")
    
    batch_content = "\n\n".join(nodes_text)
    
    prompt = f"""
            User Query: {query.text}
            
            Multiple Contents to Evaluate:
            {batch_content}
            
            """
    
    system_prompt = f"""You are an expert relevance evaluator for a knowledge graph system. Your task is to assess how relevant each piece of content is to a user's query.

                    You will receive {len(nodes)} pieces of content to evaluate. For each content, provide a relevance score between 0.0 and 1.0.

                    Scoring Guidelines:
                    - 0.9-1.0: Perfect match - directly answers the query or provides exactly what's requested
                    - 0.8-0.9: Highly relevant - very useful for answering the query, contains key information
                    - 0.6-0.7: Moderately relevant - somewhat useful, related but not central to the query
                    - 0.4-0.5: Marginally relevant - tangentially related, might provide context
                    - 0.2-0.3: Low relevance - weakly related, unlikely to be useful
                    - 0.0-0.1: Not relevant - completely unrelated to the query

                    Consider these factors:
                    1. Direct topic alignment (does the content address the query topic?)
                    2. Specificity match (does it match specific criteria like price, color, features?)
                    3. Content type appropriateness (product info for product queries, docs for technical questions)
                    4. Completeness (does it provide comprehensive information?)

                    Return exactly {len(nodes)} scores as a list, one for each content in order.
                    
                    Example format for 3 contents:
                    Query: "Find red mountain bikes under $1000"
                    Content 1: "Premium Red Mountain Bike under $900" 
                    Content 2: "Blue Mountain Bike for $750"
                    Content 3: "Camping tent setup guide"
                    Response: [0.95, 0.70, 0.05]"""
    
    try:
        client = OpenAI(
            base_url=OLLAMA_BASE_URL,#"http://localhost:11434/v1",
            api_key=OLLAMA_KEY,
        )
        response = client.beta.chat.completions.parse(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": prompt},
            ],
            response_format=BatchRelevanceScore,
        )
        
        scores = response.choices[0].message.parsed.scores
        
        # Ensure we have the right number of scores
        if len(scores) != len(nodes):
            while len(scores) < len(nodes):
                scores.append(0.5)  # Default score for missing
            scores = scores[:len(nodes)]  # Truncate if too many
        
        return scores
        
    except Exception as e:
        # Fallback to individual calls
        return [llm_judge(query, node) for node in nodes]

def entity_match(query: QueryInput, node: NodeInput) -> float:
    query_entities = set(query.entities)
    node_entities = set(node.entities)
    
    # Handle case when query has no entities
    if len(query_entities) == 0:
        # If both query and node have no entities, return neutral score
        if len(node_entities) == 0:
            return 0.5
        # If only query has no entities but node has some, return low score
        else:
            return 0.1
    
    # Normal case: calculate intersection ratio
    match_ratio = len(query_entities.intersection(node_entities)) / len(query_entities)
    return match_ratio 

def node_type_priority(query: QueryInput, node: NodeInput) -> float:
    query_intent = query.intent
    node_type = node.node_type
    
    # Handle case when node_type is not in priority_matrix
    if node_type not in priority_matrix[query_intent]:
        # Use 'unknown' as fallback
        priority_score = priority_matrix[query_intent]['unknown']
        #print(f"Node Type Priority: {priority_score:.3f} (fallback for unknown type '{node_type}')")
    else:
        priority_score = priority_matrix[query_intent][node_type]
        #print(f"Node Type Priority: {priority_score:.3f}")
    
    return priority_score

def batch_semantic_similarity(query: QueryInput, nodes: List[NodeInput]) -> List[float]:
    """Batch semantic similarity calculation using vectorized operations."""
    if not nodes:
        return []
    
    # Vectorized similarity calculation
    query_emb = query.embeddings.reshape(1, -1)
    node_embs = np.array([node.embeddings for node in nodes])
    
    similarities = cosine_similarity(query_emb, node_embs)[0]
    # Normalize to 0-1 range
    normalized_similarities = [(sim + 1) / 2 for sim in similarities]
    
    return normalized_similarities

def batch_entity_match(query: QueryInput, nodes: List[NodeInput]) -> List[float]:
    """Batch entity matching calculation."""
    if not nodes:
        return []
    
    query_entities = set(query.entities)
    
    scores = []
    for node in nodes:
        node_entities = set(node.entities)
        
        # Handle case when query has no entities
        if len(query_entities) == 0:
            if len(node_entities) == 0:
                scores.append(0.5)
            else:
                scores.append(0.1)
        else:
            # Normal case: calculate intersection ratio
            match_ratio = len(query_entities.intersection(node_entities)) / len(query_entities)
            scores.append(match_ratio)
    
    return scores

def batch_node_type_priority(query: QueryInput, nodes: List[NodeInput]) -> List[float]:
    """Batch node type priority calculation."""
    if not nodes:
        return []
    
    query_intent = query.intent
    
    scores = []
    for node in nodes:
        node_type = node.node_type
        
        # Handle case when node_type is not in priority_matrix
        if node_type not in priority_matrix[query_intent]:
            priority_score = priority_matrix[query_intent]['unknown']
        else:
            priority_score = priority_matrix[query_intent][node_type]
        
        scores.append(priority_score)
    
    return scores

def parallel_score(query: QueryInput, node: NodeInput) -> float:
    return max(
        semantic_similarity(query, node),
        llm_judge(query, node),
        entity_match(query, node),
        node_type_priority(query, node)
    )

def router_score(query: QueryInput, node: NodeInput, metrics: List[Callable[[QueryInput, NodeInput], float]]) -> float:
    return sum(metric(query, node) for metric in metrics) / len(metrics)

def composite_score(query: QueryInput, node: NodeInput) -> float:
    return (
        semantic_similarity(query, node)*0.4 +
        llm_judge(query, node)*0.3 +
        entity_match(query, node)*0.15 +
        node_type_priority(query, node)*0.15
    )

def isRelevant(query: QueryInput, node: NodeInput, scorer_type: ScorerType) -> float:
    if scorer_type == ScorerType.COMPOSITE:
        return composite_score(query, node)
    elif scorer_type == ScorerType.PARALLEL:
        return parallel_score(query, node)
    elif scorer_type == ScorerType.ROUTER:
        return router_score(query, node, [semantic_similarity, llm_judge, node_type_priority])
    elif scorer_type == ScorerType.ROUTER_ALL:
        return router_score(query, node, [semantic_similarity, llm_judge, entity_match, node_type_priority])
    elif scorer_type == ScorerType.ROUTER_TWO_SEM_LLM:
        return router_score(query, node, [semantic_similarity, llm_judge])
    elif scorer_type == ScorerType.ROUTER_TWO_ENT_TYPE:
        return router_score(query, node, [entity_match, node_type_priority])
    elif scorer_type == ScorerType.ROUTER_SINGLE_SEM:
        return semantic_similarity(query, node)
    elif scorer_type == ScorerType.ROUTER_SINGLE_LLM:
        return llm_judge(query, node)
    elif scorer_type == ScorerType.ROUTER_SINGLE_ENT:
        return entity_match(query, node)
    elif scorer_type == ScorerType.ROUTER_SINGLE_TYPE:
        return node_type_priority(query, node)
    else:
        # Fallback to composite
        return composite_score(query, node)

def batch_isRelevant(query: QueryInput, nodes: List[NodeInput], scorer_type: ScorerType, batch_size: int = 10) -> List[float]:
    """
    Batch version of isRelevant - processes multiple nodes efficiently.
    
    Args:
        query: Query input
        nodes: List of nodes to score
        scorer_type: Scoring approach to use
        batch_size: Size of batches for LLM calls (default 10)
    
    Returns:
        List of relevance scores for each node
    """
    if not nodes:
        return []
    
    #print(f"🚀 Batch processing {len(nodes)} nodes with {scorer_type.value} scorer (batch_size={batch_size})")
    
    # For single metric scorers, we can process all at once
    if scorer_type == ScorerType.ROUTER_SINGLE_SEM:
        return batch_semantic_similarity(query, nodes)
    elif scorer_type == ScorerType.ROUTER_SINGLE_ENT:
        return batch_entity_match(query, nodes)
    elif scorer_type == ScorerType.ROUTER_SINGLE_TYPE:
        return batch_node_type_priority(query, nodes)
    elif scorer_type == ScorerType.ROUTER_SINGLE_LLM:
        # Process in batches for LLM
        return _batch_process_with_llm(query, nodes, batch_size)
    
    # For composite scorers, calculate all metrics in batch
    semantic_scores = batch_semantic_similarity(query, nodes)
    entity_scores = batch_entity_match(query, nodes) 
    type_scores = batch_node_type_priority(query, nodes)
    
    # LLM scores - process in batches
    llm_scores = _batch_process_with_llm(query, nodes, batch_size) if _needs_llm_scores(scorer_type) else [0.0] * len(nodes)
    
    # Combine scores based on scorer type
    final_scores = []
    for i in range(len(nodes)):
        sem_score = semantic_scores[i]
        ent_score = entity_scores[i]
        type_score = type_scores[i]
        llm_score = llm_scores[i] if i < len(llm_scores) else 0.0
        
        if scorer_type == ScorerType.COMPOSITE:
            score = sem_score * 0.4 + llm_score * 0.3 + ent_score * 0.15 + type_score * 0.15
        elif scorer_type == ScorerType.PARALLEL:
            score = max(sem_score, llm_score, ent_score, type_score)
        elif scorer_type == ScorerType.ROUTER:
            score = (sem_score + llm_score + type_score) / 3
        elif scorer_type == ScorerType.ROUTER_ALL:
            score = (sem_score + llm_score + ent_score + type_score) / 4
        elif scorer_type == ScorerType.ROUTER_TWO_SEM_LLM:
            score = (sem_score + llm_score) / 2
        elif scorer_type == ScorerType.ROUTER_TWO_ENT_TYPE:
            score = (ent_score + type_score) / 2
        else:
            # Fallback to composite
            score = sem_score * 0.4 + llm_score * 0.3 + ent_score * 0.15 + type_score * 0.15
        
        final_scores.append(score)
    
    print(f"✅ Batch processing completed: {len(final_scores)} scores generated")
    return final_scores

def _needs_llm_scores(scorer_type: ScorerType) -> bool:
    """Check if the scorer type requires LLM scores."""
    llm_scorers = {
        ScorerType.COMPOSITE, ScorerType.PARALLEL, ScorerType.ROUTER,
        ScorerType.ROUTER_ALL, ScorerType.ROUTER_TWO_SEM_LLM, ScorerType.ROUTER_SINGLE_LLM
    }
    return scorer_type in llm_scorers

def _batch_process_with_llm(query: QueryInput, nodes: List[NodeInput], batch_size: int) -> List[float]:
    """Process nodes with LLM in batches."""
    all_scores = []
    
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i + batch_size]
        batch_scores = batch_llm_judge(query, batch)
        all_scores.extend(batch_scores)
        
        print(f"  📊 Processed batch {i//batch_size + 1}: {len(batch)} nodes")
    
    return all_scores

if __name__ == "__main__":
    start = time.time()
    from sample_nodes import create_sample_nodes
    
    # Create query and sample nodes
    query = input_query()
    sample_nodes = create_sample_nodes()
    
    #print(f"Query: {query.text}")
    #print(f"Query Intent: {query.intent.value}")
    #print(f"Query Entities: {query.entities}")
    #print("=" * 80)

    nodes_dict = {}
    
    # Use batch processing for efficiency
    #print("🚀 Using batch processing for relevance scoring...")
    composite_scores = batch_isRelevant(query, sample_nodes, ScorerType.COMPOSITE, batch_size=5)
    
    # Assign scores to nodes
    for node, score in zip(sample_nodes, composite_scores):
        nodes_dict[node.text] = node
        nodes_dict[node.text].score = score
    
    # sort nodes_dict by score
    sorted_nodes = sorted(nodes_dict.items(), key=lambda x: x[1].score, reverse=True)
    #for node_text, node in sorted_nodes:
        #print(f"{node_text}: {node.score:.3f}")
    end = time.time()
    #print(f"Time taken: {end - start:.3f} seconds")
 