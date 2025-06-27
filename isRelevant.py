from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np
from enum import Enum
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import time

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

class RelevanceScore(BaseModel):
    score: float

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
    #print(f"Semantic Similarity: {((similarity + 1) / 2):.3f}")
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
        base_url="http://localhost:11434/v1",
        api_key="gemma3:1b",
    )
    response = client.beta.chat.completions.parse(
        model="gemma3:1b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {"role": "user", "content": prompt},
        ],
        response_format=RelevanceScore,
    )
    #print(f"LLM Judge: {response.choices[0].message.parsed.score:.3f}")
    return response.choices[0].message.parsed.score

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
    #print(f"Entity Match: {match_ratio:.3f}")
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

if __name__ == "__main__":
    start = time.time()
    from sample_nodes import create_sample_nodes
    
    # Create query and sample nodes
    query = input_query()
    sample_nodes = create_sample_nodes()
    
    print(f"Query: {query.text}")
    print(f"Query Intent: {query.intent.value}")
    print(f"Query Entities: {query.entities}")
    print("=" * 80)

    nodes_dict = {}
    
    # Iterate through all sample nodes and calculate relevance scores
    for i, node in enumerate(sample_nodes, 1):
        #print(f"\nNode {i}: {node.text[:60]}{'...' if len(node.text) > 60 else ''}")
        #print(f"Node Type: {node.node_type}")
        #print(f"Node Entities: {node.entities}")
        #print("-" * 40)
        nodes_dict[node.text] = node
        composite_result = isRelevant(query, node, ScorerType.COMPOSITE)
        nodes_dict[node.text].score = composite_result
        #print("--------------------------------")
        #parallel_result = isRelevant(query, node, ScorerType.PARALLEL)
        #print("--------------------------------")
        #router_result = isRelevant(query, node, ScorerType.ROUTER)
        
        #print(f"\nFINAL SCORES:")
        #print(f"  Composite Score: {composite_result:.3f}")
        #print(f"  Parallel Score:  {parallel_result:.3f}")
        #print(f"  Router Score:    {router_result:.3f}")
        #print("=" * 80)
    
    # sort nodes_dict by score
    sorted_nodes = sorted(nodes_dict.items(), key=lambda x: x[1].score, reverse=True)
    for node_text, node in sorted_nodes:
        print(f"{node_text}: {node.score:.3f}")
    end = time.time()
    print(f"Time taken: {end - start:.3f} seconds")
 