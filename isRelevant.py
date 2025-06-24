from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import numpy as np
from enum import Enum
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

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
        'unknown': 0.1
    },
    QueryIntent.DOCUMENT_REQUEST: {
        'document': 1.0,
        'specification': 0.7,
        'product': 0.4,
        'category': 0.2,
        'unknown': 0.1
    },
    QueryIntent.TECHNICAL_SUPPORT: {
        'document': 1.0,
        'specification': 0.9,
        'product': 0.6,
        'category': 0.3,
        'unknown': 0.1
    },
    QueryIntent.COMPARISON_REQUEST: {
        'product': 1.0,
        'specification': 0.8,
        'category': 0.6,
        'document': 0.4,
        'unknown': 0.1
    },
    QueryIntent.SPECIFICATION_INQUIRY: {
        'specification': 1.0,
        'product': 0.7,
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

def create_sample_nodes() -> List[NodeInput]:
    """Create a list of sample nodes with different characteristics for testing relevance scoring"""
    nodes = []
    
    # Node 1: Highly relevant red mountain bike product
    nodes.append(NodeInput(
        text="Premium Red Mountain Bike - Trail Blazer X1 with advanced suspension and lightweight frame, perfect for off-road adventures under $900",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "001", "category": "mountain_bikes"},
        node_type="product",
        entities=["red mountain bike", "trail", "suspension", "lightweight"]
    ))
    
    # Node 2: Partially relevant - mountain bike but different color
    nodes.append(NodeInput(
        text="Blue Mountain Bike - Rugged terrain specialist with 21-speed gear system, priced at $750",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "002", "category": "mountain_bikes"},
        node_type="product",
        entities=["blue mountain bike", "terrain", "gear system"]
    ))
    
    # Node 3: Document about mountain bike maintenance
    nodes.append(NodeInput(
        text="Mountain Bike Maintenance Guide - Complete handbook for maintaining your mountain bike including brake adjustments, tire care, and gear tuning",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_001", "category": "maintenance"},
        node_type="document",
        entities=["mountain bike", "maintenance", "brake", "tire", "gear"]
    ))
    
    # Node 4: Specification document
    nodes.append(NodeInput(
        text="Technical Specifications for Mountain Bike Components - Detailed specs for handlebars, frames, wheels, and suspension systems",
        embeddings=np.random.rand(384),
        graph_relations={"type": "specification", "id": "spec_001", "category": "components"},
        node_type="specification",
        entities=["mountain bike", "handlebars", "frames", "wheels", "suspension"]
    ))
    
    # Node 5: Category node
    nodes.append(NodeInput(
        text="Mountain Bikes Category - Browse our complete selection of mountain bikes for all skill levels and terrains",
        embeddings=np.random.rand(384),
        graph_relations={"type": "category", "id": "cat_001", "parent": "bikes"},
        node_type="category",
        entities=["mountain bikes", "selection", "terrain"]
    ))
    
    # Node 6: Irrelevant node - road bike
    nodes.append(NodeInput(
        text="Professional Road Racing Bike - Lightweight carbon fiber frame designed for speed on paved roads, $2500",
        embeddings=np.random.rand(384),
        graph_relations={"type": "product", "id": "003", "category": "road_bikes"},
        node_type="product",
        entities=["road bike", "carbon fiber", "racing", "paved roads"]
    ))
    
    # Node 7: Completely unrelated node
    nodes.append(NodeInput(
        text="Camping Tent Setup Instructions - How to properly set up your 4-person camping tent for outdoor adventures",
        embeddings=np.random.rand(384),
        graph_relations={"type": "document", "id": "doc_002", "category": "camping"},
        node_type="document",
        entities=["camping tent", "setup", "outdoor", "adventures"]
    ))
    
    return nodes

def semantic_similarity(query: QueryInput, node: NodeInput) -> float:
    query_emb = query.embeddings.reshape(1, -1)
    node_emb = node.embeddings.reshape(1, -1)
    similarity = cosine_similarity(query_emb, node_emb)[0][0]
    #print(f"Semantic Similarity: {((similarity + 1) / 2):.3f}")
    return (similarity + 1) / 2


def llm_judge(query: QueryInput, node: NodeInput) -> float:
    prompt = f"""
            Evaluate the relevance of the following content to the user query on a scale of 0.0 to 1.0.
            
            User Query: {query.text}
            
            Content: {node.text}
            
            Consider semantic relevance, topic alignment, and potential usefulness.
            """
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="gemma3:1b",
    )
    response = client.beta.chat.completions.parse(
        model="gemma3:1b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that judges the relevance of a query to a node."
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
    #print(f"Entity Match: {len(query_entities.intersection(node_entities)) / len(query_entities):.3f}")
    return len(query_entities.intersection(node_entities)) / len(query_entities) 

def node_type_priority(query: QueryInput, node: NodeInput) -> float:
    query_intent = query.intent
    node_type = node.node_type
    #print(f"Node Type Priority: {priority_matrix[query_intent][node_type]:.3f}")
    return priority_matrix[query_intent][node_type]

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
        print("--------------------------------")
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
 