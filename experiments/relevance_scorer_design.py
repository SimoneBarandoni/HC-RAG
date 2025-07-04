#!/usr/bin/env python3
"""
Relevance Scorer Design Document

This module outlines the design for a comprehensive node relevance scoring system
that ranks knowledge graph nodes based on their relevance to user queries.

DESIGN PHILOSOPHY:
- Multi-dimensional scoring: combines semantic, structural, and rule-based approaches
- Query-aware: adapts scoring based on query type and intent
- Data-type aware: different scoring for products vs documents vs categories
- Explainable: provides reasoning for relevance scores
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class QueryIntent(Enum):
    """Types of user query intents"""
    PRODUCT_SEARCH = "product_search"          # "Find red mountain bikes"
    DOCUMENT_REQUEST = "document_request"      # "Show me the manual"
    COMPARISON = "comparison"                  # "Compare X vs Y"
    COMPATIBILITY = "compatibility"           # "Compatible with X"
    TROUBLESHOOTING = "troubleshooting"       # "Problem with X"
    SPECIFICATION = "specification"           # "What are the specs of X"


class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    PRODUCT = "product"
    DOCUMENT = "document"
    CATEGORY = "category"
    JSON_TABLE = "json_table"
    PDF_CHUNK = "pdf_chunk"
    ANNOTATION = "annotation"


@dataclass
class RelevanceScore:
    """Container for relevance scoring results"""
    total_score: float
    component_scores: Dict[str, float]
    reasoning: List[str]
    confidence: float
    

@dataclass
class QueryContext:
    """Complete query context for relevance scoring"""
    original_query: str
    parsed_entities: Dict[str, Any]
    query_embedding: np.ndarray
    query_intent: QueryIntent
    
    # Extracted key information
    product_keywords: List[str]
    document_keywords: List[str]
    feature_keywords: List[str]
    category_keywords: List[str]


@dataclass
class NodeContext:
    """Complete node context for relevance scoring"""
    node_id: str
    node_type: NodeType
    content: str
    metadata: Dict[str, Any]
    neo4j_data: Optional[Dict[str, Any]]
    
    # Graph properties
    similarity_score: float           # Original embedding similarity
    graph_centrality: float          # Node importance in graph
    path_distance: int               # Distance from query-matched nodes
    
    # Content properties
    content_embedding: Optional[np.ndarray]
    content_length: int
    content_quality: float           # Heuristic content quality score


# =============================================================================
# ABSTRACT SCORER INTERFACE
# =============================================================================

class RelevanceComponent(ABC):
    """Abstract base class for relevance scoring components"""
    
    @abstractmethod
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        """
        Score a node's relevance to a query
        
        Returns:
            Tuple of (score, reasoning_list)
        """
        pass
    
    @abstractmethod
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """Return the weight for this component given query intent and node type"""
        pass


# =============================================================================
# SEMANTIC SIMILARITY SCORER
# =============================================================================

class SemanticSimilarityScorer(RelevanceComponent):
    """Scores based on embedding similarity between query and node content"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        reasoning = []
        
        # Use pre-computed similarity if available
        if node_context.similarity_score > 0:
            base_score = node_context.similarity_score
            reasoning.append(f"Embedding similarity: {base_score:.3f}")
        else:
            # Compute fresh similarity
            if node_context.content_embedding is not None:
                similarity = np.dot(query_context.query_embedding, node_context.content_embedding)
                base_score = max(0, similarity)  # Ensure non-negative
                reasoning.append(f"Computed embedding similarity: {base_score:.3f}")
            else:
                base_score = 0.0
                reasoning.append("No embedding available for similarity")
        
        # Apply boost factors based on content type and quality
        boost_factor = 1.0
        
        # Boost for high-quality, relevant content
        if node_context.content_length > 100:  # Substantial content
            boost_factor *= 1.1
            reasoning.append("Boost: substantial content")
            
        if node_context.content_quality > 0.8:  # High quality content
            boost_factor *= 1.15
            reasoning.append("Boost: high quality content")
        
        final_score = min(1.0, base_score * boost_factor)
        return final_score, reasoning
    
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """Semantic similarity is important for all query types"""
        base_weight = 0.3
        
        # Increase weight for document requests
        if query_intent == QueryIntent.DOCUMENT_REQUEST:
            return base_weight + 0.1
            
        # Increase weight for PDF chunks and annotations
        if node_type in [NodeType.PDF_CHUNK, NodeType.ANNOTATION]:
            return base_weight + 0.05
            
        return base_weight


# =============================================================================
# LLM JUDGE SCORER
# =============================================================================

class LLMJudgeScorer(RelevanceComponent):
    """Uses LLM to evaluate relevance with detailed reasoning"""
    
    def __init__(self, llm_client, model_name="gemma3:1b"):
        self.llm_client = llm_client
        self.model_name = model_name
        
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        # Prepare context for LLM
        prompt = self._create_relevance_prompt(query_context, node_context)
        
        try:
            # Query LLM for relevance assessment
            response = self._query_llm(prompt)
            score, reasoning = self._parse_llm_response(response)
            
            return score, reasoning
            
        except Exception as e:
            # Fallback to heuristic scoring
            fallback_score = self._heuristic_fallback(query_context, node_context)
            return fallback_score, [f"LLM unavailable, used fallback: {fallback_score:.3f}"]
    
    def _create_relevance_prompt(self, query_context: QueryContext, node_context: NodeContext) -> str:
        return f"""
        Evaluate how relevant this knowledge graph node is to the user query.
        
        USER QUERY: "{query_context.original_query}"
        QUERY INTENT: {query_context.query_intent.value}
        
        NODE INFORMATION:
        - Type: {node_context.node_type.value}
        - Content: {node_context.content[:500]}...
        - Metadata: {node_context.metadata}
        
        Score the relevance on a scale of 0.0 to 1.0 where:
        - 1.0 = Perfectly relevant, directly answers the query
        - 0.8 = Highly relevant, very useful for answering
        - 0.6 = Moderately relevant, somewhat useful
        - 0.4 = Marginally relevant, tangentially related
        - 0.2 = Low relevance, weakly related
        - 0.0 = Not relevant at all
        
        Respond in this format:
        SCORE: 0.X
        REASONING: [explanation of why this score was assigned]
        """
    
    def _query_llm(self, prompt: str) -> str:
        """Query the LLM with the relevance prompt"""
        # Implementation would depend on your LLM setup
        # This is a placeholder
        completion = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )
        return completion.choices[0].message.content
    
    def _parse_llm_response(self, response: str) -> Tuple[float, List[str]]:
        """Parse LLM response to extract score and reasoning"""
        lines = response.strip().split('\n')
        score = 0.5  # default
        reasoning = []
        
        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith('REASONING:'):
                reasoning.append(line.split(':', 1)[1].strip())
        
        return min(1.0, max(0.0, score)), reasoning
    
    def _heuristic_fallback(self, query_context: QueryContext, node_context: NodeContext) -> float:
        """Fallback heuristic when LLM is unavailable"""
        # Simple keyword matching fallback
        query_words = set(query_context.original_query.lower().split())
        content_words = set(node_context.content.lower().split())
        
        overlap = len(query_words.intersection(content_words))
        max_possible = max(len(query_words), len(content_words))
        
        return overlap / max_possible if max_possible > 0 else 0.0
    
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """LLM judge is particularly valuable for complex queries"""
        base_weight = 0.25
        
        # Increase weight for complex intents
        if query_intent in [QueryIntent.COMPARISON, QueryIntent.TROUBLESHOOTING]:
            return base_weight + 0.1
            
        return base_weight


# =============================================================================
# ENTITY MATCH SCORER
# =============================================================================

class EntityMatchScorer(RelevanceComponent):
    """Scores based on structured entity matching between query and node"""
    
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        reasoning = []
        total_score = 0.0
        
        parsed = query_context.parsed_entities
        
        # Product entity matching
        if node_context.node_type == NodeType.PRODUCT:
            product_score = self._score_product_match(parsed.get("product", {}), node_context, reasoning)
            total_score += product_score * 0.6
        
        # Document entity matching
        if node_context.node_type in [NodeType.DOCUMENT, NodeType.PDF_CHUNK]:
            doc_score = self._score_document_match(parsed.get("document", {}), node_context, reasoning)
            total_score += doc_score * 0.4
        
        # Category matching
        if node_context.node_type == NodeType.CATEGORY:
            cat_score = self._score_category_match(parsed.get("product", {}), node_context, reasoning)
            total_score += cat_score * 0.5
        
        return min(1.0, total_score), reasoning
    
    def _score_product_match(self, product_entity: Dict, node_context: NodeContext, reasoning: List[str]) -> float:
        """Score product-specific entity matching"""
        score = 0.0
        
        if not product_entity:
            return score
        
        neo4j_data = node_context.neo4j_data or {}
        
        # Name matching
        if product_entity.get("name") and neo4j_data.get("name"):
            if product_entity["name"].lower() in neo4j_data["name"].lower():
                score += 0.5
                reasoning.append(f"Product name match: {product_entity['name']}")
        
        # Feature matching (color, size, etc.)
        features = product_entity.get("features", [])
        for feature in features:
            feature_lower = feature.lower()
            
            # Check various product attributes
            for attr in ["color", "size", "category"]:
                if attr in neo4j_data and feature_lower in str(neo4j_data[attr]).lower():
                    score += 0.2
                    reasoning.append(f"Feature match ({attr}): {feature}")
                    break
        
        # Category matching
        if product_entity.get("category") and neo4j_data.get("category"):
            if product_entity["category"].lower() in neo4j_data["category"].lower():
                score += 0.3
                reasoning.append(f"Category match: {product_entity['category']}")
        
        return min(1.0, score)
    
    def _score_document_match(self, doc_entity: Dict, node_context: NodeContext, reasoning: List[str]) -> float:
        """Score document-specific entity matching"""
        score = 0.0
        
        if not doc_entity:
            return score
        
        metadata = node_context.metadata
        
        # Document type matching
        if doc_entity.get("type"):
            doc_type = doc_entity["type"].lower()
            content_lower = node_context.content.lower()
            
            if doc_type in content_lower or doc_type in str(metadata).lower():
                score += 0.6
                reasoning.append(f"Document type match: {doc_entity['type']}")
        
        # Document name matching
        if doc_entity.get("name"):
            doc_name = doc_entity["name"].lower()
            if "document_name" in metadata and doc_name in metadata["document_name"].lower():
                score += 0.4
                reasoning.append(f"Document name match: {doc_entity['name']}")
        
        return min(1.0, score)
    
    def _score_category_match(self, product_entity: Dict, node_context: NodeContext, reasoning: List[str]) -> float:
        """Score category-specific matching"""
        score = 0.0
        
        if product_entity.get("category"):
            category = product_entity["category"].lower()
            neo4j_data = node_context.neo4j_data or {}
            
            if "name" in neo4j_data and category in neo4j_data["name"].lower():
                score += 0.8
                reasoning.append(f"Category name match: {category}")
        
        return min(1.0, score)
    
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """Entity matching is important for structured queries"""
        base_weight = 0.2
        
        # Increase weight for product searches
        if query_intent == QueryIntent.PRODUCT_SEARCH and node_type == NodeType.PRODUCT:
            return base_weight + 0.15
            
        # Increase weight for document requests
        if query_intent == QueryIntent.DOCUMENT_REQUEST and node_type in [NodeType.DOCUMENT, NodeType.PDF_CHUNK]:
            return base_weight + 0.1
            
        return base_weight


# =============================================================================
# GRAPH CENTRALITY SCORER
# =============================================================================

class GraphCentralityScorer(RelevanceComponent):
    """Scores based on graph structure and node importance"""
    
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        reasoning = []
        
        # Use pre-computed centrality if available
        centrality_score = node_context.graph_centrality
        reasoning.append(f"Graph centrality: {centrality_score:.3f}")
        
        # Boost for nodes close to query-matched nodes
        distance_penalty = max(0, 1.0 - (node_context.path_distance * 0.2))
        if node_context.path_distance > 0:
            reasoning.append(f"Path distance penalty: {distance_penalty:.3f} (distance: {node_context.path_distance})")
        
        # Boost for nodes with many connections (implied importance)
        if node_context.neo4j_data:
            # This would require relationship count from Neo4j
            connection_boost = 1.0  # Placeholder
        else:
            connection_boost = 0.8  # Non-Neo4j nodes get slight penalty
            reasoning.append("Non-graph node penalty")
        
        final_score = centrality_score * distance_penalty * connection_boost
        return min(1.0, final_score), reasoning
    
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """Graph centrality is moderately important across all scenarios"""
        base_weight = 0.15
        
        # Increase weight for compatibility and comparison queries
        if query_intent in [QueryIntent.COMPATIBILITY, QueryIntent.COMPARISON]:
            return base_weight + 0.05
            
        return base_weight


# =============================================================================
# DATA TYPE PRIORITY SCORER
# =============================================================================

class DataTypePriorityScorer(RelevanceComponent):
    """Applies rule-based scoring based on data types and query patterns"""
    
    def score(self, query_context: QueryContext, node_context: NodeContext) -> Tuple[float, List[str]]:
        reasoning = []
        
        # Priority matrix: query_intent -> node_type -> score
        priority_matrix = {
            QueryIntent.PRODUCT_SEARCH: {
                NodeType.PRODUCT: 1.0,
                NodeType.CATEGORY: 0.6,
                NodeType.DOCUMENT: 0.3,
                NodeType.PDF_CHUNK: 0.2,
                NodeType.JSON_TABLE: 0.4,
                NodeType.ANNOTATION: 0.3
            },
            QueryIntent.DOCUMENT_REQUEST: {
                NodeType.DOCUMENT: 1.0,
                NodeType.PDF_CHUNK: 0.9,
                NodeType.ANNOTATION: 0.8,
                NodeType.PRODUCT: 0.2,
                NodeType.CATEGORY: 0.1,
                NodeType.JSON_TABLE: 0.7
            },
            QueryIntent.SPECIFICATION: {
                NodeType.JSON_TABLE: 1.0,
                NodeType.DOCUMENT: 0.8,
                NodeType.PDF_CHUNK: 0.7,
                NodeType.PRODUCT: 0.6,
                NodeType.ANNOTATION: 0.5,
                NodeType.CATEGORY: 0.2
            }
        }
        
        # Get base priority score
        intent_priorities = priority_matrix.get(query_context.query_intent, {})
        base_score = intent_priorities.get(node_context.node_type, 0.5)
        
        reasoning.append(f"Data type priority: {base_score:.3f} for {node_context.node_type.value}")
        
        # Apply content-specific boosts
        boost_score = self._apply_content_boosts(query_context, node_context, reasoning)
        
        final_score = min(1.0, base_score + boost_score)
        return final_score, reasoning
    
    def _apply_content_boosts(self, query_context: QueryContext, node_context: NodeContext, reasoning: List[str]) -> float:
        """Apply content-specific boost factors"""
        boost = 0.0
        
        # Boost for tables with numerical data (specs, measurements)
        if node_context.node_type == NodeType.JSON_TABLE:
            content_lower = node_context.content.lower()
            if any(keyword in content_lower for keyword in ["pressure", "size", "weight", "speed", "temperature"]):
                boost += 0.2
                reasoning.append("Boost: technical specifications")
        
        # Boost for PDF chunks with specific content types
        if node_context.node_type == NodeType.PDF_CHUNK:
            content_lower = node_context.content.lower()
            if any(keyword in content_lower for keyword in ["manual", "instruction", "guide", "how to"]):
                boost += 0.15
                reasoning.append("Boost: instructional content")
        
        return boost
    
    def get_weight(self, query_intent: QueryIntent, node_type: NodeType) -> float:
        """Data type priority provides baseline scoring"""
        return 0.1


# =============================================================================
# MAIN RELEVANCE SCORER
# =============================================================================

class RelevanceScorer:
    """Main relevance scoring system that combines multiple scoring components"""
    
    def __init__(self, embedding_model, llm_client, neo4j_driver):
        # Initialize scoring components
        self.components = {
            "semantic_similarity": SemanticSimilarityScorer(embedding_model),
            "llm_judge": LLMJudgeScorer(llm_client),
            "entity_match": EntityMatchScorer(),
            "graph_centrality": GraphCentralityScorer(),
            "data_type_priority": DataTypePriorityScorer()
        }
        
        self.neo4j_driver = neo4j_driver
        
    def score_nodes(self, query_context: QueryContext, 
                   node_contexts: List[NodeContext]) -> List[Tuple[NodeContext, RelevanceScore]]:
        """
        Score all nodes for relevance to the query
        
        Args:
            query_context: Complete query information
            node_contexts: List of nodes to score
            
        Returns:
            List of (node, relevance_score) tuples, sorted by relevance
        """
        scored_nodes = []
        
        for node_context in node_contexts:
            relevance_score = self._score_single_node(query_context, node_context)
            scored_nodes.append((node_context, relevance_score))
        
        # Sort by total relevance score (descending)
        scored_nodes.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_nodes
    
    def _score_single_node(self, query_context: QueryContext, node_context: NodeContext) -> RelevanceScore:
        """Score a single node using all scoring components"""
        component_scores = {}
        all_reasoning = []
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for component_name, component in self.components.items():
            # Get component weight for this query/node combination
            weight = component.get_weight(query_context.query_intent, node_context.node_type)
            
            if weight > 0:
                try:
                    # Score this component
                    score, reasoning = component.score(query_context, node_context)
                    
                    # Record results
                    component_scores[component_name] = score
                    all_reasoning.extend([f"[{component_name}] {r}" for r in reasoning])
                    
                    # Add to weighted total
                    weighted_score = score * weight
                    total_weighted_score += weighted_score
                    total_weights += weight
                    
                except Exception as e:
                    # Handle component errors gracefully
                    component_scores[component_name] = 0.0
                    all_reasoning.append(f"[{component_name}] Error: {str(e)}")
        
        # Calculate final score
        if total_weights > 0:
            final_score = total_weighted_score / total_weights
        else:
            final_score = 0.0
        
        # Calculate confidence based on score consistency
        if len(component_scores) > 1:
            scores_array = np.array(list(component_scores.values()))
            confidence = 1.0 - np.std(scores_array)  # Higher std = lower confidence
        else:
            confidence = 0.5  # Moderate confidence with single component
        
        return RelevanceScore(
            total_score=final_score,
            component_scores=component_scores,
            reasoning=all_reasoning,
            confidence=max(0.0, min(1.0, confidence))
        )


# =============================================================================
# USAGE EXAMPLE AND HELPER FUNCTIONS
# =============================================================================

def create_query_context(original_query: str, parsed_entities: Dict[str, Any], 
                        query_embedding: np.ndarray) -> QueryContext:
    """Helper function to create QueryContext from RAG system outputs"""
    
    # Determine query intent (this could be more sophisticated)
    query_lower = original_query.lower()
    if any(word in query_lower for word in ["manual", "document", "guide", "instruction"]):
        intent = QueryIntent.DOCUMENT_REQUEST
    elif any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        intent = QueryIntent.COMPARISON
    elif any(word in query_lower for word in ["compatible", "work with", "fits"]):
        intent = QueryIntent.COMPATIBILITY
    elif any(word in query_lower for word in ["problem", "issue", "fix", "troubleshoot"]):
        intent = QueryIntent.TROUBLESHOOTING
    elif any(word in query_lower for word in ["spec", "specification", "dimension", "weight"]):
        intent = QueryIntent.SPECIFICATION 
    else:
        intent = QueryIntent.PRODUCT_SEARCH
    
    return QueryContext(
        original_query=original_query,
        parsed_entities=parsed_entities,
        query_embedding=query_embedding,
        query_intent=intent,
        product_keywords=[],  # Could extract from query
        document_keywords=[],
        feature_keywords=[],
        category_keywords=[]
    )


def create_node_context(node_data: Dict[str, Any]) -> NodeContext:
    """Helper function to create NodeContext from subgraph node data"""
    
    data = node_data.get("data", {})
    
    # Determine node type
    node_type_str = data.get("type", "unknown").lower()
    if "product" in node_type_str:
        node_type = NodeType.PRODUCT
    elif "document" in node_type_str:
        node_type = NodeType.DOCUMENT
    elif "category" in node_type_str:
        node_type = NodeType.CATEGORY
    elif "json" in node_type_str or "table" in node_type_str:
        node_type = NodeType.JSON_TABLE
    elif "pdf" in node_type_str or "chunk" in node_type_str:
        node_type = NodeType.PDF_CHUNK
    elif "annotation" in node_type_str:
        node_type = NodeType.ANNOTATION
    else:
        node_type = NodeType.DOCUMENT  # Default
    
    content = data.get("content", "")
    
    return NodeContext(
        node_id=data.get("id", ""),
        node_type=node_type,
        content=content,
        metadata=data.get("metadata", {}),
        neo4j_data=data.get("neo4j_data"),
        similarity_score=data.get("similarity_score", 0.0),
        graph_centrality=0.5,  # Would need to be computed
        path_distance=0,       # Would need to be computed
        content_embedding=None,  # Could be provided
        content_length=len(content),
        content_quality=0.7    # Heuristic based on content
    )


# Example usage:
"""
# Initialize the relevance scorer
scorer = RelevanceScorer(embedding_model, llm_client, neo4j_driver)

# Create query context from RAG system
query_context = create_query_context(user_query, parsed_entities, query_embedding)

# Create node contexts from subgraph nodes
node_contexts = [create_node_context(node) for node in subgraph_nodes]

# Score all nodes
scored_nodes = scorer.score_nodes(query_context, node_contexts)

# Use the results
for node_context, relevance_score in scored_nodes:
    print(f"Node {node_context.node_id}: {relevance_score.total_score:.3f}")
    print(f"  Reasoning: {relevance_score.reasoning[:2]}")  # Top 2 reasons
""" 