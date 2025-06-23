"""
Relevance Scorer Implementation

This module implements the isRelevant function and supporting components
for scoring node relevance in the RAG system knowledge graph.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScorerType(Enum):
    """Enumeration of available scorer types."""
    COMPOSITE = "composite"
    PARALLEL = "parallel"
    ROUTER = "router"


class QueryIntent(Enum):
    """Enumeration of query intent types."""
    PRODUCT_SEARCH = "product_search"
    DOCUMENT_REQUEST = "document_request"
    TECHNICAL_SUPPORT = "technical_support"
    COMPARISON_REQUEST = "comparison_request"
    SPECIFICATION_INQUIRY = "specification_inquiry"


@dataclass
class QueryInput:
    """Input structure for query information."""
    text: str
    embeddings: np.ndarray
    parsed_entities: Dict[str, Any]
    intent: QueryIntent


@dataclass
class NodeInput:
    """Input structure for node information."""
    text_or_db: Union[str, Dict[str, Any]]
    embeddings: np.ndarray
    graph_relations: Dict[str, Any]
    node_type: str = "unknown"


class RelevanceMetric(ABC):
    """Abstract base class for relevance metrics."""
    
    @abstractmethod
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate relevance score between query and node."""
        pass


class SemanticSimilarityMetric(RelevanceMetric):
    """Semantic similarity metric using embeddings."""
    
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate cosine similarity between query and node embeddings."""
        try:
            # Ensure embeddings are 2D arrays for sklearn
            query_emb = query.embeddings.reshape(1, -1)
            node_emb = node.embeddings.reshape(1, -1)
            
            similarity = cosine_similarity(query_emb, node_emb)[0][0]
            # Normalize to 0-1 range (cosine similarity can be -1 to 1)
            return (similarity + 1) / 2
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0


class LLMJudgeMetric(RelevanceMetric):
    """LLM as a judge metric for relevance assessment."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Use LLM to judge relevance between query and node."""
        try:
            if not self.llm_client:
                # Fallback to simple heuristic if no LLM available
                return self._heuristic_relevance(query, node)
            
            # Prepare content for LLM evaluation
            node_content = self._format_node_content(node)
            
            prompt = f"""
            Evaluate the relevance of the following content to the user query on a scale of 0.0 to 1.0.
            
            User Query: {query.text}
            
            Content: {node_content}
            
            Consider semantic relevance, topic alignment, and potential usefulness.
            Respond with only a number between 0.0 and 1.0.
            """
            
            # This would be replaced with actual LLM API call
            # response = self.llm_client.generate(prompt)
            # return float(response.strip())
            
            # Placeholder implementation
            return self._heuristic_relevance(query, node)
            
        except Exception as e:
            logger.warning(f"Error in LLM judge metric: {e}")
            return 0.0
    
    def _format_node_content(self, node: NodeInput) -> str:
        """Format node content for LLM evaluation."""
        if isinstance(node.text_or_db, dict):
            # Format database entry
            formatted = []
            for key, value in node.text_or_db.items():
                if key not in ['id', 'rowguid', 'ModifiedDate']:
                    formatted.append(f"{key}: {value}")
            return "; ".join(formatted)
        return str(node.text_or_db)
    
    def _heuristic_relevance(self, query: QueryInput, node: NodeInput) -> float:
        """Simple heuristic relevance calculation as fallback."""
        query_words = set(query.text.lower().split())
        node_content = self._format_node_content(node).lower()
        node_words = set(node_content.split())
        
        if not query_words or not node_words:
            return 0.0
        
        # Calculate word overlap
        overlap = len(query_words.intersection(node_words))
        return min(overlap / len(query_words), 1.0)


class EntityMatchMetric(RelevanceMetric):
    """Entity matching metric for structured entity comparison."""
    
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate entity match score between query and node."""
        try:
            if not query.parsed_entities:
                return 0.0
            
            node_entities = self._extract_node_entities(node)
            if not node_entities:
                return 0.0
            
            total_score = 0.0
            entity_count = 0
            
            # Match product entities
            if 'product' in query.parsed_entities:
                product_score = self._match_product_entities(
                    query.parsed_entities['product'], 
                    node_entities
                )
                total_score += product_score
                entity_count += 1
            
            # Match document entities
            if 'document' in query.parsed_entities:
                document_score = self._match_document_entities(
                    query.parsed_entities['document'],
                    node_entities
                )
                total_score += document_score
                entity_count += 1
            
            return total_score / entity_count if entity_count > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error in entity match metric: {e}")
            return 0.0
    
    def _extract_node_entities(self, node: NodeInput) -> Dict[str, Any]:
        """Extract entities from node content."""
        entities = {}
        
        if isinstance(node.text_or_db, dict):
            # Extract from database record
            for key, value in node.text_or_db.items():
                if key.lower() in ['name', 'category', 'color', 'size', 'type']:
                    entities[key.lower()] = str(value).lower()
        else:
            # Extract from text content (simplified)
            content = str(node.text_or_db).lower()
            entities['content'] = content
        
        return entities
    
    def _match_product_entities(self, query_product: Dict, node_entities: Dict) -> float:
        """Match product-specific entities."""
        score = 0.0
        matches = 0
        total_checks = 0
        
        # Check category match
        if 'category' in query_product and 'category' in node_entities:
            if query_product['category'].lower() in node_entities['category']:
                matches += 1
            total_checks += 1
        
        # Check features match
        if 'features' in query_product:
            for feature in query_product['features']:
                feature_lower = feature.lower()
                found = any(feature_lower in str(value) for value in node_entities.values())
                if found:
                    matches += 1
                total_checks += 1
        
        return matches / total_checks if total_checks > 0 else 0.0
    
    def _match_document_entities(self, query_document: Dict, node_entities: Dict) -> float:
        """Match document-specific entities."""
        score = 0.0
        matches = 0
        total_checks = 0
        
        # Check document type match
        if 'type' in query_document:
            doc_type = query_document['type'].lower()
            found = any(doc_type in str(value) for value in node_entities.values())
            if found:
                matches += 1
            total_checks += 1
        
        return matches / total_checks if total_checks > 0 else 0.0


class GraphCentralityMetric(RelevanceMetric):
    """Graph centrality metric for node importance assessment."""
    
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate graph centrality score for the node."""
        try:
            relations = node.graph_relations
            
            # Calculate degree centrality (normalized)
            incoming_count = len(relations.get('incoming', []))
            outgoing_count = len(relations.get('outgoing', []))
            total_degree = incoming_count + outgoing_count
            
            # Simple centrality score based on degree
            # In a real implementation, this would use proper graph algorithms
            max_expected_degree = 50  # Configurable threshold
            degree_score = min(total_degree / max_expected_degree, 1.0)
            
            # Weight by relationship quality/strength if available
            relationship_weights = relations.get('relationship_weights', {})
            if relationship_weights:
                weighted_score = sum(relationship_weights.values()) / len(relationship_weights)
                degree_score = (degree_score + weighted_score) / 2
            
            return degree_score
            
        except Exception as e:
            logger.warning(f"Error in graph centrality metric: {e}")
            return 0.0


class DataTypePriorityMetric(RelevanceMetric):
    """Data type priority metric using priority matrix."""
    
    def __init__(self):
        # Priority matrix: query_intent -> node_type -> priority_score
        self.priority_matrix = {
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
    
    def calculate(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate data type priority score."""
        try:
            intent_priorities = self.priority_matrix.get(query.intent, {})
            return intent_priorities.get(node.node_type, 0.1)
        except Exception as e:
            logger.warning(f"Error in data type priority metric: {e}")
            return 0.1


class RelevanceScorer:
    """Main relevance scorer that orchestrates different scoring strategies."""
    
    def __init__(self, llm_client=None):
        self.metrics = {
            'semantic_similarity': SemanticSimilarityMetric(),
            'llm_judge': LLMJudgeMetric(llm_client),
            'entity_match': EntityMatchMetric(),
            'graph_centrality': GraphCentralityMetric(),
            'data_type_priority': DataTypePriorityMetric()
        }
        
        # Configurable weights for composite scorer
        self.composite_weights = {
            'semantic_similarity': 0.3,
            'llm_judge': 0.25,
            'entity_match': 0.2,
            'graph_centrality': 0.15,
            'data_type_priority': 0.1
        }
        
        # Router rules for different query types and node combinations
        self.router_rules = self._initialize_router_rules()
    
    def _initialize_router_rules(self) -> Dict[str, List[str]]:
        """Initialize router rules for metric selection."""
        return {
            f"{QueryIntent.PRODUCT_SEARCH.value}_product": [
                'semantic_similarity', 'entity_match', 'data_type_priority'
            ],
            f"{QueryIntent.PRODUCT_SEARCH.value}_document": [
                'semantic_similarity', 'llm_judge'
            ],
            f"{QueryIntent.DOCUMENT_REQUEST.value}_document": [
                'semantic_similarity', 'llm_judge', 'data_type_priority'
            ],
            f"{QueryIntent.TECHNICAL_SUPPORT.value}_specification": [
                'semantic_similarity', 'entity_match', 'graph_centrality'
            ],
            # Add more rules as needed
        }
    
    def calculate_relevance(
        self, 
        query: QueryInput, 
        node: NodeInput, 
        scorer_type: ScorerType
    ) -> float:
        """Main function to calculate relevance score."""
        try:
            if scorer_type == ScorerType.COMPOSITE:
                return self._composite_score(query, node)
            elif scorer_type == ScorerType.PARALLEL:
                return self._parallel_score(query, node)
            elif scorer_type == ScorerType.ROUTER:
                return self._router_score(query, node)
            else:
                raise ValueError(f"Unknown scorer type: {scorer_type}")
                
        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0
    
    def _composite_score(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate composite weighted score."""
        total_score = 0.0
        
        for metric_name, metric in self.metrics.items():
            weight = self.composite_weights.get(metric_name, 0.0)
            if weight > 0:
                score = metric.calculate(query, node)
                total_score += weight * score
                logger.debug(f"{metric_name}: {score:.3f} (weight: {weight})")
        
        return min(total_score, 1.0)  # Ensure score doesn't exceed 1.0
    
    def _parallel_score(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate parallel (maximum) score."""
        scores = []
        
        for metric_name, metric in self.metrics.items():
            score = metric.calculate(query, node)
            scores.append(score)
            logger.debug(f"{metric_name}: {score:.3f}")
        
        return max(scores) if scores else 0.0
    
    def _router_score(self, query: QueryInput, node: NodeInput) -> float:
        """Calculate router-based score with conditional metric selection."""
        # Determine which metrics to use based on query intent and node type
        rule_key = f"{query.intent.value}_{node.node_type}"
        selected_metrics = self.router_rules.get(rule_key, ['semantic_similarity'])
        
        logger.debug(f"Router using metrics for {rule_key}: {selected_metrics}")
        
        total_score = 0.0
        metric_count = 0
        
        for metric_name in selected_metrics:
            if metric_name in self.metrics:
                score = self.metrics[metric_name].calculate(query, node)
                total_score += score
                metric_count += 1
                logger.debug(f"{metric_name}: {score:.3f}")
        
        return total_score / metric_count if metric_count > 0 else 0.0


def isRelevant(
    query: QueryInput,
    node: NodeInput,
    scorer_type: ScorerType,
    llm_client=None
) -> float:
    """
    Main isRelevant function as specified in the design document.
    
    Args:
        query: QueryInput containing text, embeddings, entities, and intent
        node: NodeInput containing content, embeddings, and graph relations
        scorer_type: ScorerType enum specifying scoring strategy
        llm_client: Optional LLM client for LLM judge metric
    
    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    scorer = RelevanceScorer(llm_client)
    return scorer.calculate_relevance(query, node, scorer_type)


# Example usage and testing functions
def create_sample_query() -> QueryInput:
    """Create a sample query for testing."""
    return QueryInput(
        text="Find red mountain bikes under $1000",
        embeddings=np.random.rand(384),  # Sample embedding
        parsed_entities={
            'product': {
                'features': ['red'],
                'category': 'mountain bikes',
                'price_range': 'under $1000'
            }
        },
        intent=QueryIntent.PRODUCT_SEARCH
    )


def create_sample_node() -> NodeInput:
    """Create a sample node for testing."""
    return NodeInput(
        text_or_db={
            'name': 'Mountain Bike Pro',
            'category': 'Mountain Bikes',
            'color': 'Red',
            'price': 899.99
        },
        embeddings=np.random.rand(384),  # Sample embedding
        graph_relations={
            'incoming': ['category_1', 'brand_2'],
            'outgoing': ['review_1', 'review_2', 'spec_1'],
            'relationship_weights': {'category': 0.8, 'brand': 0.6}
        },
        node_type='product'
    )


if __name__ == "__main__":
    # Example usage
    query = create_sample_query()
    node = create_sample_node()
    
    print("Testing isRelevant function:")
    print(f"Composite Score: {isRelevant(query, node, ScorerType.COMPOSITE):.3f}")
    print(f"Parallel Score: {isRelevant(query, node, ScorerType.PARALLEL):.3f}")
    print(f"Router Score: {isRelevant(query, node, ScorerType.ROUTER):.3f}") 