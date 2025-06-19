# 🎯 Relevance Scorer Design Document

## Overview
The `isRelevant` function needs to rank knowledge graph nodes by their relevance to user queries. This is a complex multi-dimensional problem requiring a hybrid approach.

## 🔍 Design Decisions

### 1. Query Input Format
**Recommendation: Use Multiple Query Formats Simultaneously**

```python
class QueryContext:
    original_query: str          # "Find red mountain bikes under $500"
    parsed_entities: Dict        # {product: {features: ["red"], category: "bikes"}}
    query_embedding: np.ndarray  # Vector representation
    query_intent: QueryIntent    # PRODUCT_SEARCH, DOCUMENT_REQUEST, etc.
```

**Why?** Different scoring methods need different query representations:
- LLM judges work best with natural language
- Entity matching needs structured data
- Semantic similarity needs embeddings
- Rule-based scoring needs intent classification

### 2. Multi-Layered Relevance Scoring

**Proposed Architecture: Weighted Hybrid Scoring**

```python
relevance_score = (
    0.30 * semantic_similarity_score +
    0.25 * llm_judge_score +
    0.20 * entity_match_score +
    0.15 * graph_centrality_score +
    0.10 * data_type_priority_score
)
```

## 🧠 Scoring Components

### Component 1: Semantic Similarity Scorer
- **Input**: Query embedding vs node content embedding
- **Method**: Cosine similarity with boost factors
- **Strengths**: Handles paraphrasing, semantic relationships
- **Use Cases**: All query types, especially when exact keywords don't match

```python
class SemanticSimilarityScorer:
    def score(self, query_context, node_context):
        base_similarity = cosine_similarity(query_embedding, node_embedding)
        
        # Apply boosts
        if node_context.content_length > 100:
            base_similarity *= 1.1  # Substantial content boost
        if node_context.content_quality > 0.8:
            base_similarity *= 1.15  # High quality content boost
            
        return min(1.0, base_similarity)
```

### Component 2: LLM Judge Scorer
- **Input**: Full query + node content + metadata
- **Method**: LLM evaluates relevance with reasoning
- **Strengths**: Context understanding, nuanced evaluation
- **Use Cases**: Complex queries, troubleshooting, comparisons

```python
class LLMJudgeScorer:
    def score(self, query_context, node_context):
        prompt = f"""
        Query: "{query_context.original_query}"
        Node Type: {node_context.node_type}
        Content: {node_context.content[:500]}
        
        Rate relevance 0.0-1.0 and explain why.
        """
        
        response = llm_client.query(prompt)
        return parse_score_and_reasoning(response)
```

### Component 3: Entity Match Scorer
- **Input**: Parsed entities vs node structured data
- **Method**: Exact and fuzzy matching on entities
- **Strengths**: Precise matching on structured data
- **Use Cases**: Product searches with specific features

```python
class EntityMatchScorer:
    def score(self, query_context, node_context):
        score = 0.0
        
        # Product name matching
        if query_entities.product.name in node.neo4j_data.name:
            score += 0.5
            
        # Feature matching (color, size, etc.)
        for feature in query_entities.product.features:
            if feature in node.neo4j_data.values():
                score += 0.2
                
        # Category matching
        if query_entities.product.category == node.category:
            score += 0.3
            
        return min(1.0, score)
```

### Component 4: Graph Centrality Scorer
- **Input**: Node position and connections in knowledge graph
- **Method**: PageRank, betweenness centrality, path distance
- **Strengths**: Identifies important hub nodes
- **Use Cases**: When structural importance matters

```python
class GraphCentralityScorer:
    def score(self, query_context, node_context):
        centrality_score = node_context.graph_centrality
        
        # Penalize nodes far from query matches
        distance_penalty = max(0, 1.0 - (node_context.path_distance * 0.2))
        
        return centrality_score * distance_penalty
```

### Component 5: Data Type Priority Scorer
- **Input**: Query intent + node type
- **Method**: Rule-based priority matrix
- **Strengths**: Ensures appropriate content types are prioritized
- **Use Cases**: When query intent is clear

```python
PRIORITY_MATRIX = {
    QueryIntent.PRODUCT_SEARCH: {
        NodeType.PRODUCT: 1.0,
        NodeType.CATEGORY: 0.6,
        NodeType.DOCUMENT: 0.3
    },
    QueryIntent.DOCUMENT_REQUEST: {
        NodeType.DOCUMENT: 1.0,
        NodeType.PDF_CHUNK: 0.9,
        NodeType.PRODUCT: 0.2
    }
}
```

## 🎛️ Adaptive Weighting

**Key Insight**: Different query types need different scoring emphasis.

```python
def get_adaptive_weights(query_intent, node_type):
    if query_intent == QueryIntent.DOCUMENT_REQUEST:
        return {
            "semantic_similarity": 0.4,  # Increased
            "llm_judge": 0.2,
            "entity_match": 0.1,         # Decreased
            "graph_centrality": 0.2,
            "data_type_priority": 0.1
        }
    elif query_intent == QueryIntent.PRODUCT_SEARCH:
        return {
            "semantic_similarity": 0.25,
            "llm_judge": 0.2,
            "entity_match": 0.35,        # Increased
            "graph_centrality": 0.1,
            "data_type_priority": 0.1
        }
    # ... more intent-specific weights
```

## 🏗️ Implementation Strategy

### Phase 1: Basic Implementation
1. Implement semantic similarity scorer (use existing embeddings)
2. Implement simple entity match scorer
3. Implement rule-based data type priority
4. Test with basic weighted combination

### Phase 2: Advanced Features
1. Add LLM judge scorer with fallback
2. Implement graph centrality calculations
3. Add adaptive weighting based on query intent
4. Optimize weights based on evaluation metrics

### Phase 3: Optimization
1. A/B test different weight combinations
2. Add learning component to improve weights over time
3. Implement caching for expensive operations (LLM calls)
4. Add explainability features

## 🧪 Evaluation Approach

### Human Evaluation Dataset
- Create 100 query-subgraph pairs
- Have humans rate node relevance (0-5 scale)
- Use as ground truth for tuning weights

### Metrics
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **Precision@k**: Precision at top k results
- **Human Agreement**: Correlation with human ratings

### A/B Testing
- Test different weight combinations
- Measure user satisfaction with results
- Track click-through rates on top results

## 🔧 Implementation Interface

```python
class RelevanceScorer:
    def __init__(self, embedding_model, llm_client, neo4j_driver):
        self.components = {
            "semantic_similarity": SemanticSimilarityScorer(embedding_model),
            "llm_judge": LLMJudgeScorer(llm_client),
            "entity_match": EntityMatchScorer(),
            "graph_centrality": GraphCentralityScorer(),
            "data_type_priority": DataTypePriorityScorer()
        }
    
    def score_nodes(self, query_context: QueryContext, 
                   node_contexts: List[NodeContext]) -> List[ScoredNode]:
        """
        Main entry point: score and rank all nodes
        
        Returns:
            List of nodes with scores, sorted by relevance
        """
        scored_nodes = []
        
        for node_context in node_contexts:
            total_score = 0.0
            component_scores = {}
            reasoning = []
            
            # Get adaptive weights for this query/node combination
            weights = self.get_adaptive_weights(
                query_context.query_intent, 
                node_context.node_type
            )
            
            # Score with each component
            for component_name, component in self.components.items():
                weight = weights.get(component_name, 0.0)
                if weight > 0:
                    score, reasons = component.score(query_context, node_context)
                    component_scores[component_name] = score
                    reasoning.extend(reasons)
                    total_score += score * weight
            
            scored_nodes.append(ScoredNode(
                node_context=node_context,
                total_score=total_score,
                component_scores=component_scores,
                reasoning=reasoning
            ))
        
        # Sort by total score (descending)
        return sorted(scored_nodes, key=lambda x: x.total_score, reverse=True)
```

## 🚀 Next Steps

1. **Choose starting approach**: I recommend starting with semantic similarity + entity matching + data type priority (simpler, faster to implement)

2. **Decide on query input**: Should we modify the existing query parsing to include intent detection?

3. **Define evaluation criteria**: How will we know if the relevance scoring is working well?

4. **Implementation order**: Which component should we implement first?

What do you think about this design? Should we proceed with the implementation, or would you like to discuss any specific aspects in more detail? 