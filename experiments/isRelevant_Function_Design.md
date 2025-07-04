# isRelevant Function Design Specification

## Overview

The `isRelevant` function serves as a critical component in our RAG (Retrieval-Augmented Generation) system, designed to determine the relevance score of knowledge graph nodes in relation to user queries. This function enables intelligent node ranking and selection for context inclusion, optimizing the quality of information provided to the language model while respecting context window constraints.

## Function Purpose

The primary objective of the `isRelevant` function is to quantify how relevant a specific node within the currently identified subgraph is to the user's query. This relevance scoring enables the system to:

- Rank nodes by their importance to the query
- Select the most relevant nodes for RAG context inclusion
- Optimize context window utilization based on resource length and relevance scores
- Provide consistent and accurate relevance assessment across different query types and node varieties

## Function Signature

```python
def isRelevant(
    query: QueryInput,
    node: NodeInput,
    scorer_type: ScorerType
) -> float:
    """
    Calculate relevance score for a graph node relative to user query.
    
    Returns: Relevance score (typically 0.0 to 1.0)
    """
```

## Input Components

### Query Input Components

The query component encompasses all processed information about the user's request:

#### 1. Query Text
- **Type**: String
- **Description**: Original, unprocessed text of the user query
- **Purpose**: Provides raw semantic context for text-based scoring methods

#### 2. Query Embeddings
- **Type**: Vector (typically float array)
- **Description**: Vector representation of the query text generated using embedding models
- **Purpose**: Enables semantic similarity calculations between query and nodes

#### 3. Query Parsed Entities
- **Type**: Structured object/dictionary
- **Description**: Named entities extracted from the query through NER, LLM parsing, or other extraction methods
- **Format Example**: 
  ```json
  {
    "product": {
      "features": ["red", "lightweight"],
      "category": "bikes"
    },
    "document": {
      "type": "manual"
    }
  }
  ```
- **Purpose**: Enables precise entity matching between query requirements and node content

#### 4. Query Intent/Class
- **Type**: Enumerated label
- **Description**: Classified intent or category of the user query
- **Example Values**: 
  - `PRODUCT_SEARCH`
  - `DOCUMENT_REQUEST`
  - `TECHNICAL_SUPPORT`
  - `COMPARISON_REQUEST`
  - `SPECIFICATION_INQUIRY`
- **Purpose**: Guides scorer selection and metric weighting based on query type

### Node Input Components

The node component contains all available information about a knowledge graph node:

#### 1. Node Text/Database Entry
- **Type**: String or structured data
- **Description**: Original textual content or database record associated with the node
- **Variants**:
  - Raw text for document nodes
  - Structured database entries for product/category nodes
  - Extracted table data for specification nodes
- **Purpose**: Provides content for text-based analysis and LLM evaluation

#### 2. Node Embeddings
- **Type**: Vector (typically float array)
- **Description**: Vector representation of the node's content
- **Purpose**: Enables semantic similarity calculations with query embeddings

#### 3. Node Graph Relations
- **Type**: Graph structure data
- **Description**: Information about the node's position and connections within the knowledge graph
- **Components**:
  - Incoming/outgoing relationships
  - Relationship types and strengths
  - Node centrality measures
  - Connected node types and counts
- **Purpose**: Enables graph-based relevance scoring through centrality and relationship analysis

### Scorer Selection
- **Type**: Enumerated type
- **Options**: `COMPOSITE`, `PARALLEL`, `ROUTER`
- **Description**: Determines the scoring strategy and metric combination approach

## Scorer Types

The system implements three distinct scorer types, each orchestrating the five core metrics differently:

### 1. Composite Scorer
- **Strategy**: Weighted combination of all five metrics
- **Calculation**: `relevance = Σ(weight_i × metric_i)`
- **Characteristics**:
  - All metrics contribute to final score
  - Weights can be tuned based on domain requirements
  - Provides balanced consideration of all relevance factors
  - Suitable for general-purpose relevance assessment

### 2. Parallel Scorer
- **Strategy**: Maximum score across all five metrics
- **Calculation**: `relevance = max(metric_1, metric_2, metric_3, metric_4, metric_5)`
- **Characteristics**:
  - Optimistic scoring approach
  - Node needs to excel in only one dimension
  - Useful when different nodes have distinct strengths
  - Prevents good nodes from being penalized by weak dimensions

### 3. Router Scorer
- **Strategy**: Conditional metric selection based on query type and node characteristics
- **Calculation**: Dynamic selection and combination of relevant metrics
- **Characteristics**:
  - Adaptive scoring based on context
  - Different metric combinations for different scenarios
  - Most sophisticated but requires careful configuration
  - Optimizes for query-specific relevance patterns

## Core Scoring Metrics

The system employs five fundamental metrics to assess node relevance:

### 1. Semantic Similarity
- **Input**: Query embeddings, Node embeddings
- **Method**: Vector similarity calculation (cosine similarity, dot product, etc.)
- **Strength**: Captures semantic relationships and conceptual alignment
- **Use Case**: Universal applicability across all content types

### 2. LLM as a Judge
- **Input**: Query text, Node text/database content
- **Method**: Language model evaluation of relevance
- **Strength**: Sophisticated understanding of context and nuanced relationships
- **Use Case**: Complex queries requiring reasoning and interpretation

### 3. Entity Match
- **Input**: Query parsed entities, Node content
- **Method**: Structured comparison of extracted entities
- **Strength**: Precise matching of specific requirements (features, categories, etc.)
- **Use Case**: Product searches with specific criteria

### 4. Graph Centrality
- **Input**: Node graph relations
- **Method**: Centrality measures (degree, betweenness, PageRank, etc.)
- **Strength**: Identifies important nodes within the graph structure
- **Use Case**: Finding authoritative or well-connected information sources

### 5. Data Type Priority
- **Input**: Node content type, Query intent/class, Priority matrix
- **Method**: Lookup-based scoring using predefined priority relationships
- **Strength**: Domain-specific relevance based on query-content type matching
- **Use Case**: Ensuring appropriate content types are prioritized for specific query intents

## System Integration

### Orchestration Service
The `isRelevant` function operates within a broader orchestration service that:

1. **Node Ranking**: Organizes nodes by relevance scores in descending order
2. **Context Selection**: Determines which nodes to include in RAG context
3. **Resource Management**: Considers node content length and LLM context window constraints
4. **Optimization**: Balances relevance scores with practical context limitations

### Implementation Workflow
1. Query preprocessing and entity extraction
2. Subgraph identification and node collection
3. Relevance scoring for each node using `isRelevant`
4. Node ranking based on scores
5. Context window optimization and final node selection
6. Context assembly for RAG system

## Configuration and Extensibility

### Scorer Configuration
- Composite scorer weights should be configurable
- Router scorer rules should be externally defined
- Metric implementations should be pluggable

### Performance Considerations
- Batch processing capabilities for multiple nodes
- Caching strategies for expensive operations (LLM calls, embedding calculations)
- Scalability for large subgraphs

### Monitoring and Evaluation
- Relevance score distribution tracking
- A/B testing capabilities for different scorer configurations
- Performance metrics for scoring speed and accuracy

## Implementation Guidelines

1. **Modular Design**: Each metric should be implemented as a separate, testable component
2. **Error Handling**: Graceful degradation when specific metrics fail
3. **Logging**: Comprehensive logging for debugging and optimization
4. **Testing**: Unit tests for individual metrics and integration tests for scorer combinations
5. **Documentation**: Clear documentation of scorer configurations and tuning parameters

This design specification provides the foundation for implementing a robust, flexible, and scalable relevance scoring system that can adapt to various query types and content characteristics while maintaining high performance and accuracy. 