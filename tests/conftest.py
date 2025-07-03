import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the functions we need to test
from knowledge_graph import test_neo4j_connection
from neo4j_rag_langgraph import app, call_ollama_llm, analyze_query, evaluate_context, generate_answer, revise_question, sample_neo4j_nodes, score_semantic_similarity, RetrievalState, expand_subgraph, score_expanded_nodes_with_isrelevant
from isRelevant import (
    llm_judge, batch_semantic_similarity, batch_entity_match, QueryInput, NodeInput, QueryIntent, 
    batch_node_type_priority, composite_score, CompositeWeights, DEFAULT_COMPOSITE_WEIGHTS, batch_isRelevant, ScorerType
)

# Common fixtures that can be shared across test files
@pytest.fixture
def sample_query():
    """Fixture providing a sample query for testing."""
    return QueryInput(
        text="Find red mountain bikes under $1000",
        embeddings=np.random.rand(384),
        entities=["red mountain bike"],
        intent=QueryIntent.PRODUCT_SEARCH
    )

@pytest.fixture
def sample_nodes():
    """Fixture providing sample nodes for testing."""
    return [
        NodeInput("Red mountain bike", np.random.rand(384), {}, "product", ["red mountain bike"]),
        NodeInput("Blue road bike", np.random.rand(384), {}, "product", ["blue road bike"]),
    ]

@pytest.fixture
def mock_neo4j_connection():
    """Fixture providing mocked Neo4j connection."""
    with patch('knowledge_graph.GraphDatabase.driver') as mock_driver:
        yield mock_driver

@pytest.fixture
def mock_llm_client():
    """Fixture providing mocked LLM client."""
    with patch('neo4j_rag_langgraph.OpenAI') as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client 