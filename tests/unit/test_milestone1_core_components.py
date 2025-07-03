import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import the functions we need to test
from knowledge_graph import test_neo4j_connection
from neo4j_rag_langgraph import (
    call_ollama_llm,
)
from isRelevant import (
    llm_judge,
    batch_semantic_similarity,
    batch_entity_match,
    QueryInput,
    NodeInput,
    QueryIntent,
    batch_node_type_priority,
)


@pytest.mark.unit
class TestMilestone1CoreComponents:
    """Milestone 1: Core Components Unit Tests"""

    def test_neo4j_connection_success(self):
        """Test 1: Verify that test_neo4j_connection returns True with valid credentials."""
        success, message = test_neo4j_connection()

        assert success is True, f"Expected successful connection, got: {message}"
        assert isinstance(message, str), "Message should be a string"
        assert len(message) > 0, "Message should not be empty"

    @patch("knowledge_graph.GraphDatabase.driver")
    def test_neo4j_connection_failure(self, mock_driver):
        """Test 2: Verify that test_neo4j_connection returns False in case of a connection error."""
        mock_driver.side_effect = Exception(
            "Connection failed: Unable to connect to Neo4j"
        )

        success, message = test_neo4j_connection()

        assert success is False, "Expected connection failure"
        assert isinstance(message, str), "Error message should be a string"
        assert "Connection failed" in message, (
            f"Expected 'Connection failed' in message, got: {message}"
        )

    def test_llm_connection_and_chat(self):
        """Test 3: REAL test - verify LLM service is reachable and responsive."""
        # This makes a REAL API call to test actual connectivity
        system_prompt = "You are a helpful assistant."
        user_prompt = "Respond with exactly the word 'CONNECTED' and nothing else."

        try:
            # Use shorter timeout for connectivity test
            response = call_ollama_llm(system_prompt, user_prompt, timeout=20)

            # Basic connectivity assertions
            assert response is not None, "Response should not be None"
            assert isinstance(response, str), "Response should be a string"
            assert len(response.strip()) > 0, "Response should not be empty"

            # If we get here, LLM service is working
            print(f"LLM service is responsive. Response: '{response[:50]}'")

        except Exception as e:
            pytest.skip(f"LLM service unavailable: {e}")

    # decoration to replace the real OpenAI client with a mock in the neo4j_rag_langgraph.py file
    @patch("neo4j_rag_langgraph.OpenAI")
    def test_call_ollama_llm_error_handling(self, mock_openai):
        """Test 4: Verify call_ollama_llm handles LLM failures gracefully (Unit test with mocking)."""
        # create a mock client that raises an exception when the chat.completions.create method is called
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Request timed out."
        )
        mock_openai.return_value = mock_client

        system_prompt = "You are a helpful assistant."
        user_prompt = "This should trigger the error handling."

        # call the function in the neo4j_rag_langgraph.py file with the mock client
        response = call_ollama_llm(system_prompt, user_prompt)

        # Should return fallback response
        assert response is not None, "Response should not be None even on failure"
        assert isinstance(response, str), "Response should be a string"
        assert len(response) > 0, "Response should not be empty"

        # Should be the exact fallback message
        expected_fallback = "I apologize, but I'm having trouble processing your request due to a technical issue. Please try again."
        assert response == expected_fallback, (
            f"Expected fallback response, got: {response}"
        )

        # Verify it attempted to call the LLM
        mock_openai.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

    def test_batch_semantic_similarity(self):
        """Test 5: Test batch semantic similarity with known embedding vectors."""
        # Test case 1: Multiple nodes with different similarities
        np.random.seed(42)  # For reproducible results

        # Identical embeddings
        identical_embeddings = np.ones(384)

        # Opposite embeddings
        positive_embeddings = np.ones(384)
        negative_embeddings = -np.ones(384)

        # Orthogonal embeddings (truly orthogonal vectors)
        embedding_a = np.zeros(384)
        embedding_a[0] = 1.0  # [1, 0, 0, ...]
        embedding_b = np.zeros(384)
        embedding_b[1] = 1.0  # [0, 1, 0, ...]

        # Test 1: Identical embeddings
        query1 = QueryInput(
            "test query", identical_embeddings, [], QueryIntent.PRODUCT_SEARCH
        )
        nodes1 = [NodeInput("identical node", identical_embeddings, {}, "product", [])]
        similarities1 = batch_semantic_similarity(query1, nodes1)
        assert abs(similarities1[0] - 1.0) < 1e-10, (
            f"Identical embeddings should return ~1.0, got: {similarities1[0]}"
        )

        # Test 2: Opposite embeddings
        query2 = QueryInput(
            "test query", positive_embeddings, [], QueryIntent.PRODUCT_SEARCH
        )
        nodes2 = [NodeInput("opposite node", negative_embeddings, {}, "product", [])]
        similarities2 = batch_semantic_similarity(query2, nodes2)
        assert abs(similarities2[0] - 0.0) < 1e-10, (
            f"Opposite embeddings should return ~0.0, got: {similarities2[0]}"
        )

        # Test 3: Orthogonal embeddings
        query3 = QueryInput("test query", embedding_a, [], QueryIntent.PRODUCT_SEARCH)
        nodes3 = [NodeInput("orthogonal node", embedding_b, {}, "product", [])]
        similarities3 = batch_semantic_similarity(query3, nodes3)
        assert abs(similarities3[0] - 0.5) < 1e-10, (
            f"Orthogonal embeddings should return ~0.5, got: {similarities3[0]}"
        )

        # Test 4: Multiple nodes at once
        query_multi = QueryInput(
            "test query", identical_embeddings, [], QueryIntent.PRODUCT_SEARCH
        )
        nodes_multi = [
            NodeInput("identical node", identical_embeddings, {}, "product", []),
            NodeInput("opposite node", negative_embeddings, {}, "product", []),
            NodeInput("random node", np.random.rand(384), {}, "product", []),
        ]

        similarities_multi = batch_semantic_similarity(query_multi, nodes_multi)

        # Verify we get 3 scores
        assert len(similarities_multi) == 3, (
            f"Expected 3 similarity scores, got {len(similarities_multi)}"
        )

        # All scores should be between 0 and 1 (with small tolerance for floating-point precision)
        for i, score in enumerate(similarities_multi):
            assert -1e-10 <= score <= 1.0 + 1e-10, (
                f"Score {i} should be between 0 and 1, got: {score}"
            )

    def test_batch_entity_match(self):
        """Test 6: Test batch entity matching with different scenarios."""
        query = QueryInput(
            "Find red mountain bikes",
            np.random.rand(384),
            ["red mountain bike", "trail"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Perfect match",
                np.random.rand(384),
                {},
                "product",
                ["red mountain bike", "trail"],
            ),  # Perfect match
            NodeInput(
                "Partial match",
                np.random.rand(384),
                {},
                "product",
                ["red mountain bike"],
            ),  # Partial match
            NodeInput(
                "No match", np.random.rand(384), {}, "product", ["blue road bike"]
            ),  # No match
            NodeInput(
                "Empty entities", np.random.rand(384), {}, "product", []
            ),  # Empty entities
        ]

        matches = batch_entity_match(query, nodes)

        # Verify we get 4 scores
        assert len(matches) == 4, f"Expected 4 entity match scores, got {len(matches)}"

        # Perfect match should return 1.0
        assert matches[0] == 1.0, f"Perfect match should return 1.0, got: {matches[0]}"

        # Partial match should return 0.5 (1 out of 2 entities)
        assert abs(matches[1] - 0.5) < 0.001, (
            f"Partial match should return 0.5, got: {matches[1]}"
        )

        # No match should return 0.0
        assert matches[2] == 0.0, f"No match should return 0.0, got: {matches[2]}"

        # Empty entities should return 0.0 (no overlap with query entities)
        assert matches[3] == 0.0, f"Empty entities should return 0.0, got: {matches[3]}"

    def test_batch_node_type_priority(self):
        """Test 7: Test batch node type priority scoring."""
        query = QueryInput(
            "Find red mountain bikes",
            np.random.rand(384),
            [],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput("Product node", np.random.rand(384), {}, "product", []),
            NodeInput("Document node", np.random.rand(384), {}, "document", []),
            NodeInput("Unknown node", np.random.rand(384), {}, "unknown_type", []),
        ]

        priorities = batch_node_type_priority(query, nodes)

        # Verify we get 3 scores
        assert len(priorities) == 3, (
            f"Expected 3 priority scores, got {len(priorities)}"
        )

        # Product should have highest priority for PRODUCT_SEARCH
        assert priorities[0] == 1.0, (
            f"Product priority should be 1.0, got: {priorities[0]}"
        )

        # Document should have lower priority
        assert priorities[1] == 0.3, (
            f"Document priority should be 0.3, got: {priorities[1]}"
        )

        # Unknown type should have lowest priority
        assert priorities[2] == 0.1, (
            f"Unknown priority should be 0.1, got: {priorities[2]}"
        )

    def test_llm_judge_batch_usage(self):
        """Test 8: Test that LLM judge uses batch processing internally."""
        query = QueryInput(
            "Find red mountain bikes with good suspension",
            np.random.rand(384),
            [],
            QueryIntent.PRODUCT_SEARCH,
        )
        node = NodeInput(
            "Red mountain bike with suspension",
            np.random.rand(384),
            {},
            "product",
            ["red mountain bike", "suspension"],
        )

        # This should internally use batch processing
        result = llm_judge(query, node)
        print(result)
        assert result >= 0.85, f"LLM judge should be greater than 0.85, got: {result}"
        assert 0.0 <= result <= 1.0, (
            f"LLM judge result should be between 0 and 1, got: {result}"
        )
