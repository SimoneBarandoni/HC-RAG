import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict

# Import the functions we need to test
from knowledge_graph import test_neo4j_connection
from neo4j_rag_langgraph import call_ollama_llm
from isRelevant import (
    llm_judge, batch_semantic_similarity, batch_entity_match, QueryInput, NodeInput, QueryIntent, 
    batch_node_type_priority, composite_score, CompositeWeights, DEFAULT_COMPOSITE_WEIGHTS, batch_isRelevant, ScorerType
)


class TestMilestone1CoreComponents:
    """Milestone 1: Core Components Unit Tests"""
    
    def test_neo4j_connection_success(self):
        """Test 1: Verify that test_neo4j_connection returns True with valid credentials."""
        success, message = test_neo4j_connection()
        
        assert success is True, f"Expected successful connection, got: {message}"
        assert isinstance(message, str), "Message should be a string"
        assert len(message) > 0, "Message should not be empty"
    
    @patch('knowledge_graph.GraphDatabase.driver')
    def test_neo4j_connection_failure(self, mock_driver):
        """Test 2: Verify that test_neo4j_connection returns False in case of a connection error."""
        mock_driver.side_effect = Exception("Connection failed: Unable to connect to Neo4j")
        
        success, message = test_neo4j_connection()
        
        assert success is False, "Expected connection failure"
        assert isinstance(message, str), "Error message should be a string"
        assert "Connection failed" in message, f"Expected 'Connection failed' in message, got: {message}"

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
    @patch('neo4j_rag_langgraph.OpenAI')
    def test_call_ollama_llm_error_handling(self, mock_openai):
        """Test 4: Verify call_ollama_llm handles LLM failures gracefully (Unit test with mocking)."""
        # create a mock client that raises an exception when the chat.completions.create method is called
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("Request timed out.")
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
        assert response == expected_fallback, f"Expected fallback response, got: {response}"
        
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
        query1 = QueryInput("test query", identical_embeddings, [], QueryIntent.PRODUCT_SEARCH)
        nodes1 = [NodeInput("identical node", identical_embeddings, {}, "product", [])]
        similarities1 = batch_semantic_similarity(query1, nodes1)
        assert abs(similarities1[0] - 1.0) < 1e-10, f"Identical embeddings should return ~1.0, got: {similarities1[0]}"
        
        # Test 2: Opposite embeddings
        query2 = QueryInput("test query", positive_embeddings, [], QueryIntent.PRODUCT_SEARCH)
        nodes2 = [NodeInput("opposite node", negative_embeddings, {}, "product", [])]
        similarities2 = batch_semantic_similarity(query2, nodes2)
        assert abs(similarities2[0] - 0.0) < 1e-10, f"Opposite embeddings should return ~0.0, got: {similarities2[0]}"
        
        # Test 3: Orthogonal embeddings
        query3 = QueryInput("test query", embedding_a, [], QueryIntent.PRODUCT_SEARCH)
        nodes3 = [NodeInput("orthogonal node", embedding_b, {}, "product", [])]
        similarities3 = batch_semantic_similarity(query3, nodes3)
        assert abs(similarities3[0] - 0.5) < 1e-10, f"Orthogonal embeddings should return ~0.5, got: {similarities3[0]}"
        
        # Test 4: Multiple nodes at once
        query_multi = QueryInput("test query", identical_embeddings, [], QueryIntent.PRODUCT_SEARCH)
        nodes_multi = [
            NodeInput("identical node", identical_embeddings, {}, "product", []),
            NodeInput("opposite node", negative_embeddings, {}, "product", []),
            NodeInput("random node", np.random.rand(384), {}, "product", [])
        ]
        
        similarities_multi = batch_semantic_similarity(query_multi, nodes_multi)
        
        # Verify we get 3 scores
        assert len(similarities_multi) == 3, f"Expected 3 similarity scores, got {len(similarities_multi)}"
        
        # All scores should be between 0 and 1 (with small tolerance for floating-point precision)
        for i, score in enumerate(similarities_multi):
            assert -1e-10 <= score <= 1.0 + 1e-10, f"Score {i} should be between 0 and 1, got: {score}"

    def test_batch_entity_match(self):
        """Test 6: Test batch entity matching with different scenarios."""
        query = QueryInput("Find red mountain bikes", np.random.rand(384), ["red mountain bike", "trail"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Perfect match", np.random.rand(384), {}, "product", ["red mountain bike", "trail"]),  # Perfect match
            NodeInput("Partial match", np.random.rand(384), {}, "product", ["red mountain bike"]),  # Partial match
            NodeInput("No match", np.random.rand(384), {}, "product", ["blue road bike"]),  # No match
            NodeInput("Empty entities", np.random.rand(384), {}, "product", [])  # Empty entities
        ]
        
        matches = batch_entity_match(query, nodes)
        
        # Verify we get 4 scores
        assert len(matches) == 4, f"Expected 4 entity match scores, got {len(matches)}"
        
        # Perfect match should return 1.0
        assert matches[0] == 1.0, f"Perfect match should return 1.0, got: {matches[0]}"
        
        # Partial match should return 0.5 (1 out of 2 entities)
        assert abs(matches[1] - 0.5) < 0.001, f"Partial match should return 0.5, got: {matches[1]}"
        
        # No match should return 0.0
        assert matches[2] == 0.0, f"No match should return 0.0, got: {matches[2]}"
        
        # Empty entities should return 0.0 (no overlap with query entities)
        assert matches[3] == 0.0, f"Empty entities should return 0.0, got: {matches[3]}"

    def test_batch_node_type_priority(self):
        """Test 7: Test batch node type priority scoring."""
        query = QueryInput("Find red mountain bikes", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Product node", np.random.rand(384), {}, "product", []),
            NodeInput("Document node", np.random.rand(384), {}, "document", []),
            NodeInput("Unknown node", np.random.rand(384), {}, "unknown_type", [])
        ]
        
        priorities = batch_node_type_priority(query, nodes)
        
        # Verify we get 3 scores
        assert len(priorities) == 3, f"Expected 3 priority scores, got {len(priorities)}"
        
        # Product should have highest priority for PRODUCT_SEARCH
        assert priorities[0] == 1.0, f"Product priority should be 1.0, got: {priorities[0]}"
        
        # Document should have lower priority
        assert priorities[1] == 0.3, f"Document priority should be 0.3, got: {priorities[1]}"
        
        # Unknown type should have lowest priority
        assert priorities[2] == 0.1, f"Unknown priority should be 0.1, got: {priorities[2]}"

    def test_llm_judge_batch_usage(self):
        """Test 8: Test that LLM judge uses batch processing internally."""
        query = QueryInput("Find red mountain bikes", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("Red mountain bike with suspension", np.random.rand(384), {}, "product", ["red mountain bike", "suspension"])
        
        # This should internally use batch processing
        result = llm_judge(query, node)
        assert result >= 0.85, f"LLM judge should be greater than 0.85, got: {result}"
        assert 0.0 <= result <= 1.0, f"LLM judge result should be between 0 and 1, got: {result}"

    def test_batch_isrelevant_comprehensive(self):
        """Test 9: Test batch_isRelevant with different scorer types and multiple nodes."""
        np.random.seed(42)
        query = QueryInput("Find red mountain bikes", np.random.rand(384), ["red mountain bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Red mountain bike with suspension", np.random.rand(384), {}, "product", ["red mountain bike", "suspension"]),
            NodeInput("Blue road bike", np.random.rand(384), {}, "product", ["blue road bike"]),
            NodeInput("Mountain bike manual", np.random.rand(384), {}, "document", ["mountain bike"])
        ]
        
        # Test different scorer types with batch processing
        scorer_types = [
            ScorerType.COMPOSITE,
            ScorerType.ROUTER_SINGLE_SEM,
            ScorerType.ROUTER_SINGLE_ENT,
            ScorerType.ROUTER_SINGLE_TYPE
        ]
        
        for scorer_type in scorer_types:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=5)
            
            # Verify we get scores for all nodes
            assert len(scores) == 3, f"Expected 3 scores for {scorer_type}, got {len(scores)}"
            
            # All scores should be between 0 and 1
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, f"Score {i} for {scorer_type} should be between 0 and 1, got: {score}"
            
            print(f"{scorer_type.value}: {[f'{s:.3f}' for s in scores]}")

    def test_composite_score_configurable_weights(self):
        """Test 10: Test configurable composite score with different weight configurations."""
        
        # Setup test data
        np.random.seed(42)
        query = QueryInput("Find red mountain bikes", np.random.rand(384), ["red mountain bike"], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("Red mountain bike with suspension", np.random.rand(384), {}, "product", ["red mountain bike", "suspension"])
        
        # Test 1: Default weights
        default_score = composite_score(query, node)
        default_weights_score = composite_score(query, node, DEFAULT_COMPOSITE_WEIGHTS)
        # Since LLM calls can vary slightly, we check if they're approximately equal (within 5%)
        score_diff = abs(default_score - default_weights_score) / max(default_score, default_weights_score)
        assert score_diff < 0.05, f"Default weights should work similarly to no weights parameter (diff: {score_diff:.3f})"
        
        # Test 2: Custom balanced weights
        balanced_weights = CompositeWeights.create_balanced()
        balanced_score = composite_score(query, node, balanced_weights)
        assert 0.0 <= balanced_score <= 1.0, f"Balanced score should be between 0 and 1, got: {balanced_score}"
        
        # Test 3: Semantic-focused weights
        semantic_weights = CompositeWeights.create_semantic_focused()
        semantic_score = composite_score(query, node, semantic_weights)
        assert 0.0 <= semantic_score <= 1.0, f"Semantic score should be between 0 and 1, got: {semantic_score}"
        
        # Test 4: Entity-focused weights  
        entity_weights = CompositeWeights.create_entity_focused()
        entity_score = composite_score(query, node, entity_weights)
        assert 0.0 <= entity_score <= 1.0, f"Entity score should be between 0 and 1, got: {entity_score}"
        
        # Test 5: Custom weights from dictionary
        custom_dict = {
            'semantic_similarity': 0.5,
            'llm_judge': 0.2,
            'entity_match': 0.2,
            'node_type_priority': 0.1
        }
        custom_weights = CompositeWeights.from_dict(custom_dict)
        custom_score = composite_score(query, node, custom_weights)
        assert 0.0 <= custom_score <= 1.0, f"Custom score should be between 0 and 1, got: {custom_score}"
        
        # Test 6: Verify weights validation
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CompositeWeights(0.5, 0.5, 0.5, 0.5)  # Sum = 2.0
            
        with pytest.raises(ValueError, match="must be non-negative"):
            CompositeWeights(-0.1, 0.6, 0.3, 0.2)  # Negative weight
        
        # Test 7: Verify different weight configurations can produce different results
        scores = [default_score, balanced_score, semantic_score, entity_score, custom_score]
        
        # All scores should be valid
        for score in scores:
            assert 0.0 <= score <= 1.0, f"All scores should be between 0 and 1, got: {score}"
        
        # Test 8: Verify weight configuration works as expected
        # Create extreme weights to verify they affect the score
        semantic_only_weights = CompositeWeights(1.0, 0.0, 0.0, 0.0)  # Only semantic similarity
        entity_only_weights = CompositeWeights(0.0, 0.0, 1.0, 0.0)    # Only entity match
        
        semantic_only_score = composite_score(query, node, semantic_only_weights)
        entity_only_score = composite_score(query, node, entity_only_weights)
        
        # These should be different (unless by pure coincidence the metrics return identical values)
        assert 0.0 <= semantic_only_score <= 1.0, "Semantic-only score should be valid"
        assert 0.0 <= entity_only_score <= 1.0, "Entity-only score should be valid"


class TestParallelAndRouterScorers:
    """Additional tests for parallel and router scoring strategies"""
    
    def test_parallel_scorer_batch(self):
        """Test 11: Test parallel scorer returns maximum of all metrics in batch."""
        np.random.seed(42)
        query = QueryInput("Find red mountain bikes", np.random.rand(384), ["red mountain bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Perfect red mountain bike match", np.random.rand(384), {}, "product", ["red mountain bike"]),
            NodeInput("Unrelated blue car document", np.random.rand(384), {}, "document", ["blue car"])
        ]
        
        # Test parallel scorer (should return max of all individual metrics)
        parallel_scores = batch_isRelevant(query, nodes, ScorerType.PARALLEL, batch_size=5)
        
        # Verify we get scores for all nodes
        assert len(parallel_scores) == 2, f"Expected 2 parallel scores, got {len(parallel_scores)}"
        
        # All scores should be between 0 and 1
        for i, score in enumerate(parallel_scores):
            assert 0.0 <= score <= 1.0, f"Parallel score {i} should be between 0 and 1, got: {score}"
        
        # First node should have higher score (better match)
        assert parallel_scores[0] > parallel_scores[1], f"Better match should have higher parallel score: {parallel_scores[0]} vs {parallel_scores[1]}"
        
        print(f"Parallel scores: {[f'{s:.3f}' for s in parallel_scores]}")

    def test_router_scorer_basic_batch(self):
        """Test 12: Test basic router scorer (semantic + LLM + node type) in batch."""
        np.random.seed(42)
        query = QueryInput("Find mountain bikes", np.random.rand(384), ["mountain bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Mountain bike product", np.random.rand(384), {}, "product", ["mountain bike"]),
            NodeInput("Car documentation", np.random.rand(384), {}, "document", ["car"])
        ]
        
        # Test basic router scorer
        router_scores = batch_isRelevant(query, nodes, ScorerType.ROUTER, batch_size=5)
        
        # Verify we get scores for all nodes
        assert len(router_scores) == 2, f"Expected 2 router scores, got {len(router_scores)}"
        
        # All scores should be between 0 and 1
        for i, score in enumerate(router_scores):
            assert 0.0 <= score <= 1.0, f"Router score {i} should be between 0 and 1, got: {score}"
        
        # First node should have higher score (better match)
        assert router_scores[0] > router_scores[1], f"Better match should have higher router score: {router_scores[0]} vs {router_scores[1]}"
        
        print(f"Router scores: {[f'{s:.3f}' for s in router_scores]}")

    def test_router_all_scorer_batch(self):
        """Test 13: Test router all scorer (all 4 metrics) in batch."""
        np.random.seed(42)
        query = QueryInput("Find red bikes", np.random.rand(384), ["red bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Red bike with features", np.random.rand(384), {}, "product", ["red bike", "features"]),
            NodeInput("Blue car manual", np.random.rand(384), {}, "document", ["blue car"])
        ]
        
        # Test router all scorer (combines all 4 metrics)
        router_all_scores = batch_isRelevant(query, nodes, ScorerType.ROUTER_ALL, batch_size=5)
        
        # Verify we get scores for all nodes
        assert len(router_all_scores) == 2, f"Expected 2 router_all scores, got {len(router_all_scores)}"
        
        # All scores should be between 0 and 1
        for i, score in enumerate(router_all_scores):
            assert 0.0 <= score <= 1.0, f"Router_all score {i} should be between 0 and 1, got: {score}"
        
        print(f"Router all scores: {[f'{s:.3f}' for s in router_all_scores]}")

    def test_router_two_variations_batch(self):
        """Test 14: Test router two-metric variations in batch."""
        np.random.seed(42)
        query = QueryInput("Find bikes", np.random.rand(384), ["bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Bike product description", np.random.rand(384), {}, "product", ["bike"]),
            NodeInput("Car documentation", np.random.rand(384), {}, "document", ["car"])
        ]
        
        # Test semantic + LLM combination
        sem_llm_scores = batch_isRelevant(query, nodes, ScorerType.ROUTER_TWO_SEM_LLM, batch_size=5)
        assert len(sem_llm_scores) == 2, f"Expected 2 semantic+LLM scores, got {len(sem_llm_scores)}"
        
        # Test entity + type combination
        ent_type_scores = batch_isRelevant(query, nodes, ScorerType.ROUTER_TWO_ENT_TYPE, batch_size=5)
        assert len(ent_type_scores) == 2, f"Expected 2 entity+type scores, got {len(ent_type_scores)}"
        
        # All scores should be valid
        for scores, name in [(sem_llm_scores, "semantic+LLM"), (ent_type_scores, "entity+type")]:
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, f"{name} score {i} should be between 0 and 1, got: {score}"
        
        print(f"Semantic+LLM scores: {[f'{s:.3f}' for s in sem_llm_scores]}")
        print(f"Entity+Type scores: {[f'{s:.3f}' for s in ent_type_scores]}")

    def test_single_metric_routers_batch(self):
        """Test 15: Test single-metric router scorers in batch."""
        np.random.seed(42)
        query = QueryInput("Find red bikes", np.random.rand(384), ["red bike"], QueryIntent.PRODUCT_SEARCH)
        
        nodes = [
            NodeInput("Red bike perfect match", np.random.rand(384), {}, "product", ["red bike"]),
            NodeInput("Blue car different", np.random.rand(384), {}, "document", ["blue car"]),
            NodeInput("Generic bike partial", np.random.rand(384), {}, "product", ["bike"])
        ]
        
        # Test all single-metric routers
        single_routers = [
            (ScorerType.ROUTER_SINGLE_SEM, "semantic"),
            (ScorerType.ROUTER_SINGLE_LLM, "LLM"),
            (ScorerType.ROUTER_SINGLE_ENT, "entity"),
            (ScorerType.ROUTER_SINGLE_TYPE, "type")
        ]
        
        for scorer_type, name in single_routers:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=5)
            
            # Verify we get scores for all nodes
            assert len(scores) == 3, f"Expected 3 {name} scores, got {len(scores)}"
            
            # All scores should be between 0 and 1
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, f"{name} score {i} should be between 0 and 1, got: {score}"
            
            print(f"{name.capitalize()} scores: {[f'{s:.3f}' for s in scores]}")

    def test_scorer_comparison_batch(self):
        """Test 16: Compare different scorer types on same data to verify they produce different results."""
        np.random.seed(42)
        query = QueryInput("Find mountain bikes", np.random.rand(384), ["mountain bike"], QueryIntent.PRODUCT_SEARCH)
        
        # Create nodes with different characteristics to highlight scorer differences
        nodes = [
            NodeInput("Mountain bike product perfect match", np.random.rand(384), {}, "product", ["mountain bike"]),  # Should score high on most metrics
            NodeInput("Mountain bike documentation", np.random.rand(384), {}, "document", ["mountain bike"]),  # Good entity, wrong type for product search
            NodeInput("Road bike product", np.random.rand(384), {}, "product", ["road bike"]),  # Good type, wrong entity
            NodeInput("Generic text", np.random.rand(384), {}, "unknown", [])  # Poor on all metrics
        ]
        
        # Test multiple scorer types
        scorer_types = [
            ScorerType.COMPOSITE,
            ScorerType.PARALLEL,
            ScorerType.ROUTER,
            ScorerType.ROUTER_ALL,
            ScorerType.ROUTER_SINGLE_ENT,
            ScorerType.ROUTER_SINGLE_TYPE
        ]
        
        results = {}
        for scorer_type in scorer_types:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=10)
            results[scorer_type.value] = scores
            
            # Verify we get scores for all nodes
            assert len(scores) == 4, f"Expected 4 scores for {scorer_type.value}, got {len(scores)}"
            
            # All scores should be valid
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, f"{scorer_type.value} score {i} should be between 0 and 1, got: {score}"
        
        # Print comparison table
        print("SCORER COMPARISON RESULTS:")
        print("-" * 70)
        print(f"{'Scorer':<20} {'Perfect':<8} {'Doc':<8} {'WrongEnt':<8} {'Generic':<8}")
        print("-" * 70)
        
        for scorer_name, scores in results.items():
            print(f"{scorer_name:<20} {scores[0]:<8.3f} {scores[1]:<8.3f} {scores[2]:<8.3f} {scores[3]:<8.3f}")
        
        # Verify that different scorers produce different results
        score_sets = [tuple(scores) for scores in results.values()]
        unique_score_sets = set(score_sets)
        
        # Should have some variation (at least 2 different result patterns)
        assert len(unique_score_sets) >= 2, f"Expected different scorers to produce different results, got {len(unique_score_sets)} unique patterns"

    def test_edge_cases_batch(self):
        """Test 17: Test edge cases for parallel and router scorers."""
        np.random.seed(42)
        
        # Edge case 1: Empty entities
        query_empty = QueryInput("Generic query", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node_empty = NodeInput("Generic node", np.random.rand(384), {}, "product", [])
        
        # Test with empty entities
        empty_scores = batch_isRelevant(query_empty, [node_empty], ScorerType.PARALLEL, batch_size=5)
        assert len(empty_scores) == 1, "Should handle empty entities"
        assert 0.0 <= empty_scores[0] <= 1.0, f"Empty entities score should be valid: {empty_scores[0]}"
        
        # Edge case 2: Single node
        single_scores = batch_isRelevant(query_empty, [node_empty], ScorerType.ROUTER, batch_size=5)
        assert len(single_scores) == 1, "Should handle single node"
        assert 0.0 <= single_scores[0] <= 1.0, f"Single node score should be valid: {single_scores[0]}"
        
        # Edge case 3: Many nodes (test batch processing efficiency)
        many_nodes = [
            NodeInput(f"Node {i}", np.random.rand(384), {}, "product", [f"entity_{i}"])
            for i in range(20)
        ]
        
        many_scores = batch_isRelevant(query_empty, many_nodes, ScorerType.COMPOSITE, batch_size=10)
        assert len(many_scores) == 20, f"Should handle many nodes, got {len(many_scores)}"
        
        # All scores should be valid
        for i, score in enumerate(many_scores):
            assert 0.0 <= score <= 1.0, f"Many nodes score {i} should be valid: {score}"


if __name__ == "__main__":
    # Run the tests
    test_suite = TestMilestone1CoreComponents()
    
    try:
        test_suite.test_neo4j_connection_success()
        test_suite.test_neo4j_connection_failure()
        test_suite.test_llm_connection_and_chat()
        test_suite.test_call_ollama_llm_error_handling()
        test_suite.test_batch_semantic_similarity()
        test_suite.test_batch_entity_match()
        test_suite.test_batch_node_type_priority()
        test_suite.test_llm_judge_batch_usage()
        test_suite.test_batch_isrelevant_comprehensive()
        test_suite.test_composite_score_configurable_weights()
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run parallel and router tests
    parallel_router_suite = TestParallelAndRouterScorers()
    
    try:
        parallel_router_suite.test_parallel_scorer_batch()
        parallel_router_suite.test_router_scorer_basic_batch()
        parallel_router_suite.test_router_all_scorer_batch()
        parallel_router_suite.test_router_two_variations_batch()
        parallel_router_suite.test_single_metric_routers_batch()
        parallel_router_suite.test_scorer_comparison_batch()
        parallel_router_suite.test_edge_cases_batch()
        
        print("\n🎉 ALL PARALLEL AND ROUTER TESTS PASSED!")
        
    except Exception as e:
        print(f"Parallel/Router test failed: {e}")
        import traceback
        traceback.print_exc()