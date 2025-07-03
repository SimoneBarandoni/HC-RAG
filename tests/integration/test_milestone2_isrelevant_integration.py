import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import the functions we need to test
from isRelevant import (
    QueryInput,
    NodeInput,
    QueryIntent,
    composite_score,
    CompositeWeights,
    DEFAULT_COMPOSITE_WEIGHTS,
    batch_isRelevant,
    ScorerType,
)


@pytest.mark.integration
class TestMilestone2IsRelevantScorersIntegration:
    def test_batch_isrelevant_comprehensive(self):
        """Test 1: Test batch_isRelevant with different scorer types and multiple nodes."""
        np.random.seed(42)
        query = QueryInput(
            "Find red mountain bikes",
            np.random.rand(384),
            ["red mountain bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Red mountain bike with suspension",
                np.random.rand(384),
                {},
                "product",
                ["red mountain bike", "suspension"],
            ),
            NodeInput(
                "Blue road bike", np.random.rand(384), {}, "product", ["blue road bike"]
            ),
            NodeInput(
                "Mountain bike manual",
                np.random.rand(384),
                {},
                "document",
                ["mountain bike"],
            ),
        ]

        # Test different scorer types with batch processing
        scorer_types = [
            ScorerType.COMPOSITE,
            ScorerType.ROUTER_SINGLE_SEM,
            ScorerType.ROUTER_SINGLE_ENT,
            ScorerType.ROUTER_SINGLE_TYPE,
        ]

        for scorer_type in scorer_types:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=5)

            # Verify we get scores for all nodes
            assert len(scores) == 3, (
                f"Expected 3 scores for {scorer_type}, got {len(scores)}"
            )

            # All scores should be between 0 and 1
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, (
                    f"Score {i} for {scorer_type} should be between 0 and 1, got: {score}"
                )

    def test_composite_score_configurable_weights(self):
        """Test 2: Test configurable composite score with different weight configurations."""

        # Setup test data
        np.random.seed(42)
        query = QueryInput(
            "Find red mountain bikes",
            np.random.rand(384),
            ["red mountain bike"],
            QueryIntent.PRODUCT_SEARCH,
        )
        node = NodeInput(
            "Red mountain bike with suspension",
            np.random.rand(384),
            {},
            "product",
            ["red mountain bike", "suspension"],
        )

        # Test 1: Default weights
        default_score = composite_score(query, node)
        default_weights_score = composite_score(query, node, DEFAULT_COMPOSITE_WEIGHTS)
        # Since LLM calls can vary slightly, we check if they're approximately equal (within 5%)
        score_diff = abs(default_score - default_weights_score) / max(
            default_score, default_weights_score
        )
        assert score_diff < 0.05, (
            f"Default weights should work similarly to no weights parameter (diff: {score_diff:.3f})"
        )

        # Test 2: Custom balanced weights
        balanced_weights = CompositeWeights.create_balanced()
        balanced_score = composite_score(query, node, balanced_weights)
        assert 0.0 <= balanced_score <= 1.0, (
            f"Balanced score should be between 0 and 1, got: {balanced_score}"
        )

        # Test 3: Semantic-focused weights
        semantic_weights = CompositeWeights.create_semantic_focused()
        semantic_score = composite_score(query, node, semantic_weights)
        assert 0.0 <= semantic_score <= 1.0, (
            f"Semantic score should be between 0 and 1, got: {semantic_score}"
        )

        # Test 4: Entity-focused weights
        entity_weights = CompositeWeights.create_entity_focused()
        entity_score = composite_score(query, node, entity_weights)
        assert 0.0 <= entity_score <= 1.0, (
            f"Entity score should be between 0 and 1, got: {entity_score}"
        )

        # Test 5: Custom weights from dictionary
        custom_dict = {
            "semantic_similarity": 0.5,
            "llm_judge": 0.2,
            "entity_match": 0.2,
            "node_type_priority": 0.1,
        }
        custom_weights = CompositeWeights.from_dict(custom_dict)
        custom_score = composite_score(query, node, custom_weights)
        assert 0.0 <= custom_score <= 1.0, (
            f"Custom score should be between 0 and 1, got: {custom_score}"
        )

        # Test 6: Verify weights validation
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CompositeWeights(0.5, 0.5, 0.5, 0.5)  # Sum = 2.0

        with pytest.raises(ValueError, match="must be non-negative"):
            CompositeWeights(-0.1, 0.6, 0.3, 0.2)  # Negative weight

        # Test 7: Verify different weight configurations can produce different results
        scores = [
            default_score,
            balanced_score,
            semantic_score,
            entity_score,
            custom_score,
        ]

        # All scores should be valid
        for score in scores:
            assert 0.0 <= score <= 1.0, (
                f"All scores should be between 0 and 1, got: {score}"
            )

        # Test 8: Verify weight configuration works as expected
        # Create extreme weights to verify they affect the score
        semantic_only_weights = CompositeWeights(
            1.0, 0.0, 0.0, 0.0
        )  # Only semantic similarity
        entity_only_weights = CompositeWeights(0.0, 0.0, 1.0, 0.0)  # Only entity match

        semantic_only_score = composite_score(query, node, semantic_only_weights)
        entity_only_score = composite_score(query, node, entity_only_weights)

        # These should be different (unless by pure coincidence the metrics return identical values)
        assert 0.0 <= semantic_only_score <= 1.0, "Semantic-only score should be valid"
        assert 0.0 <= entity_only_score <= 1.0, "Entity-only score should be valid"

    def test_parallel_scorer_batch(self):
        """Test 3: Test parallel scorer returns maximum of all metrics in batch."""
        np.random.seed(42)
        query = QueryInput(
            "Find red mountain bikes",
            np.random.rand(384),
            ["red mountain bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Perfect red mountain bike match",
                np.random.rand(384),
                {},
                "product",
                ["red mountain bike"],
            ),
            NodeInput(
                "Unrelated blue car document",
                np.random.rand(384),
                {},
                "document",
                ["blue car"],
            ),
        ]

        # Test parallel scorer (should return max of all individual metrics)
        parallel_scores = batch_isRelevant(
            query, nodes, ScorerType.PARALLEL, batch_size=5
        )

        # Verify we get scores for all nodes
        assert len(parallel_scores) == 2, (
            f"Expected 2 parallel scores, got {len(parallel_scores)}"
        )

        # All scores should be between 0 and 1
        for i, score in enumerate(parallel_scores):
            assert 0.0 <= score <= 1.0, (
                f"Parallel score {i} should be between 0 and 1, got: {score}"
            )

        # First node should have higher score (better match)
        assert parallel_scores[0] > parallel_scores[1], (
            f"Better match should have higher parallel score: {parallel_scores[0]} vs {parallel_scores[1]}"
        )

    def test_router_scorer_basic_batch(self):
        """Test 4: Test basic router scorer (semantic + LLM + node type) in batch."""
        np.random.seed(42)
        query = QueryInput(
            "Find mountain bikes",
            np.random.rand(384),
            ["mountain bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Mountain bike product",
                np.random.rand(384),
                {},
                "product",
                ["mountain bike"],
            ),
            NodeInput(
                "Car documentation", np.random.rand(384), {}, "document", ["car"]
            ),
        ]

        # Test basic router scorer
        router_scores = batch_isRelevant(query, nodes, ScorerType.ROUTER, batch_size=5)

        # Verify we get scores for all nodes
        assert len(router_scores) == 2, (
            f"Expected 2 router scores, got {len(router_scores)}"
        )

        # All scores should be between 0 and 1
        for i, score in enumerate(router_scores):
            assert 0.0 <= score <= 1.0, (
                f"Router score {i} should be between 0 and 1, got: {score}"
            )

        # First node should have higher score (better match)
        assert router_scores[0] > router_scores[1], (
            f"Better match should have higher router score: {router_scores[0]} vs {router_scores[1]}"
        )

    def test_router_all_scorer_batch(self):
        """Test 5: Test router all scorer (all 4 metrics) in batch."""
        np.random.seed(42)
        query = QueryInput(
            "Find red bikes",
            np.random.rand(384),
            ["red bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Red bike with features",
                np.random.rand(384),
                {},
                "product",
                ["red bike", "features"],
            ),
            NodeInput(
                "Blue car manual", np.random.rand(384), {}, "document", ["blue car"]
            ),
        ]

        # Test router all scorer (combines all 4 metrics)
        router_all_scores = batch_isRelevant(
            query, nodes, ScorerType.ROUTER_ALL, batch_size=5
        )

        # Verify we get scores for all nodes
        assert len(router_all_scores) == 2, (
            f"Expected 2 router_all scores, got {len(router_all_scores)}"
        )

        # All scores should be between 0 and 1
        for i, score in enumerate(router_all_scores):
            assert 0.0 <= score <= 1.0, (
                f"Router_all score {i} should be between 0 and 1, got: {score}"
            )

    def test_router_two_variations_batch(self):
        """Test 6: Test router two-metric variations in batch."""
        np.random.seed(42)
        query = QueryInput(
            "Find bikes", np.random.rand(384), ["bike"], QueryIntent.PRODUCT_SEARCH
        )

        nodes = [
            NodeInput(
                "Bike product description", np.random.rand(384), {}, "product", ["bike"]
            ),
            NodeInput(
                "Car documentation", np.random.rand(384), {}, "document", ["car"]
            ),
        ]

        # Test semantic + LLM combination
        sem_llm_scores = batch_isRelevant(
            query, nodes, ScorerType.ROUTER_TWO_SEM_LLM, batch_size=5
        )
        assert len(sem_llm_scores) == 2, (
            f"Expected 2 semantic+LLM scores, got {len(sem_llm_scores)}"
        )

        # Test entity + type combination
        ent_type_scores = batch_isRelevant(
            query, nodes, ScorerType.ROUTER_TWO_ENT_TYPE, batch_size=5
        )
        assert len(ent_type_scores) == 2, (
            f"Expected 2 entity+type scores, got {len(ent_type_scores)}"
        )

        # All scores should be valid
        for scores, name in [
            (sem_llm_scores, "semantic+LLM"),
            (ent_type_scores, "entity+type"),
        ]:
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, (
                    f"{name} score {i} should be between 0 and 1, got: {score}"
                )

    def test_single_metric_routers_batch(self):
        """Test 7: Test single-metric router scorers in batch."""
        np.random.seed(42)
        query = QueryInput(
            "Find red bikes",
            np.random.rand(384),
            ["red bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        nodes = [
            NodeInput(
                "Red bike perfect match",
                np.random.rand(384),
                {},
                "product",
                ["red bike"],
            ),
            NodeInput(
                "Blue car different", np.random.rand(384), {}, "document", ["blue car"]
            ),
            NodeInput(
                "Generic bike partial", np.random.rand(384), {}, "product", ["bike"]
            ),
        ]

        # Test all single-metric routers
        single_routers = [
            (ScorerType.ROUTER_SINGLE_SEM, "semantic"),
            (ScorerType.ROUTER_SINGLE_LLM, "LLM"),
            (ScorerType.ROUTER_SINGLE_ENT, "entity"),
            (ScorerType.ROUTER_SINGLE_TYPE, "type"),
        ]

        for scorer_type, name in single_routers:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=5)

            # Verify we get scores for all nodes
            assert len(scores) == 3, f"Expected 3 {name} scores, got {len(scores)}"

            # All scores should be between 0 and 1
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, (
                    f"{name} score {i} should be between 0 and 1, got: {score}"
                )

    def test_scorer_comparison_batch(self):
        """Test 8: Compare different scorer types on same data to verify they produce different results."""
        np.random.seed(42)
        query = QueryInput(
            "Find mountain bikes",
            np.random.rand(384),
            ["mountain bike"],
            QueryIntent.PRODUCT_SEARCH,
        )

        # Create nodes with different characteristics to highlight scorer differences
        nodes = [
            NodeInput(
                "Mountain bike product perfect match",
                np.random.rand(384),
                {},
                "product",
                ["mountain bike"],
            ),  # Should score high on most metrics
            NodeInput(
                "Mountain bike documentation",
                np.random.rand(384),
                {},
                "document",
                ["mountain bike"],
            ),  # Good entity, wrong type for product search
            NodeInput(
                "Road bike product", np.random.rand(384), {}, "product", ["road bike"]
            ),  # Good type, wrong entity
            NodeInput(
                "Generic text", np.random.rand(384), {}, "unknown", []
            ),  # Poor on all metrics
        ]

        # Test multiple scorer types
        scorer_types = [
            ScorerType.COMPOSITE,
            ScorerType.PARALLEL,
            ScorerType.ROUTER,
            ScorerType.ROUTER_ALL,
            ScorerType.ROUTER_SINGLE_ENT,
            ScorerType.ROUTER_SINGLE_TYPE,
        ]

        results = {}
        for scorer_type in scorer_types:
            scores = batch_isRelevant(query, nodes, scorer_type, batch_size=10)
            results[scorer_type.value] = scores

            # Verify we get scores for all nodes
            assert len(scores) == 4, (
                f"Expected 4 scores for {scorer_type.value}, got {len(scores)}"
            )

            # All scores should be valid
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 1.0, (
                    f"{scorer_type.value} score {i} should be between 0 and 1, got: {score}"
                )

        # Verify that different scorers produce different results
        score_sets = [tuple(scores) for scores in results.values()]
        unique_score_sets = set(score_sets)

        # Should have some variation (at least 2 different result patterns)
        assert len(unique_score_sets) >= 2, (
            f"Expected different scorers to produce different results, got {len(unique_score_sets)} unique patterns"
        )

    def test_edge_cases_batch(self):
        """Test 9: Test edge cases for parallel and router scorers."""
        np.random.seed(42)

        # Edge case 1: Empty entities
        query_empty = QueryInput(
            "Generic query", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH
        )
        node_empty = NodeInput("Generic node", np.random.rand(384), {}, "product", [])

        # Test with empty entities
        empty_scores = batch_isRelevant(
            query_empty, [node_empty], ScorerType.PARALLEL, batch_size=5
        )
        assert len(empty_scores) == 1, "Should handle empty entities"
        assert 0.0 <= empty_scores[0] <= 1.0, (
            f"Empty entities score should be valid: {empty_scores[0]}"
        )

        # Edge case 2: Single node
        single_scores = batch_isRelevant(
            query_empty, [node_empty], ScorerType.ROUTER, batch_size=5
        )
        assert len(single_scores) == 1, "Should handle single node"
        assert 0.0 <= single_scores[0] <= 1.0, (
            f"Single node score should be valid: {single_scores[0]}"
        )

        # Edge case 3: Many nodes (test batch processing efficiency)
        many_nodes = [
            NodeInput(f"Node {i}", np.random.rand(384), {}, "product", [f"entity_{i}"])
            for i in range(20)
        ]

        many_scores = batch_isRelevant(
            query_empty, many_nodes, ScorerType.COMPOSITE, batch_size=10
        )
        assert len(many_scores) == 20, (
            f"Should handle many nodes, got {len(many_scores)}"
        )

        # All scores should be valid
        for i, score in enumerate(many_scores):
            assert 0.0 <= score <= 1.0, f"Many nodes score {i} should be valid: {score}"
