import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List

from isRelevant import (
    QueryInput, NodeInput, QueryIntent, ScorerType, RelevanceScore,
    semantic_similarity, llm_judge, entity_match, node_type_priority,
    composite_score, parallel_score, router_score, isRelevant,
    priority_matrix
)


class TestQueryInput:
    """Test QueryInput dataclass"""
    
    def test_query_input_creation(self):
        embeddings = np.random.rand(384)
        query = QueryInput(
            text="Find red mountain bikes",
            embeddings=embeddings,
            entities=["red mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        assert query.text == "Find red mountain bikes"
        assert len(query.embeddings) == 384
        assert query.entities == ["red mountain bike"]
        assert query.intent == QueryIntent.PRODUCT_SEARCH


class TestNodeInput:
    """Test NodeInput dataclass"""
    
    def test_node_input_creation(self):
        embeddings = np.random.rand(384)
        node = NodeInput(
            text="Red mountain bike description",
            embeddings=embeddings,
            graph_relations={"type": "product", "id": "123"},
            node_type="product",
            entities=["red mountain bike"]
        )
        
        assert node.text == "Red mountain bike description"
        assert len(node.embeddings) == 384
        assert node.graph_relations == {"type": "product", "id": "123"}
        assert node.node_type == "product"
        assert node.entities == ["red mountain bike"]


class TestSemanticSimilarity:
    """Test semantic similarity function"""
    
    def test_semantic_similarity_identical_embeddings(self):
        embeddings = np.ones(384)
        query = QueryInput("test", embeddings, [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", embeddings, {}, "product", [])
        
        similarity = semantic_similarity(query, node)
        assert similarity == 1.0
    
    def test_semantic_similarity_opposite_embeddings(self):
        query_embeddings = np.ones(384)
        node_embeddings = -np.ones(384)
        
        query = QueryInput("test", query_embeddings, [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", node_embeddings, {}, "product", [])
        
        similarity = semantic_similarity(query, node)
        assert similarity == 0.0
    
    def test_semantic_similarity_random_embeddings(self):
        query_embeddings = np.random.rand(384)
        node_embeddings = np.random.rand(384)
        
        query = QueryInput("test", query_embeddings, [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", node_embeddings, {}, "product", [])
        
        similarity = semantic_similarity(query, node)
        assert 0.0 <= similarity <= 1.0


class TestEntityMatch:
    """Test entity matching function"""
    
    def test_entity_match_perfect_match(self):
        entities = ["red mountain bike", "trail"]
        query = QueryInput("test", np.random.rand(384), entities, QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", entities)
        
        match_score = entity_match(query, node)
        assert match_score == 1.0
    
    def test_entity_match_partial_match(self):
        query_entities = ["red mountain bike", "trail", "suspension"]
        node_entities = ["red mountain bike", "trail"]
        
        query = QueryInput("test", np.random.rand(384), query_entities, QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", node_entities)
        
        match_score = entity_match(query, node)
        assert match_score == 2/3  # 2 out of 3 entities match
    
    def test_entity_match_no_match(self):
        query_entities = ["red mountain bike"]
        node_entities = ["blue road bike"]
        
        query = QueryInput("test", np.random.rand(384), query_entities, QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", node_entities)
        
        match_score = entity_match(query, node)
        assert match_score == 0.0
    
    def test_entity_match_empty_query_entities(self):
        query_entities = []
        node_entities = ["red mountain bike"]
        
        query = QueryInput("test", np.random.rand(384), query_entities, QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", node_entities)
        
        # This should handle division by zero
        with pytest.raises(ZeroDivisionError):
            entity_match(query, node)


class TestNodeTypePriority:
    """Test node type priority function"""
    
    def test_node_type_priority_perfect_match(self):
        query = QueryInput("test", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", [])
        
        priority = node_type_priority(query, node)
        assert priority == 1.0
    
    def test_node_type_priority_document_for_technical_support(self):
        query = QueryInput("test", np.random.rand(384), [], QueryIntent.TECHNICAL_SUPPORT)
        node = NodeInput("test", np.random.rand(384), {}, "document", [])
        
        priority = node_type_priority(query, node)
        assert priority == 1.0
    
    def test_node_type_priority_unknown_node_type(self):
        query = QueryInput("test", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "unknown", [])
        
        priority = node_type_priority(query, node)
        assert priority == 0.1
    
    def test_all_query_intents_have_priority_matrix(self):
        for intent in QueryIntent:
            assert intent in priority_matrix
            
        for intent_priorities in priority_matrix.values():
            assert 'product' in intent_priorities
            assert 'document' in intent_priorities
            assert 'specification' in intent_priorities
            assert 'category' in intent_priorities
            assert 'unknown' in intent_priorities


class TestLLMJudge:
    """Test LLM judge function with mocking"""
    
    @patch('isRelevant.OpenAI')
    def test_llm_judge_high_relevance(self, mock_openai):
        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[0].message.parsed.score = 0.95
        
        mock_client = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = QueryInput("Find red mountain bikes", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("Premium Red Mountain Bike under $900", np.random.rand(384), {}, "product", [])
        
        score = llm_judge(query, node)
        assert score == 0.95
        
        # Verify the OpenAI client was called correctly
        mock_openai.assert_called_once_with(
            base_url="http://localhost:11434/v1",
            api_key="gemma3:1b",
        )
    
    @patch('isRelevant.OpenAI')
    def test_llm_judge_low_relevance(self, mock_openai):
        # Mock low relevance score
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[0].message.parsed.score = 0.1
        
        mock_client = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = QueryInput("Find red mountain bikes", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("Kitchen blender for smoothies", np.random.rand(384), {}, "product", [])
        
        score = llm_judge(query, node)
        assert score == 0.1


class TestScoringStrategies:
    """Test different scoring strategies"""
    
    def setup_method(self):
        """Setup common test data"""
        self.query = QueryInput(
            text="Find red mountain bikes under $1000",
            embeddings=np.ones(384),  # Use consistent embeddings for testing
            entities=["red mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        self.relevant_node = NodeInput(
            text="Premium Red Mountain Bike under $900",
            embeddings=np.ones(384),  # Same embeddings for high similarity
            graph_relations={"type": "product"},
            node_type="product",
            entities=["red mountain bike"]
        )
        
        self.irrelevant_node = NodeInput(
            text="Kitchen blender for smoothies",
            embeddings=-np.ones(384),  # Opposite embeddings for low similarity
            graph_relations={"type": "product"},
            node_type="product",
            entities=["kitchen blender"]
        )
    
    @patch('isRelevant.llm_judge')
    def test_composite_score_relevant_node(self, mock_llm):
        mock_llm.return_value = 0.95
        
        score = composite_score(self.query, self.relevant_node)
        
        # Composite score: semantic(1.0)*0.4 + llm(0.95)*0.3 + entity(1.0)*0.15 + priority(1.0)*0.15
        expected = 1.0 * 0.4 + 0.95 * 0.3 + 1.0 * 0.15 + 1.0 * 0.15
        assert abs(score - expected) < 0.01
    
    @patch('isRelevant.llm_judge')
    def test_composite_score_irrelevant_node(self, mock_llm):
        mock_llm.return_value = 0.1
        
        score = composite_score(self.query, self.irrelevant_node)
        
        # Should be much lower due to poor semantic similarity and entity match
        assert score < 0.5
    
    @patch('isRelevant.llm_judge')
    def test_parallel_score(self, mock_llm):
        mock_llm.return_value = 0.8
        
        score = parallel_score(self.query, self.relevant_node)
        
        # Parallel score takes the maximum of all metrics
        assert score == 1.0  # Max of semantic_similarity(1.0), llm(0.8), entity(1.0), priority(1.0)
    
    @patch('isRelevant.llm_judge')
    def test_router_score(self, mock_llm):
        mock_llm.return_value = 0.8
        
        metrics = [semantic_similarity, mock_llm, node_type_priority]
        score = router_score(self.query, self.relevant_node, metrics)
        
        # Router score averages the specified metrics
        expected = (1.0 + 0.8 + 1.0) / 3  # semantic + llm + priority / 3
        assert abs(score - expected) < 0.01


class TestIsRelevantMainFunction:
    """Test the main isRelevant function"""
    
    def setup_method(self):
        """Setup test data"""
        self.query = QueryInput(
            text="Find red mountain bikes",
            embeddings=np.random.rand(384),
            entities=["red mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        self.node = NodeInput(
            text="Red mountain bike",
            embeddings=np.random.rand(384),
            graph_relations={},
            node_type="product",
            entities=["red mountain bike"]
        )
    
    @patch('isRelevant.composite_score')
    def test_isrelevant_composite_scorer(self, mock_composite):
        mock_composite.return_value = 0.85
        
        score = isRelevant(self.query, self.node, ScorerType.COMPOSITE)
        
        assert score == 0.85
        mock_composite.assert_called_once_with(self.query, self.node)
    
    @patch('isRelevant.parallel_score')
    def test_isrelevant_parallel_scorer(self, mock_parallel):
        mock_parallel.return_value = 0.90
        
        score = isRelevant(self.query, self.node, ScorerType.PARALLEL)
        
        assert score == 0.90
        mock_parallel.assert_called_once_with(self.query, self.node)
    
    @patch('isRelevant.router_score')
    def test_isrelevant_router_scorer(self, mock_router):
        mock_router.return_value = 0.75
        
        score = isRelevant(self.query, self.node, ScorerType.ROUTER)
        
        assert score == 0.75
        mock_router.assert_called_once()


class TestEndToEndScenarios:
    """End-to-end test scenarios with real data"""
    
    @patch('isRelevant.OpenAI')
    def test_perfect_match_scenario(self, mock_openai):
        """Test scenario where query and node are perfect matches"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[0].message.parsed.score = 0.95
        
        mock_client = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = QueryInput(
            text="Find red mountain bikes under $1000",
            embeddings=np.ones(384),
            entities=["red mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        node = NodeInput(
            text="Premium Red Mountain Bike - Trail Blazer X1 under $900",
            embeddings=np.ones(384),  # Same embeddings for perfect semantic match
            graph_relations={"type": "product", "category": "mountain_bikes"},
            node_type="product",
            entities=["red mountain bike"]
        )
        
        # Test all scorer types
        composite_score_result = isRelevant(query, node, ScorerType.COMPOSITE)
        parallel_score_result = isRelevant(query, node, ScorerType.PARALLEL)
        router_score_result = isRelevant(query, node, ScorerType.ROUTER)
        
        # All scores should be high for perfect match
        assert composite_score_result > 0.8
        assert parallel_score_result > 0.8
        assert router_score_result > 0.8
    
    @patch('isRelevant.OpenAI')
    def test_irrelevant_match_scenario(self, mock_openai):
        """Test scenario where query and node are completely irrelevant"""
        # Mock LLM response for irrelevant content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[0].message.parsed.score = 0.05
        
        mock_client = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        query = QueryInput(
            text="Find red mountain bikes under $1000",
            embeddings=np.ones(384),
            entities=["red mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        node = NodeInput(
            text="Stainless Steel Kitchen Blender for smoothies",
            embeddings=-np.ones(384),  # Opposite embeddings for poor semantic match
            graph_relations={"type": "product", "category": "kitchen_appliances"},
            node_type="product",
            entities=["kitchen blender"]
        )
        
        # Test all scorer types
        composite_score_result = isRelevant(query, node, ScorerType.COMPOSITE)
        parallel_score_result = isRelevant(query, node, ScorerType.PARALLEL)
        router_score_result = isRelevant(query, node, ScorerType.ROUTER)
        
        # All scores should be low for irrelevant match
        assert composite_score_result < 0.5
        assert parallel_score_result < 0.5
        assert router_score_result < 0.5
    
    @patch('isRelevant.OpenAI')
    def test_different_query_intents(self, mock_openai):
        """Test how different query intents affect scoring"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.parsed = Mock()
        mock_response.choices[0].message.parsed.score = 0.7
        
        mock_client = Mock()
        mock_client.beta.chat.completions.parse.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Document node
        document_node = NodeInput(
            text="Mountain Bike Maintenance Guide",
            embeddings=np.random.rand(384),
            graph_relations={"type": "document"},
            node_type="document",
            entities=["mountain bike"]
        )
        
        # Test with TECHNICAL_SUPPORT intent (should favor documents)
        technical_query = QueryInput(
            text="How to maintain mountain bikes",
            embeddings=np.random.rand(384),
            entities=["mountain bike"],
            intent=QueryIntent.TECHNICAL_SUPPORT
        )
        
        # Test with PRODUCT_SEARCH intent (should not favor documents as much)
        product_query = QueryInput(
            text="Find mountain bikes",
            embeddings=np.random.rand(384),
            entities=["mountain bike"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        technical_score = isRelevant(technical_query, document_node, ScorerType.COMPOSITE)
        product_score = isRelevant(product_query, document_node, ScorerType.COMPOSITE)
        
        # Technical support should score higher for document nodes
        assert technical_score > product_score
    
    def test_score_bounds(self):
        """Test that all scores are within valid bounds [0, 1]"""
        query = QueryInput(
            text="test query",
            embeddings=np.random.rand(384),
            entities=["test"],
            intent=QueryIntent.PRODUCT_SEARCH
        )
        
        node = NodeInput(
            text="test node",
            embeddings=np.random.rand(384),
            graph_relations={},
            node_type="product",
            entities=["test"]
        )
        
        # Test semantic similarity bounds
        semantic_score = semantic_similarity(query, node)
        assert 0.0 <= semantic_score <= 1.0
        
        # Test entity match bounds
        entity_score = entity_match(query, node)
        assert 0.0 <= entity_score <= 1.0
        
        # Test node type priority bounds
        priority_score = node_type_priority(query, node)
        assert 0.0 <= priority_score <= 1.0


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_embeddings(self):
        """Test behavior with empty embeddings"""
        query = QueryInput("test", np.array([]), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.array([]), {}, "product", [])
        
        with pytest.raises((ValueError, IndexError)):
            semantic_similarity(query, node)
    
    def test_mismatched_embedding_dimensions(self):
        """Test behavior with mismatched embedding dimensions"""
        query = QueryInput("test", np.random.rand(256), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", [])
        
        with pytest.raises(ValueError):
            semantic_similarity(query, node)
    
    def test_empty_entities_query(self):
        """Test behavior when query has no entities"""
        query = QueryInput("test", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "product", ["entity"])
        
        with pytest.raises(ZeroDivisionError):
            entity_match(query, node)
    
    def test_unknown_node_type(self):
        """Test behavior with unknown node type"""
        query = QueryInput("test", np.random.rand(384), [], QueryIntent.PRODUCT_SEARCH)
        node = NodeInput("test", np.random.rand(384), {}, "unknown_type", [])
        
        # Should default to 'unknown' priority
        priority = node_type_priority(query, node)
        assert priority == 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 