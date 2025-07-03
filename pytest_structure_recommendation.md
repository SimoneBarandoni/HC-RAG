# Recommended Pytest Structure for RAG System

## Current Issues
- Single file with 1054 lines
- Mixed test types (unit/integration/E2E)
- Shared state between tests
- Long test methods
- No pytest fixtures

## Recommended File Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/
│   ├── test_knowledge_graph.py    # Unit tests for knowledge graph module
│   ├── test_isrelevant.py         # Unit tests for scoring functions
│   └── test_llm_utils.py          # Unit tests for LLM utilities
├── integration/
│   ├── test_scorer_integration.py  # Integration tests for scorers
│   └── test_pipeline_components.py # Integration tests for pipeline
├── e2e/
│   ├── test_langgraph_workflow.py  # End-to-end workflow tests
│   └── test_complete_scenarios.py  # Complete scenario tests
└── fixtures/
    ├── test_data.py               # Test data factories
    └── mock_responses.py          # Mock response data
```

## Benefits of This Structure

### 1. **Separation of Concerns**
- **Unit tests**: Fast, isolated, test individual functions
- **Integration tests**: Test component interactions
- **E2E tests**: Test complete workflows

### 2. **Maintainability**
- Smaller, focused files (200-300 lines each)
- Easier to navigate and understand
- Clear responsibility boundaries

### 3. **Test Performance**
- Run only relevant test suites
- Faster feedback during development
- Parallel test execution

### 4. **Reusability**
- Shared fixtures in conftest.py
- Common test data in fixtures/
- Reduced code duplication

## Implementation Example

### tests/conftest.py
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from isRelevant import QueryInput, NodeInput, QueryIntent, CompositeWeights

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
```

### tests/unit/test_isrelevant.py
```python
import pytest
import numpy as np
from isRelevant import (
    batch_semantic_similarity, 
    batch_entity_match, 
    composite_score,
    ScorerType
)

class TestSemanticSimilarity:
    """Unit tests for semantic similarity functions."""
    
    def test_identical_embeddings(self, sample_query, sample_nodes):
        """Test semantic similarity with identical embeddings."""
        # Arrange
        identical_embeddings = np.ones(384)
        sample_query.embeddings = identical_embeddings
        sample_nodes[0].embeddings = identical_embeddings
        
        # Act
        similarities = batch_semantic_similarity(sample_query, [sample_nodes[0]])
        
        # Assert
        assert abs(similarities[0] - 1.0) < 1e-10

    def test_orthogonal_embeddings(self, sample_query, sample_nodes):
        """Test semantic similarity with orthogonal embeddings."""
        # Arrange
        embedding_a = np.zeros(384)
        embedding_a[0] = 1.0
        embedding_b = np.zeros(384)
        embedding_b[1] = 1.0
        
        sample_query.embeddings = embedding_a
        sample_nodes[0].embeddings = embedding_b
        
        # Act
        similarities = batch_semantic_similarity(sample_query, [sample_nodes[0]])
        
        # Assert
        assert abs(similarities[0] - 0.5) < 1e-10

class TestEntityMatching:
    """Unit tests for entity matching functions."""
    
    def test_perfect_match(self, sample_query, sample_nodes):
        """Test entity matching with perfect match."""
        # Arrange
        sample_query.entities = ["red mountain bike", "trail"]
        sample_nodes[0].entities = ["red mountain bike", "trail"]
        
        # Act
        matches = batch_entity_match(sample_query, [sample_nodes[0]])
        
        # Assert
        assert matches[0] == 1.0

    def test_partial_match(self, sample_query, sample_nodes):
        """Test entity matching with partial match."""
        # Arrange
        sample_query.entities = ["red mountain bike", "trail"]
        sample_nodes[0].entities = ["red mountain bike"]
        
        # Act
        matches = batch_entity_match(sample_query, [sample_nodes[0]])
        
        # Assert
        assert abs(matches[0] - 0.5) < 0.001
```

### tests/integration/test_scorer_integration.py
```python
import pytest
from isRelevant import batch_isRelevant, ScorerType

class TestScorerIntegration:
    """Integration tests for different scorer types."""
    
    @pytest.mark.parametrize("scorer_type", [
        ScorerType.COMPOSITE,
        ScorerType.PARALLEL,
        ScorerType.ROUTER,
        ScorerType.ROUTER_SINGLE_SEM,
    ])
    def test_scorer_types(self, sample_query, sample_nodes, scorer_type):
        """Test different scorer types produce valid results."""
        # Act
        scores = batch_isRelevant(sample_query, sample_nodes, scorer_type)
        
        # Assert
        assert len(scores) == len(sample_nodes)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_scorer_comparison(self, sample_query, sample_nodes):
        """Test that different scorers produce different results."""
        # Arrange
        scorer_types = [ScorerType.COMPOSITE, ScorerType.PARALLEL]
        
        # Act
        results = {}
        for scorer_type in scorer_types:
            results[scorer_type] = batch_isRelevant(sample_query, sample_nodes, scorer_type)
        
        # Assert
        assert len(set(tuple(scores) for scores in results.values())) >= 1
```

### tests/e2e/test_langgraph_workflow.py
```python
import pytest
from neo4j_rag_langgraph import app

class TestLangGraphWorkflow:
    """End-to-end tests for complete LangGraph workflows."""
    
    @pytest.fixture
    def initial_state(self):
        """Provide initial state for workflow tests."""
        return {
            "question": "What mountain bikes do you have?",
            "revision_history": [],
            "sampled_nodes": [],
            # ... other initial state
        }

    @pytest.mark.slow
    def test_complete_workflow(self, initial_state):
        """Test complete LangGraph workflow execution."""
        # Act
        final_state = app.invoke(initial_state, {"recursion_limit": 10})
        
        # Assert
        assert "final_answer" in final_state
        assert isinstance(final_state["final_answer"], str)
        assert len(final_state["final_answer"]) > 50

    @pytest.mark.slow
    def test_workflow_with_revision(self, initial_state):
        """Test workflow that requires question revision."""
        # Arrange
        initial_state["question"] = "Tell me about options for the trail"
        
        # Act
        final_state = app.invoke(initial_state, {"recursion_limit": 10})
        
        # Assert
        assert "final_answer" in final_state
        # Check if revision occurred
        revision_history = final_state.get("revision_history", [])
        # May or may not have revisions depending on data quality
```

## Running Tests

### pytest.ini configuration
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    unit: marks tests as unit tests
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    requires_neo4j: marks tests that require Neo4j connection
    requires_llm: marks tests that require LLM service
```

### Running specific test types
```bash
# Run only unit tests (fast)
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run E2E tests (slow)
pytest tests/e2e/ -v -m slow

# Run all tests except slow ones
pytest -m "not slow"

# Run tests that don't require external services
pytest -m "not requires_neo4j and not requires_llm"
```

## Migration Strategy

### Phase 1: Create structure
1. Create new directory structure
2. Move conftest.py with common fixtures
3. Create test data factories

### Phase 2: Split unit tests
1. Move Milestone 1 tests to unit/ directory
2. Split by module being tested
3. Add proper fixtures

### Phase 3: Split integration tests
1. Move Milestone 2 tests to integration/ directory
2. Focus on component interactions
3. Add integration fixtures

### Phase 4: Split E2E tests
1. Move Milestone 3 tests to e2e/ directory
2. Add slow marker
3. Remove shared state, use fixtures instead

### Phase 5: Cleanup
1. Remove original large file
2. Update CI/CD to use new structure
3. Add performance benchmarks

## Additional Recommendations

### 1. **Use Pytest Markers**
```python
@pytest.mark.slow
@pytest.mark.requires_neo4j
@pytest.mark.requires_llm
def test_expensive_operation():
    pass
```

### 2. **Parametrized Tests**
```python
@pytest.mark.parametrize("scorer_type,expected_range", [
    (ScorerType.COMPOSITE, (0.0, 1.0)),
    (ScorerType.PARALLEL, (0.0, 1.0)),
])
def test_scorer_range(scorer_type, expected_range):
    pass
```

### 3. **Test Data Factories**
```python
# tests/fixtures/test_data.py
class QueryInputFactory:
    @staticmethod
    def create(text="default query", **kwargs):
        defaults = {
            "embeddings": np.random.rand(384),
            "entities": [],
            "intent": QueryIntent.PRODUCT_SEARCH
        }
        defaults.update(kwargs)
        return QueryInput(text, **defaults)
```

### 4. **Performance Tests**
```python
@pytest.mark.benchmark
def test_batch_processing_performance(benchmark):
    result = benchmark(batch_isRelevant, query, nodes, ScorerType.COMPOSITE)
    assert len(result) == len(nodes)
```

This structure will make your tests more maintainable, faster to run, and easier to understand while following pytest best practices. 