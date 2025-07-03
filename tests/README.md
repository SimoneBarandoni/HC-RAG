# Test Structure

This directory contains the restructured pytest test suite following best practices.

## Directory Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (fast, isolated)
│   └── test_milestone1_core_components.py
├── integration/                # Integration tests (external services)
│   └── test_milestone2_isrelevant_integration.py
├── e2e/                       # End-to-end tests (full pipeline)
│   └── test_milestone3_langgraph_e2e.py
└── fixtures/                  # Test data and fixtures
```

## Running Tests

### All tests
```bash
pytest
```

### By milestone/category
```bash
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only  
pytest tests/e2e/          # E2E tests only
```

### By markers
```bash
pytest -m unit              # Unit tests
pytest -m integration       # Integration tests
pytest -m e2e               # E2E tests
pytest -m "not slow"        # Skip slow tests
```

### Specific test files
```bash
pytest tests/unit/test_milestone1_core_components.py
pytest tests/integration/test_milestone2_isrelevant_integration.py
pytest tests/e2e/test_milestone3_langgraph_e2e.py
```

## Test Categories

- **Unit tests**: Fast, isolated tests for individual functions and components
- **Integration tests**: Tests that verify component interactions and scorer integrations
- **E2E tests**: Full pipeline tests including LangGraph workflow scenarios

## Shared Fixtures

Common fixtures are defined in `conftest.py` and available to all tests:
- `sample_query`: Sample QueryInput for testing
- `sample_nodes`: Sample NodeInput list for testing
- `mock_neo4j_connection`: Mocked Neo4j connection
- `mock_llm_client`: Mocked LLM client 