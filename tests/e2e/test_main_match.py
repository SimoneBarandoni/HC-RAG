import pytest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import the functions we need to test
from neo4j_rag_langgraph import (
    main,
)


@pytest.mark.e2e
class TestMainMatch:

    def test_main_matches_behavior(self):
        main()
