import pytest
from unittest.mock import patch
from time import time
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import the functions we need to test
from neo4j_rag_langgraph import (
    app,
    analyze_query,
    evaluate_context,
    generate_answer,
    revise_question,
    sample_neo4j_nodes,
    score_semantic_similarity,
    expand_subgraph,
    score_expanded_nodes_with_isrelevant,
)
from isRelevant import (
    QueryInput,
    QueryIntent,
)


@pytest.mark.e2e
class TestMilestone3LangGraphEndToEnd:
    # Shared state for all test scenarios
    pipeline_scenarios = {}

    def test_scenario_1_standard_product_search_e2e(self):
        """Test 1: E2E Test with Correct Answer - Standard product search workflow

        This test covers:
        - Component integration (analyze_query → sample_nodes → score_similarity → expand → final_score)
        - End-to-end workflow with clear query producing relevant answer
        """

        # Clear query that should produce good results
        test_question = "What mountain bikes do you have?"

        # Initialize scenario state
        scenario_name = "standard_product_search"
        TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = {
            "question": test_question
        }
        state = TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name]

        try:
            # Step 1: Analyze Query (Component + E2E validation)
            result = analyze_query(state)
            state.update(result)

            # Validate analyze_query component behavior
            assert "query_input" in result, "Should produce query_input"
            query_input = result["query_input"]
            assert hasattr(query_input, "text"), "QueryInput should have text"
            assert hasattr(query_input, "embeddings"), (
                "QueryInput should have embeddings"
            )
            assert hasattr(query_input, "entities"), "QueryInput should have entities"
            assert hasattr(query_input, "intent"), "QueryInput should have intent"
            assert query_input.text == test_question, "Text should match input"
            assert isinstance(query_input.intent, QueryIntent), (
                "Intent should be QueryIntent enum"
            )

            # Step 2: Sample Neo4j Nodes (Component + E2E validation)
            result = sample_neo4j_nodes(state)
            state.update(result)

            # Validate sample_neo4j_nodes component behavior
            assert "sampled_nodes" in result, "Should produce sampled_nodes"
            sampled_nodes = result["sampled_nodes"]
            assert isinstance(sampled_nodes, list), "Sampled nodes should be list"
            assert len(sampled_nodes) <= 20, "Should sample max 20 nodes"

            # Step 3: Score Semantic Similarity (Component + E2E validation)
            result = score_semantic_similarity(state)
            state.update(result)

            # Validate score_semantic_similarity component behavior
            assert "semantic_scored_nodes" in result, (
                "Should produce semantic_scored_nodes"
            )
            semantic_scored_nodes = result["semantic_scored_nodes"]
            assert isinstance(semantic_scored_nodes, list), (
                "Semantic scored nodes should be list"
            )

            # Validate NodeInput structure and scoring
            for i, node in enumerate(semantic_scored_nodes):
                assert hasattr(node, "text"), f"Node {i} should have text"
                assert hasattr(node, "score"), f"Node {i} should have score"
                assert 0.0 <= node.score <= 1.0, (
                    f"Node {i} score should be valid: {node.score}"
                )
                assert node.score >= 0.60, (
                    f"Node {i} should meet threshold: {node.score}"
                )

            # Validate sorting
            if len(semantic_scored_nodes) > 1:
                for i in range(len(semantic_scored_nodes) - 1):
                    assert (
                        semantic_scored_nodes[i].score
                        >= semantic_scored_nodes[i + 1].score
                    ), "Semantic nodes should be sorted by score"

            # Step 4: Expand Subgraph (Component + E2E validation)
            result = expand_subgraph(state)
            state.update(result)

            # Validate expand_subgraph component behavior
            assert "expanded_subgraph" in result, "Should produce expanded_subgraph"
            assert "expanded_nodes" in result, "Should produce expanded_nodes"
            expanded_subgraph = result["expanded_subgraph"]
            expanded_nodes = result["expanded_nodes"]
            assert isinstance(expanded_subgraph, list), (
                "Expanded subgraph should be list"
            )
            assert isinstance(expanded_nodes, list), "Expanded nodes should be list"

            # Step 5: Score All Nodes with isRelevant (Component + E2E validation)
            result = score_expanded_nodes_with_isrelevant(state)
            state.update(result)

            # Validate score_expanded_nodes_with_isrelevant component behavior
            assert "expanded_scored_nodes" in result, (
                "Should produce expanded_scored_nodes"
            )
            assert "final_relevant_nodes" in result, (
                "Should produce final_relevant_nodes"
            )
            # expanded_scored_nodes = result["expanded_scored_nodes"]
            final_relevant_nodes = result["final_relevant_nodes"]

            assert isinstance(final_relevant_nodes, list), "Final nodes should be list"
            assert len(final_relevant_nodes) <= 15, "Should limit to 15 final nodes"

            # Validate final node structure and scoring
            for i, node in enumerate(final_relevant_nodes):
                assert hasattr(node, "score"), f"Final node {i} should have score"
                assert hasattr(node, "text"), f"Final node {i} should have text"
                assert hasattr(node, "node_type"), (
                    f"Final node {i} should have node_type"
                )
                assert 0.0 <= node.score <= 1.0, (
                    f"Final node {i} score should be valid: {node.score}"
                )

            # Validate final sorting
            if len(final_relevant_nodes) > 1:
                for i in range(len(final_relevant_nodes) - 1):
                    assert (
                        final_relevant_nodes[i].score
                        >= final_relevant_nodes[i + 1].score
                    ), "Final nodes should be sorted by score"

            # Step 6: End-to-End Answer Generation
            result = generate_answer(state)
            state.update(result)

            # Validate end-to-end answer generation
            assert "final_answer" in result, "Should produce final_answer"
            final_answer = result["final_answer"]
            assert isinstance(final_answer, str), "Final answer should be string"
            assert len(final_answer.strip()) > 0, "Final answer should not be empty"
            assert len(final_answer) > 50, (
                "Final answer should be substantive"
            )  # At least a sentence

            print(f"    Generated answer ({len(final_answer)} chars)")
            print(f"    Answer preview: {final_answer}")

        except Exception as e:
            if "Neo4j" in str(e) or "database" in str(e).lower():
                pytest.skip(f"Database unavailable for E2E test: {e}")
            else:
                raise AssertionError(f"Standard product search E2E test failed: {e}")

    def test_scenario_2_insufficient_context_and_revision_e2e(self):
        """Test 2: E2E Test with Insufficient Context and Revision

        Tests the revision loop when context is insufficient.
        Given an ambiguous query, system should:
        1. Initially fail to find relevant nodes
        2. Enter revision loop via evaluate_context
        3. Second iteration should produce valid answer
        """

        # Ambiguous query that should trigger revision
        test_question = "Tell me about options for the trail"

        scenario_name = "insufficient_context_revision"
        TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = {
            "question": test_question,
            "revision_history": [],
        }
        state = TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name]

        try:
            # First iteration - should be insufficient

            # Run pipeline first iteration
            state.update(analyze_query(state))
            state.update(sample_neo4j_nodes(state))
            state.update(score_semantic_similarity(state))
            state.update(expand_subgraph(state))
            state.update(score_expanded_nodes_with_isrelevant(state))

            # Evaluate context - should be insufficient
            result = evaluate_context(state)
            state.update(result)

            # If sufficient on first try, that's actually fine (good data)
            if state.get("decision") == "sufficient":
                result = generate_answer(state)
                state.update(result)
                assert "final_answer" in result, "Should generate answer"
                print(result["final_answer"])
                return

            # If insufficient, test revision
            assert state.get("decision") == "revision", (
                "Should decide on revision for ambiguous query"
            )

            # Revision step
            result = revise_question(state)
            state.update(result)

            assert "question" in result, "Revision should produce new question"
            assert "revision_history" in result, "Should track revision history"
            revised_question = result["question"]
            revision_history = result["revision_history"]

            assert revised_question != test_question, (
                "Revised question should be different"
            )
            assert len(revision_history) > 0, "Should have revision history"
            assert test_question in revision_history, "Should track original question"

            print(f"    Original: '{test_question}'")
            print(f"    Revised:  '{revised_question}'")

            # Run pipeline second iteration
            state.update(analyze_query(state))
            state.update(sample_neo4j_nodes(state))
            state.update(score_semantic_similarity(state))
            state.update(expand_subgraph(state))
            state.update(score_expanded_nodes_with_isrelevant(state))

            # Second evaluation
            result = evaluate_context(state)
            state.update(result)

            # Generate final answer
            result = generate_answer(state)
            state.update(result)

            assert "final_answer" in result, "Should generate answer after revision"
            final_answer = result["final_answer"]
            assert isinstance(final_answer, str), "Answer should be string"
            assert len(final_answer.strip()) > 0, "Answer should not be empty"

            print("    Generated answer after revision:")
            print(final_answer)

        except Exception as e:
            if "Neo4j" in str(e) or "database" in str(e).lower():
                pytest.skip(f"Database unavailable for revision E2E test: {e}")
            else:
                raise AssertionError(
                    f"Insufficient context and revision E2E test failed: {e}"
                )

    def test_scenario_3_llm_failure_handling_e2e(self):
        """Test 3: LLM Failure Test

        Tests system resilience when LLM fails.
        System should:
        1. Not crash when LLM fails
        2. Handle error gracefully
        3. Return fallback response
        """

        test_question = "Find mountain bikes under $500"

        scenario_name = "llm_failure"
        TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = {
            "question": test_question
        }
        state = TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name]

        # Test LLM failure in different components

        with patch("neo4j_rag_langgraph.call_ollama_llm") as mock_llm:
            # Make LLM fail for intent analysis
            mock_llm.side_effect = Exception("LLM service unavailable")

            try:
                result = analyze_query(state)
                # Should not crash, should have fallback
                assert "query_input" in result, "Should have fallback query_input"
                query_input = result["query_input"]
                assert isinstance(query_input.intent, QueryIntent), (
                    "Should have fallback intent"
                )
            except Exception as e:
                # If it crashes, that's actually a problem
                raise AssertionError(
                    f"Query analysis should handle LLM failure gracefully: {e}"
                )

        # First run normal pipeline to get to evaluation step
        state.update(analyze_query(state))
        state.update(sample_neo4j_nodes(state))
        state.update(score_semantic_similarity(state))
        state.update(expand_subgraph(state))
        state.update(score_expanded_nodes_with_isrelevant(state))

        with patch("neo4j_rag_langgraph.call_ollama_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM service timeout")

            try:
                result = evaluate_context(state)
                # Should not crash, should have fallback decision
                assert "decision" in result, "Should have fallback decision"
                decision = result["decision"]
                assert decision in ["sufficient", "revision"], (
                    f"Should have valid fallback decision: {decision}"
                )
            except Exception as e:
                raise AssertionError(
                    f"Context evaluation should handle LLM failure gracefully: {e}"
                )

        # Test 3: LLM failure in answer generation
        state["decision"] = "sufficient"  # Force sufficient to reach answer generation

        with patch("neo4j_rag_langgraph.call_ollama_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM connection lost")

            try:
                result = generate_answer(state)
                # Should not crash, should have fallback answer
                assert "final_answer" in result, "Should have fallback answer"
                final_answer = result["final_answer"]
                assert isinstance(final_answer, str), "Fallback answer should be string"
                assert len(final_answer) > 0, "Fallback answer should not be empty"
                print(f"    Fallback answer: '{final_answer}...'")
            except Exception as e:
                raise AssertionError(
                    f"Answer generation should handle LLM failure gracefully: {e}"
                )

    def test_scenario_4_empty_data_and_no_results_e2e(self):
        """Test 4: Test with Empty or Unexpected Data

        Tests system behavior when no relevant nodes can be found.
        Should gracefully handle and generate appropriate response.
        """

        # Query that's very unlikely to match anything
        test_question = "Find purple flying unicorn bicycles with laser beams"

        scenario_name = "empty_data"
        TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = {
            "question": test_question,
            "revision_history": [],
        }
        state = TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name]

        try:
            # Run complete pipeline
            state.update(analyze_query(state))
            state.update(sample_neo4j_nodes(state))
            state.update(score_semantic_similarity(state))
            state.update(expand_subgraph(state))
            state.update(score_expanded_nodes_with_isrelevant(state))

            # Check if we got any results
            final_relevant_nodes = state.get("final_relevant_nodes", [])
            print(f"    Found {len(final_relevant_nodes)} relevant nodes")

            # If we found results, that's fine (good test data), but test with empty results
            if len(final_relevant_nodes) > 0:
                # Artificially create empty scenario
                state["final_relevant_nodes"] = []

            # Test context evaluation with no results
            result = evaluate_context(state)
            state.update(result)

            decision = state.get("decision", "unknown")
            print(f"    Decision with no results: {decision}")

            # Test multiple revisions without improvement
            revision_count = 0
            max_revisions = 3

            while decision == "revision" and revision_count < max_revisions:
                revision_count += 1
                print(f"    Revision {revision_count}...")

                # Revise question
                result = revise_question(state)
                state.update(result)

                # Run pipeline again (still expecting no results)
                state.update(analyze_query(state))
                state.update(sample_neo4j_nodes(state))
                state.update(score_semantic_similarity(state))
                state.update(expand_subgraph(state))
                state.update(score_expanded_nodes_with_isrelevant(state))

                # Force empty results for testing
                state["final_relevant_nodes"] = []

                # Evaluate again
                result = evaluate_context(state)
                state.update(result)
                decision = state.get("decision", "unknown")

                print(f"    Decision after revision {revision_count}: {decision}")

            # Force sufficient to test answer generation with no data
            state["decision"] = "sufficient"

            # Generate answer with no relevant nodes
            result = generate_answer(state)
            state.update(result)

            assert "final_answer" in result, "Should generate answer even with no data"
            final_answer = result["final_answer"]
            assert isinstance(final_answer, str), "Answer should be string"
            assert len(final_answer) > 0, "Answer should not be empty"

            # Answer should indicate lack of information
            lower_answer = final_answer.lower()
            lack_indicators = [
                "no information",
                "not found",
                "unable to find",
                "no results",
                "not available",
                "sorry",
                "apologize",
            ]
            has_lack_indicator = any(
                indicator in lower_answer for indicator in lack_indicators
            )

            if not has_lack_indicator:
                print(
                    f"    Answer doesn't clearly indicate lack of information: '{final_answer}...'"
                )
            else:
                print("   Answer appropriately indicates lack of information")

            print("Scenario 4 PASSED: System handles empty/no results gracefully")
            print(f"   Query: '{test_question}'")
            print(f"   Revisions attempted: {revision_count}")
            print(f"   Final answer length: {len(final_answer)} chars")
            print(f"   Answer indicates lack of info: {has_lack_indicator}")

        except Exception as e:
            if "Neo4j" in str(e) or "database" in str(e).lower():
                pytest.skip(f"Database unavailable for empty data E2E test: {e}")
            else:
                raise AssertionError(f"Empty data E2E test failed: {e}")

    def test_scenario_5_complete_langgraph_workflow_e2e(self):
        """Test 5: Complete LangGraph Workflow E2E

        Tests the complete LangGraph workflow using the compiled app.
        This is the ultimate end-to-end test.
        """

        test_question = "What road bikes do you have under $800?"

        scenario_name = "complete_langgraph"
        TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = {}

        try:

            # Prepare initial state for LangGraph
            initial_state = {
                "question": test_question,
                "revision_history": [],
                "sampled_nodes": [],
                "semantic_scored_nodes": [],
                "expanded_nodes": [],
                "expanded_scored_nodes": [],
                "final_relevant_nodes": [],
                "expanded_subgraph": [],
            }

            # Run complete LangGraph workflow
            start_time = time()
            final_state = app.invoke(initial_state, {"recursion_limit": 10})
            total_duration = time() - start_time

            # Store results
            TestMilestone3LangGraphEndToEnd.pipeline_scenarios[scenario_name] = (
                final_state
            )

            # Validate complete workflow results
            assert isinstance(final_state, dict), "Final state should be dict"
            assert "question" in final_state, "Should preserve question"
            assert "final_answer" in final_state, "Should produce final answer"

            final_answer = final_state["final_answer"]
            assert isinstance(final_answer, str), "Final answer should be string"
            assert len(final_answer.strip()) > 0, "Final answer should not be empty"
            assert len(final_answer) > 50, "Final answer should be substantive"

            # Validate workflow progression
            expected_keys = [
                "query_input",
                "sampled_nodes",
                "semantic_scored_nodes",
                "expanded_nodes",
                "final_relevant_nodes",
                "final_answer",
            ]

            for key in expected_keys:
                assert key in final_state, f"Workflow should produce {key}"

            # Validate data flow
            if "query_input" in final_state:
                query_input = final_state["query_input"]
                assert isinstance(query_input, QueryInput), (
                    "Should have valid QueryInput"
                )
                assert query_input.text == test_question, (
                    "Should preserve original question"
                )

            if "final_relevant_nodes" in final_state:
                final_nodes = final_state["final_relevant_nodes"]
                assert isinstance(final_nodes, list), "Final nodes should be list"

                for node in final_nodes:
                    assert hasattr(node, "score"), "Final nodes should have scores"
                    assert hasattr(node, "text"), "Final nodes should have text"

            # Performance validation
            assert total_duration < 300, (
                f"Complete workflow took too long: {total_duration:.2f}s"
            )

            print(f"   Answer preview: '{final_answer}'")

            # Workflow statistics
            if "sampled_nodes" in final_state:
                print(f"   Sampled nodes: {len(final_state['sampled_nodes'])}")
            if "semantic_scored_nodes" in final_state:
                print(f"   Semantic nodes: {len(final_state['semantic_scored_nodes'])}")
            if "final_relevant_nodes" in final_state:
                print(f"   Final nodes: {len(final_state['final_relevant_nodes'])}")
            if "revision_history" in final_state:
                print(f"   Revisions: {len(final_state['revision_history'])}")

        except Exception as e:
            if "Neo4j" in str(e) or "database" in str(e).lower():
                pytest.skip(
                    f"Database unavailable for complete LangGraph E2E test: {e}"
                )
            else:
                raise AssertionError(
                    f"Complete LangGraph workflow E2E test failed: {e}"
                )
