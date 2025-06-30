#!/usr/bin/env python3
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Import the main workflow and configuration functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from neo4j_rag_langgraph import (
    app,
    set_scorer_type,
    set_random_seed,
    set_batch_processing,
    reset_global_config,
    neo4j_driver,
    ScorerType,
)

# Test configuration
TEST_SEED = 42  # Fixed seed for consistent node sampling
TEST_QUESTION = "Show me products under $500"

# All scoring approaches to test
SCORING_APPROACHES = [
    ("Composite (Weighted)", ScorerType.COMPOSITE),
    ("Parallel (Max)", ScorerType.PARALLEL),
    ("Router (Original)", ScorerType.ROUTER),
    ("Router All Metrics", ScorerType.ROUTER_ALL),
    ("Router Two: Semantic + LLM", ScorerType.ROUTER_TWO_SEM_LLM),
    ("Router Two: Entity + Type", ScorerType.ROUTER_TWO_ENT_TYPE),
    ("Single: Semantic Only", ScorerType.ROUTER_SINGLE_SEM),
    ("Single: LLM Judge Only", ScorerType.ROUTER_SINGLE_LLM),
    ("Single: Entity Match Only", ScorerType.ROUTER_SINGLE_ENT),
    ("Single: Node Type Only", ScorerType.ROUTER_SINGLE_TYPE),
]

# Batch processing test configurations
BATCH_CONFIGS = [
    ("Batch Size 5", True, 5),
    ("Batch Size 10", True, 10),
    ("Batch Size 20", True, 20),
    ("Individual Processing", False, 1),
]


class TestResult:
    """Class to store test results for analysis."""

    def __init__(self, approach_name: str, scorer_type: ScorerType, batch_enabled: bool = True, batch_size: int = 10):
        self.approach_name = approach_name
        self.scorer_type = scorer_type
        self.batch_enabled = batch_enabled
        self.batch_size = batch_size
        self.execution_time = 0.0
        self.final_answer = ""
        self.semantic_nodes = []
        self.final_nodes = []
        self.expanded_nodes_count = 0
        self.avg_semantic_score = 0.0
        self.avg_final_score = 0.0
        self.high_relevance_count = 0
        self.success = False
        self.error_message = ""
        self.batch_processing_used = False
        self.processing_mode = "unknown"


def test_neo4j_connection() -> bool:
    """Test if Neo4j is available."""
    try:
        with neo4j_driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as total_nodes").single()
            total_nodes = result["total_nodes"]
            print(f"✅ Neo4j connected - {total_nodes} nodes available")
            return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


def run_single_test(approach_name: str, scorer_type: ScorerType, batch_enabled: bool = True, batch_size: int = 10) -> TestResult:
    """Run a single test with the specified scoring approach and batch configuration."""

    result = TestResult(approach_name, scorer_type, batch_enabled, batch_size)

    batch_info = f"batch_size={batch_size}" if batch_enabled else "individual"
    print(f"\n{'=' * 60}")
    print(f"🧪 TESTING: {approach_name}")
    print(f"📊 Scorer Type: {scorer_type.value}")
    print(f"⚡ Processing: {batch_info}")
    print(f"{'=' * 60}")

    try:
        # Configure the test
        set_random_seed(TEST_SEED)
        set_scorer_type(scorer_type)
        set_batch_processing(batch_enabled, batch_size)
        
        result.processing_mode = "batch" if batch_enabled else "individual"

        # Prepare inputs
        inputs = {
            "question": TEST_QUESTION,
            "revision_history": [],
            "sampled_nodes": [],
            "semantic_scored_nodes": [],
            "expanded_nodes": [],
            "expanded_scored_nodes": [],
            "final_relevant_nodes": [],
            "expanded_subgraph": [],
        }

        # Run the workflow
        start_time = time.time()
        final_state = app.invoke(inputs, {"recursion_limit": 12})
        end_time = time.time()

        result.execution_time = end_time - start_time
        result.success = True

        # Extract results
        if "final_answer" in final_state:
            result.final_answer = final_state["final_answer"]

        if "semantic_scored_nodes" in final_state:
            result.semantic_nodes = final_state["semantic_scored_nodes"]
            if result.semantic_nodes:
                scores = [getattr(node, "score", 0) for node in result.semantic_nodes]
                result.avg_semantic_score = sum(scores) / len(scores)

        if "final_relevant_nodes" in final_state:
            result.final_nodes = final_state["final_relevant_nodes"]
            if result.final_nodes:
                scores = [getattr(node, "score", 0) for node in result.final_nodes]
                result.avg_final_score = sum(scores) / len(scores)
                result.high_relevance_count = len(
                    [n for n in result.final_nodes if getattr(n, "score", 0) > 0.7]
                )

        if "expanded_nodes" in final_state:
            result.expanded_nodes_count = len(final_state["expanded_nodes"])

        print(f"✅ Test completed successfully in {result.execution_time:.2f}s")
        print(f"📈 Final nodes: {len(result.final_nodes)}")
        print(f"🎯 High relevance nodes (>0.7): {result.high_relevance_count}")
        print(f"📊 Average final score: {result.avg_final_score:.3f}")

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        print(f"❌ Test failed: {e}")

    return result


def analyze_results(results: List[TestResult]) -> Dict[str, Any]:
    """Analyze and compare test results."""

    analysis = {
        "timestamp": datetime.now().isoformat(),
        "test_question": TEST_QUESTION,
        "test_seed": TEST_SEED,
        "total_tests": len(results),
        "successful_tests": len([r for r in results if r.success]),
        "failed_tests": len([r for r in results if not r.success]),
        "results_summary": [],
        "performance_ranking": [],
        "score_distribution": {},
    }

    successful_results = [r for r in results if r.success]

    if not successful_results:
        analysis["error"] = "No successful tests to analyze"
        return analysis

    # Performance ranking by average final score
    performance_ranking = sorted(
        successful_results, key=lambda x: x.avg_final_score, reverse=True
    )
    analysis["performance_ranking"] = [
        {
            "rank": i + 1,
            "approach": r.approach_name,
            "scorer_type": r.scorer_type.value,
            "avg_final_score": round(r.avg_final_score, 3),
            "high_relevance_count": r.high_relevance_count,
            "execution_time": round(r.execution_time, 2),
            "final_nodes_count": len(r.final_nodes),
        }
        for i, r in enumerate(performance_ranking)
    ]

    # Results summary
    for result in results:
        summary = {
            "approach": result.approach_name,
            "scorer_type": result.scorer_type.value,
            "processing_mode": result.processing_mode,
            "batch_enabled": result.batch_enabled,
            "batch_size": result.batch_size,
            "success": result.success,
            "execution_time": round(result.execution_time, 2),
            "final_answer": result.final_answer if result.final_answer else "",
            "final_answer_length": len(result.final_answer)
            if result.final_answer
            else 0,
            "semantic_nodes_count": len(result.semantic_nodes),
            "final_nodes_count": len(result.final_nodes),
            "expanded_nodes_count": result.expanded_nodes_count,
            "avg_semantic_score": round(result.avg_semantic_score, 3),
            "avg_final_score": round(result.avg_final_score, 3),
            "high_relevance_count": result.high_relevance_count,
        }

        if not result.success:
            summary["error"] = result.error_message

        analysis["results_summary"].append(summary)

    # Score distribution analysis
    score_ranges = {
        "0.0-0.3": 0,
        "0.3-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9-1.0": 0,
    }

    for result in successful_results:
        for node in result.final_nodes:
            score = getattr(node, "score", 0)
            if score < 0.3:
                score_ranges["0.0-0.3"] += 1
            elif score < 0.5:
                score_ranges["0.3-0.5"] += 1
            elif score < 0.7:
                score_ranges["0.5-0.7"] += 1
            elif score < 0.9:
                score_ranges["0.7-0.9"] += 1
            else:
                score_ranges["0.9-1.0"] += 1

    analysis["score_distribution"] = score_ranges

    return analysis


def print_analysis_report(analysis: Dict[str, Any]):
    """Print a comprehensive analysis report."""

    print(f"\n{'=' * 80}")
    print(f"📊 COMPREHENSIVE TEST ANALYSIS REPORT")
    print(f"{'=' * 80}")
    print(f"🕐 Test Run: {analysis['timestamp']}")
    print(f"❓ Question: {analysis['test_question']}")
    print(f"🎯 Random Seed: {analysis['test_seed']}")
    print(
        f"✅ Successful Tests: {analysis['successful_tests']}/{analysis['total_tests']}"
    )

    if analysis["failed_tests"] > 0:
        print(f"❌ Failed Tests: {analysis['failed_tests']}")

    print(f"\n{'🏆 PERFORMANCE RANKING (by avg final score)'}")
    print(f"{'-' * 80}")

    for rank_data in analysis["performance_ranking"]:
        print(
            f"{rank_data['rank']:2d}. {rank_data['approach']:25s} | "
            f"Score: {rank_data['avg_final_score']:.3f} | "
            f"High: {rank_data['high_relevance_count']} | "
            f"Time: {rank_data['execution_time']:5.2f}s | "
            f"Nodes: {rank_data['final_nodes_count']}"
        )

    print(f"\n{'📈 SCORE DISTRIBUTION ACROSS ALL APPROACHES'}")
    print(f"{'-' * 50}")
    for range_name, count in analysis["score_distribution"].items():
        print(f"{range_name:10s}: {count:3d} nodes")

    # Show the best performing approach's full answer
    if analysis["performance_ranking"]:
        best_approach_name = analysis["performance_ranking"][0]["approach"]
        best_result = next((r for r in analysis["results_summary"] if r["approach"] == best_approach_name), None)
        if best_result and best_result.get("final_answer"):
            print(f"\n{'🏆 BEST PERFORMING APPROACH FULL ANSWER'}")
            print(f"{'-' * 60}")
            print(f"Approach: {best_approach_name}")
            print(f"Score: {analysis['performance_ranking'][0]['avg_final_score']:.3f}")
            print(f"Answer:")
            print("-" * 40)
            print(best_result["final_answer"])
            print("-" * 40)

    print(f"\n{'📝 DETAILED RESULTS SUMMARY'}")
    print(f"{'-' * 80}")

    for summary in analysis["results_summary"]:
        print(f"\n🔍 {summary['approach']}")
        print(f"   Scorer: {summary['scorer_type']}")
        print(f"   Processing: {summary['processing_mode']}" + 
              (f" (batch_size={summary['batch_size']})" if summary['batch_enabled'] else ""))
        if summary["success"]:
            print(f"   ✅ Success in {summary['execution_time']}s")
            print(
                f"   📊 Scores: Semantic {summary['avg_semantic_score']:.3f} → Final {summary['avg_final_score']:.3f}"
            )
            print(
                f"   📈 Nodes: {summary['semantic_nodes_count']} semantic → {summary['final_nodes_count']} final (+{summary['expanded_nodes_count']} expanded)"
            )
            print(f"   🎯 High relevance: {summary['high_relevance_count']} nodes")
            print(f"   📝 Answer length: {summary['final_answer_length']} chars")
            
            # Show preview of final answer
            if summary['final_answer']:
                answer_preview = summary['final_answer'][:100] + "..." if len(summary['final_answer']) > 100 else summary['final_answer']
                print(f"   💬 Answer preview: {answer_preview}")
        else:
            print(f"   ❌ Failed: {summary['error']}")


def save_results_to_file(analysis: Dict[str, Any], filename: str = None):
    """Save analysis results to JSON file."""

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"

    try:
        with open(filename, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️ Failed to save results: {e}")


def run_batch_processing_tests():
    """Run batch processing performance comparison tests."""
    print("🚀 Starting Batch Processing Performance Tests")
    print("=" * 80)
    
    # Test with Composite scorer (best overall performer)
    test_scorer = ScorerType.COMPOSITE
    batch_results = []
    
    for batch_name, batch_enabled, batch_size in BATCH_CONFIGS:
        print(f"\n🔧 Testing {batch_name}...")
        result = run_single_test(f"Composite ({batch_name})", test_scorer, batch_enabled, batch_size)
        batch_results.append(result)
        
        # Reset configuration between tests
        reset_global_config()
        time.sleep(1)
    
    # Analyze batch processing results
    print(f"\n{'=' * 60}")
    print("📊 BATCH PROCESSING ANALYSIS")
    print(f"{'=' * 60}")
    
    successful_batch_results = [r for r in batch_results if r.success]
    
    if successful_batch_results:
        print("Performance Comparison:")
        print("-" * 40)
        for result in successful_batch_results:
            mode = f"Batch({result.batch_size})" if result.batch_enabled else "Individual"
            print(f"{mode:15s} | Time: {result.execution_time:6.2f}s | Score: {result.avg_final_score:.3f} | Nodes: {len(result.final_nodes)}")
        
        # Find fastest approach
        fastest = min(successful_batch_results, key=lambda x: x.execution_time)
        print(f"\n🏆 Fastest: {fastest.approach_name} ({fastest.execution_time:.2f}s)")
        
        # Calculate efficiency gains
        individual_result = next((r for r in successful_batch_results if not r.batch_enabled), None)
        if individual_result:
            print(f"\nEfficiency Gains vs Individual Processing:")
            for result in successful_batch_results:
                if result.batch_enabled:
                    speedup = individual_result.execution_time / result.execution_time
                    print(f"  Batch({result.batch_size:2d}): {speedup:.2f}x faster")
    
    return batch_results

def main():
    """Run the comprehensive test suite."""

    print("🚀 Starting Comprehensive Neo4j RAG Scoring Approaches Test")
    print("=" * 80)

    # Check Neo4j connection
    if not test_neo4j_connection():
        print("❌ Cannot proceed without Neo4j connection")
        return

    # Run main scoring approaches tests (with default batch processing)
    print("\n📊 PHASE 1: Testing All Scoring Approaches")
    print("=" * 60)
    
    results = []
    
    for approach_name, scorer_type in SCORING_APPROACHES:
        result = run_single_test(approach_name, scorer_type)
        results.append(result)

        # Reset configuration between tests
        reset_global_config()

        # Small delay between tests
        time.sleep(1)

    # Run batch processing comparison tests
    print("\n⚡ PHASE 2: Batch Processing Performance Analysis")
    print("=" * 60)
    
    batch_results = run_batch_processing_tests()
    
    # Combine all results
    all_results = results + batch_results

    # Analyze results
    print(f"\n{'=' * 60}")
    print("🔍 ANALYZING ALL RESULTS...")
    print(f"{'=' * 60}")

    analysis = analyze_results(all_results)

    # Print analysis report
    print_analysis_report(analysis)

    # Save results
    save_results_to_file(analysis)

    print(f"\n{'=' * 80}")
    print("🎉 COMPREHENSIVE TEST COMPLETED!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
