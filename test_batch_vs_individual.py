#!/usr/bin/env python3
"""
Batch vs Individual Processing Performance Test

This script specifically tests the performance difference between batch processing
and individual node processing for different isRelevant scoring approaches.
"""

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

# Scoring approaches to test (focusing on LLM-heavy ones for batch benefits)
SCORING_APPROACHES = [
    ("Composite (Weighted)", ScorerType.COMPOSITE),
    ("Parallel (Max)", ScorerType.PARALLEL),
    ("Router All Metrics", ScorerType.ROUTER_ALL),
    ("Router Two: Semantic + LLM", ScorerType.ROUTER_TWO_SEM_LLM),
    ("Single: LLM Judge Only", ScorerType.ROUTER_SINGLE_LLM),
]

# Batch configurations to test
BATCH_CONFIGS = [
    ("Individual Processing", False, 1),
    ("Batch Size 5", True, 5),
    ("Batch Size 10", True, 10),
    ("Batch Size 15", True, 15),
]

class PerformanceResult:
    """Class to store performance test results."""
    
    def __init__(self, approach_name: str, scorer_type: ScorerType, batch_enabled: bool, batch_size: int):
        self.approach_name = approach_name
        self.scorer_type = scorer_type
        self.batch_enabled = batch_enabled
        self.batch_size = batch_size
        self.execution_time = 0.0
        self.final_answer = ""
        self.final_nodes_count = 0
        self.avg_final_score = 0.0
        self.high_relevance_count = 0
        self.success = False
        self.error_message = ""

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

def run_performance_test(approach_name: str, scorer_type: ScorerType, batch_enabled: bool, batch_size: int) -> PerformanceResult:
    """Run a single performance test with the specified configuration."""
    
    result = PerformanceResult(approach_name, scorer_type, batch_enabled, batch_size)
    
    config_name = f"Batch {batch_size}" if batch_enabled else "Individual"
    print(f"\n{'=' * 70}")
    print(f"🧪 TESTING: {approach_name} | {config_name}")
    print(f"📊 Scorer: {scorer_type.value} | Batch: {batch_enabled} | Size: {batch_size}")
    print(f"{'=' * 70}")
    
    try:
        # Configure the test
        set_random_seed(TEST_SEED)
        set_scorer_type(scorer_type)
        set_batch_processing(batch_enabled, batch_size)
        
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
        
        if "final_relevant_nodes" in final_state:
            final_nodes = final_state["final_relevant_nodes"]
            result.final_nodes_count = len(final_nodes)
            
            if final_nodes:
                scores = [getattr(node, "score", 0) for node in final_nodes]
                result.avg_final_score = sum(scores) / len(scores)
                result.high_relevance_count = len([n for n in final_nodes if getattr(n, "score", 0) > 0.7])
        
        print(f"✅ Test completed successfully in {result.execution_time:.2f}s")
        print(f"📈 Final nodes: {result.final_nodes_count}")
        print(f"🎯 High relevance nodes (>0.7): {result.high_relevance_count}")
        print(f"📊 Average final score: {result.avg_final_score:.3f}")
        
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        print(f"❌ Test failed: {e}")
    
    return result

def analyze_performance_results(results: List[PerformanceResult]) -> Dict[str, Any]:
    """Analyze performance comparison results."""
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "test_question": TEST_QUESTION,
        "test_seed": TEST_SEED,
        "total_tests": len(results),
        "successful_tests": len([r for r in results if r.success]),
        "failed_tests": len([r for r in results if not r.success]),
        "approach_comparisons": {},
        "batch_size_analysis": {},
        "performance_summary": [],
    }
    
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        analysis["error"] = "No successful tests to analyze"
        return analysis
    
    # Group by approach for comparison
    approach_groups = {}
    for result in successful_results:
        if result.approach_name not in approach_groups:
            approach_groups[result.approach_name] = []
        approach_groups[result.approach_name].append(result)
    
    # Analyze each approach
    for approach_name, approach_results in approach_groups.items():
        individual_result = None
        batch_results = []
        
        for result in approach_results:
            if not result.batch_enabled:
                individual_result = result
            else:
                batch_results.append(result)
        
        if individual_result and batch_results:
            best_batch = min(batch_results, key=lambda x: x.execution_time)
            speedup = individual_result.execution_time / best_batch.execution_time if best_batch.execution_time > 0 else 1.0
            
            analysis["approach_comparisons"][approach_name] = {
                "individual_time": round(individual_result.execution_time, 2),
                "best_batch_time": round(best_batch.execution_time, 2),
                "best_batch_size": best_batch.batch_size,
                "speedup": round(speedup, 2),
                "time_saved": round(individual_result.execution_time - best_batch.execution_time, 2),
                "score_consistency": abs(individual_result.avg_final_score - best_batch.avg_final_score) < 0.05
            }
    
    # Batch size analysis
    batch_sizes = set()
    for result in successful_results:
        if result.batch_enabled:
            batch_sizes.add(result.batch_size)
    
    for batch_size in sorted(batch_sizes):
        batch_size_results = [r for r in successful_results if r.batch_enabled and r.batch_size == batch_size]
        if batch_size_results:
            avg_time = sum(r.execution_time for r in batch_size_results) / len(batch_size_results)
            analysis["batch_size_analysis"][f"batch_{batch_size}"] = {
                "average_time": round(avg_time, 2),
                "test_count": len(batch_size_results)
            }
    
    # Performance summary
    for result in successful_results:
        config_name = f"Batch {result.batch_size}" if result.batch_enabled else "Individual" 
        analysis["performance_summary"].append({
            "approach": result.approach_name,
            "configuration": config_name,
            "execution_time": round(result.execution_time, 2),
            "avg_final_score": round(result.avg_final_score, 3),
            "high_relevance_count": result.high_relevance_count,
            "final_nodes_count": result.final_nodes_count,
            "success": result.success
        })
    
    return analysis

def print_performance_report(analysis: Dict[str, Any]):
    """Print a comprehensive performance analysis report."""
    
    print(f"\n{'=' * 80}")
    print(f"🚀 BATCH vs INDIVIDUAL PROCESSING PERFORMANCE REPORT")
    print(f"{'=' * 80}")
    print(f"🕐 Test Run: {analysis['timestamp']}")
    print(f"❓ Question: {analysis['test_question']}")
    print(f"🎯 Random Seed: {analysis['test_seed']}")
    print(f"✅ Successful Tests: {analysis['successful_tests']}/{analysis['total_tests']}")
    
    if analysis["failed_tests"] > 0:
        print(f"❌ Failed Tests: {analysis['failed_tests']}")
    
    print(f"\n{'🏆 APPROACH-BY-APPROACH PERFORMANCE COMPARISON'}")
    print(f"{'-' * 80}")
    
    for approach, comparison in analysis["approach_comparisons"].items():
        print(f"\n🔍 {approach}")
        print(f"   Individual: {comparison['individual_time']:6.2f}s")
        print(f"   Best Batch: {comparison['best_batch_time']:6.2f}s (size {comparison['best_batch_size']})")
        print(f"   Speedup:    {comparison['speedup']:6.2f}x")
        print(f"   Time Saved: {comparison['time_saved']:6.2f}s")
        print(f"   Score Consistent: {'✅' if comparison['score_consistency'] else '❌'}")
    
    print(f"\n{'📊 BATCH SIZE ANALYSIS'}")
    print(f"{'-' * 50}")
    
    for batch_config, stats in analysis["batch_size_analysis"].items():
        batch_size = batch_config.replace("batch_", "")
        print(f"Batch Size {batch_size:2s}: {stats['average_time']:6.2f}s avg ({stats['test_count']} tests)")
    
    print(f"\n{'📈 DETAILED PERFORMANCE SUMMARY'}")
    print(f"{'-' * 80}")
    
    for summary in analysis["performance_summary"]:
        config_width = 12
        print(f"{summary['approach']:25s} | {summary['configuration']:>{config_width}s} | "
              f"Time: {summary['execution_time']:6.2f}s | "
              f"Score: {summary['avg_final_score']:.3f} | "
              f"High: {summary['high_relevance_count']}")

def save_performance_results(analysis: Dict[str, Any], filename: str = None):
    """Save performance analysis results to JSON file."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_performance_results_{timestamp}.json"
    
    try:
        with open(filename, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Performance results saved to: {filename}")
    except Exception as e:
        print(f"\n⚠️ Failed to save results: {e}")

def main():
    """Run the batch vs individual processing performance test."""
    
    print("🚀 Starting Batch vs Individual Processing Performance Test")
    print("=" * 80)
    
    # Check Neo4j connection
    if not test_neo4j_connection():
        print("❌ Cannot proceed without Neo4j connection")
        return
    
    # Run all tests
    results = []
    
    for approach_name, scorer_type in SCORING_APPROACHES:
        for config_name, batch_enabled, batch_size in BATCH_CONFIGS:
            result = run_performance_test(approach_name, scorer_type, batch_enabled, batch_size)
            results.append(result)
            
            # Reset configuration between tests
            reset_global_config()
            
            # Small delay between tests
            time.sleep(1)
    
    # Analyze results
    print(f"\n{'=' * 60}")
    print("🔍 ANALYZING PERFORMANCE RESULTS...")
    print(f"{'=' * 60}")
    
    analysis = analyze_performance_results(results)
    
    # Print performance report
    print_performance_report(analysis)
    
    # Save results
    save_performance_results(analysis)
    
    print(f"\n{'=' * 80}")
    print("🎉 BATCH PERFORMANCE TEST COMPLETED!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 