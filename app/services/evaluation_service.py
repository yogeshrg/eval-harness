"""Main evaluation service coordinating benchmarks, judges, and statistics."""

import logging
from typing import Any
from uuid import UUID, uuid4

from app.core.config import settings
from app.models.schemas import (
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkType,
    EvaluationRequest,
    EvaluationResult,
    ComparisonRequest,
    ComparisonResult,
    StatisticalTest,
    StatisticalResult,
    HealthResponse,
    Leaderboard,
    LeaderboardEntry,
)
from app.services.benchmark_service import benchmark_service
from app.services.judge_service import judge_service
from app.services.statistics_service import statistics_service

logger = logging.getLogger(__name__)


class EvaluationService:
    """Unified evaluation service."""

    def __init__(self) -> None:
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all sub-services."""
        if self._initialized:
            return

        logger.info("Initializing evaluation service...")
        benchmark_service.initialize()
        judge_service.initialize()
        statistics_service.initialize()

        self._initialized = True
        logger.info("Evaluation service initialized")

    # ========== Benchmarks ==========

    def list_benchmarks(self) -> list[BenchmarkSuite]:
        """List all available benchmarks."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return benchmark_service.list_benchmarks()

    def get_benchmark(self, benchmark_id: UUID) -> BenchmarkSuite | None:
        """Get a benchmark by ID."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return benchmark_service.get_benchmark(benchmark_id)

    def create_benchmark(self, name: str, tasks: list[BenchmarkTask]) -> BenchmarkSuite:
        """Create a custom benchmark."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return benchmark_service.create_custom_benchmark(name, tasks)

    # ========== Evaluation ==========

    async def evaluate(self, request: EvaluationRequest) -> list[EvaluationResult]:
        """Evaluate model outputs against a benchmark."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return await benchmark_service.evaluate(request)

    async def quick_evaluate(
        self,
        benchmark_id: UUID,
        model_name: str,
        outputs: list[str],
        judge_type: str = "exact_match",
    ) -> EvaluationResult:
        """Quick evaluation with default settings."""
        request = EvaluationRequest(
            benchmark_id=benchmark_id,
            model_outputs={model_name: outputs},
            judge_type=judge_type,
        )
        results = await self.evaluate(request)
        return results[0] if results else None

    # ========== Comparison ==========

    async def compare(self, request: ComparisonRequest) -> ComparisonResult:
        """Run pairwise comparison between models."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return await judge_service.compare(request)

    async def quick_compare(
        self,
        model_a: str,
        model_b: str,
        prompts: list[str],
        comparison_type: str = "pairwise",
    ) -> ComparisonResult:
        """Quick pairwise comparison."""
        request = ComparisonRequest(
            comparison_type=comparison_type,
            model_a=model_a,
            model_b=model_b,
            prompts=prompts,
        )
        return await self.compare(request)

    # ========== Statistics ==========

    def run_statistical_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run statistical significance test."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return statistics_service.run_test(sample_a, sample_b, test)

    def compare_evaluation_results(
        self,
        result_a: ComparisonResult,
        result_b: ComparisonResult,
    ) -> StatisticalResult:
        """Compare two comparison results."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")
        return statistics_service.compare_results(result_a, result_b)

    # ========== Leaderboard ==========

    def get_leaderboard(self, benchmark_id: UUID) -> Leaderboard | None:
        """Get leaderboard for a benchmark."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        benchmark = benchmark_service.get_benchmark(benchmark_id)
        if not benchmark:
            return None

        # Load results from disk
        entries = self._load_leaderboard_entries(benchmark_id)

        entries.sort(key=lambda x: x.score, reverse=True)
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return Leaderboard(
            benchmark_id=benchmark_id,
            benchmark_name=benchmark.name,
            entries=entries,
        )

    def _load_leaderboard_entries(self, benchmark_id: UUID) -> list[LeaderboardEntry]:
        """Load leaderboard entries from results."""
        import json
        from pathlib import Path

        entries = []
        results_dir = settings.results_dir

        for filepath in results_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    if data.get("benchmark_id") == str(benchmark_id):
                        score = data.get("scores", {}).get("mean", 0)
                        entries.append(LeaderboardEntry(
                            rank=0,
                            model_name=data.get("model_name", "Unknown"),
                            score=score,
                            benchmark_id=benchmark_id,
                            metadata=data.get("metadata", {}),
                        ))
            except Exception:
                continue

        return entries

    # ========== Health ==========

    async def get_health(self) -> HealthResponse:
        """Get health status."""
        benchmarks = benchmark_service.list_benchmarks()
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            benchmarks_available=len(benchmarks),
            judge_ready=True,
        )


# Global instance
evaluation_service = EvaluationService()