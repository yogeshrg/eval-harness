"""API routes for Eval Harness."""

from fastapi import APIRouter, HTTPException
from typing import Any
from uuid import UUID

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
)
from app.services.evaluation_service import evaluation_service

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return await evaluation_service.get_health()


# ========== Benchmarks ==========

@router.get("/benchmarks", response_model=list[BenchmarkSuite])
async def list_benchmarks() -> list[BenchmarkSuite]:
    """List all available benchmarks."""
    return evaluation_service.list_benchmarks()


@router.get("/benchmarks/{benchmark_id}", response_model=BenchmarkSuite)
async def get_benchmark(benchmark_id: UUID) -> BenchmarkSuite:
    """Get a benchmark by ID."""
    benchmark = evaluation_service.get_benchmark(benchmark_id)
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return benchmark


@router.post("/benchmarks", response_model=BenchmarkSuite)
async def create_benchmark(
    name: str,
    benchmark_type: BenchmarkType = BenchmarkType.CUSTOM,
    tasks: list[BenchmarkTask] = [],
) -> BenchmarkSuite:
    """Create a custom benchmark."""
    return evaluation_service.create_benchmark(name, tasks)


# ========== Evaluation ==========

@router.post("/evaluate", response_model=list[EvaluationResult])
async def evaluate(request: EvaluationRequest) -> list[EvaluationResult]:
    """Evaluate model outputs against a benchmark."""
    try:
        return await evaluation_service.evaluate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate/quick", response_model=EvaluationResult)
async def quick_evaluate(
    benchmark_id: UUID,
    model_name: str,
    outputs: list[str],
    judge_type: str = "exact_match",
) -> EvaluationResult:
    """Quick evaluation with defaults."""
    result = await evaluation_service.quick_evaluate(
        benchmark_id=benchmark_id,
        model_name=model_name,
        outputs=outputs,
        judge_type=judge_type,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation failed")
    return result


# ========== Comparison ==========

@router.post("/compare", response_model=ComparisonResult)
async def compare(request: ComparisonRequest) -> ComparisonResult:
    """Run pairwise comparison between models."""
    try:
        return await evaluation_service.compare(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare/quick", response_model=ComparisonResult)
async def quick_compare(
    model_a: str,
    model_b: str,
    prompts: list[str],
    comparison_type: str = "pairwise",
) -> ComparisonResult:
    """Quick pairwise comparison."""
    return await evaluation_service.quick_compare(model_a, model_b, prompts, comparison_type)


# ========== Statistics ==========

@router.post("/statistics/test", response_model=StatisticalResult)
async def run_statistical_test(
    sample_a: list[float],
    sample_b: list[float],
    test: StatisticalTest,
) -> StatisticalResult:
    """Run statistical significance test."""
    try:
        return evaluation_service.run_statistical_test(sample_a, sample_b, test)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/statistics/compare", response_model=StatisticalResult)
async def compare_results(
    result_a: ComparisonResult,
    result_b: ComparisonResult,
) -> StatisticalResult:
    """Compare two comparison results."""
    return evaluation_service.compare_evaluation_results(result_a, result_b)


# ========== Leaderboard ==========

@router.get("/leaderboard/{benchmark_id}", response_model=Leaderboard)
async def get_leaderboard(benchmark_id: UUID) -> Leaderboard:
    """Get leaderboard for a benchmark."""
    leaderboard = evaluation_service.get_leaderboard(benchmark_id)
    if not leaderboard:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    return leaderboard


@router.get("/leaderboard/{benchmark_id}/top/{n}", response_model=list[dict[str, Any]])
async def get_top_models(benchmark_id: UUID, n: int = 10) -> list[dict[str, Any]]:
    """Get top N models from leaderboard."""
    leaderboard = evaluation_service.get_leaderboard(benchmark_id)
    if not leaderboard:
        raise HTTPException(status_code=404, detail="Benchmark not found")

    top = leaderboard.entries[:n]
    return [
        {
            "rank": e.rank,
            "model_name": e.model_name,
            "score": e.score,
        }
        for e in top
    ]