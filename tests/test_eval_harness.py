"""Tests for Eval Harness."""

import pytest
from uuid import UUID, uuid4
from datetime import datetime

from app.core.config import settings
from app.models.schemas import (
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkType,
    EvaluationRequest,
    EvaluationResult,
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    JudgeType,
    StatisticalTest,
    StatisticalResult,
    HealthResponse,
    Leaderboard,
    LeaderboardEntry,
)
from app.services.benchmark_service import BenchmarkService
from app.services.judge_service import JudgeService
from app.services.statistics_service import StatisticsService


class TestConfig:
    """Test configuration loading."""

    def test_settings_defaults(self):
        """Test default settings values."""
        assert settings.app_name == "Eval Harness"
        assert settings.port == 8005
        assert settings.judge_model == "gpt-4o"
        assert settings.confidence_level == 0.95


class TestSchemas:
    """Test Pydantic schemas."""

    def test_benchmark_suite(self):
        """Test benchmark suite model."""
        benchmark = BenchmarkSuite(
            name="Test Benchmark",
            benchmark_type=BenchmarkType.MMLU,
            tasks=[
                BenchmarkTask(
                    id="task1",
                    prompt="Test prompt",
                    expected="A",
                ),
            ],
        )
        assert benchmark.name == "Test Benchmark"
        assert len(benchmark.tasks) == 1

    def test_evaluation_request(self):
        """Test evaluation request model."""
        request = EvaluationRequest(
            benchmark_id=uuid4(),
            model_outputs={"model_a": ["output1", "output2"]},
            judge_type=JudgeType.EXACT_MATCH,
        )
        assert len(request.model_outputs) == 1

    def test_comparison_request(self):
        """Test comparison request model."""
        request = ComparisonRequest(
            comparison_type=ComparisonType.PAIRWISE,
            model_a="model_a",
            model_b="model_b",
            prompts=["prompt1", "prompt2"],
        )
        assert len(request.prompts) == 2

    def test_statistical_test(self):
        """Test statistical test model."""
        test = StatisticalTest(
            test_type="bootstrap",
            confidence_level=0.95,
            num_iterations=1000,
        )
        assert test.test_type == "bootstrap"

    def test_health_response(self):
        """Test health response model."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            benchmarks_available=3,
            judge_ready=True,
        )
        assert health.status == "healthy"
        assert health.benchmarks_available == 3


class TestBenchmarkService:
    """Test benchmark service."""

    @pytest.fixture
    def benchmark_service(self):
        """Create benchmark service instance."""
        service = BenchmarkService()
        service.initialize()
        return service

    def test_initialize(self, benchmark_service):
        """Test initialization."""
        assert benchmark_service._initialized
        assert len(benchmark_service._benchmarks) >= 3  # Built-in benchmarks

    def test_list_benchmarks(self, benchmark_service):
        """Test listing benchmarks."""
        benchmarks = benchmark_service.list_benchmarks()
        assert len(benchmarks) >= 3

    def test_get_benchmark(self, benchmark_service):
        """Test getting a benchmark."""
        benchmarks = benchmark_service.list_benchmarks()
        benchmark_id = benchmarks[0].id
        benchmark = benchmark_service.get_benchmark(benchmark_id)
        assert benchmark is not None
        assert benchmark.id == benchmark_id

    def test_create_custom_benchmark(self, benchmark_service):
        """Test creating custom benchmark."""
        tasks = [
            BenchmarkTask(id="t1", prompt="Test 1", expected="A"),
            BenchmarkTask(id="t2", prompt="Test 2", expected="B"),
        ]
        benchmark = benchmark_service.create_custom_benchmark("Custom Test", tasks)
        assert benchmark.name == "Custom Test"
        assert len(benchmark.tasks) == 2
        assert benchmark.benchmark_type.value == "custom"


class TestJudgeService:
    """Test judge service."""

    @pytest.fixture
    def judge_service(self):
        """Create judge service instance."""
        service = JudgeService()
        service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_compare_pairwise(self, judge_service):
        """Test pairwise comparison."""
        request = ComparisonRequest(
            comparison_type=ComparisonType.PAIRWISE,
            model_a="model_a",
            model_b="model_b",
            prompts=["Test prompt 1", "Test prompt 2"],
            num_judges=1,
        )

        result = await judge_service.compare(request)

        assert isinstance(result, ComparisonResult)
        assert result.model_a == "model_a"
        assert result.model_b == "model_b"
        assert result.comparison_type == ComparisonType.PAIRWISE
        assert result.wins_a + result.wins_b + result.ties == 2
        assert 0 <= result.win_rate_a <= 1
        assert 0 <= result.win_rate_b <= 1


class TestStatisticsService:
    """Test statistics service."""

    @pytest.fixture
    def stats_service(self):
        """Create statistics service instance."""
        service = StatisticsService()
        service.initialize()
        return service

    def test_bootstrap_test(self, stats_service):
        """Test bootstrap test."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [2.0, 3.0, 4.0, 5.0, 6.0]

        test = StatisticalTest(test_type="bootstrap", num_iterations=100)
        result = stats_service.bootstrap_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        assert result.test_type == "bootstrap"
        assert 0 <= result.p_value <= 1
        assert len(result.confidence_interval) == 2

    def test_t_test(self, stats_service):
        """Test t-test."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [10.0, 11.0, 12.0, 13.0, 14.0]  # Clearly different

        test = StatisticalTest(test_type="t_test")
        result = stats_service.t_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        assert result.test_type == "t_test"
        assert 0 <= result.p_value <= 1
        # Should be significant
        assert result.significant

    def test_t_test_not_significant(self, stats_service):
        """Test t-test with similar samples."""
        sample_a = [5.0, 5.1, 4.9, 5.0, 5.0]
        sample_b = [5.0, 5.1, 4.9, 5.0, 5.0]

        test = StatisticalTest(test_type="t_test")
        result = stats_service.t_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        # Should not be significant
        assert not result.significant

    def test_wilcoxon_test(self, stats_service):
        """Test Wilcoxon signed-rank test."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [2.0, 3.0, 4.0, 5.0, 6.0]

        test = StatisticalTest(test_type="wilcoxon")
        result = stats_service.wilcoxon_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        assert result.test_type == "wilcoxon"
        assert 0 <= result.p_value <= 1

    def test_permutation_test(self, stats_service):
        """Test permutation test."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        test = StatisticalTest(test_type="permutation", num_iterations=100)
        result = stats_service.permutation_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        assert result.test_type == "permutation"
        assert 0 <= result.p_value <= 1


class TestEvaluationService:
    """Test evaluation service integration."""

    @pytest.fixture
    async def eval_service(self):
        """Create evaluation service instance."""
        from app.services.evaluation_service import evaluation_service
        await evaluation_service.initialize()
        return evaluation_service

    @pytest.mark.asyncio
    async def test_list_benchmarks(self, eval_service):
        """Test listing benchmarks."""
        benchmarks = eval_service.list_benchmarks()
        assert len(benchmarks) >= 3

    @pytest.mark.asyncio
    async def test_quick_evaluate(self, eval_service):
        """Test quick evaluation."""
        benchmarks = eval_service.list_benchmarks()
        benchmark_id = benchmarks[0].id
        benchmark = eval_service.get_benchmark(benchmark_id)

        result = await eval_service.quick_evaluate(
            benchmark_id=benchmark_id,
            model_name="test_model",
            outputs=[task.expected or "A" for task in benchmark.tasks[:3]],
            judge_type="exact_match",
        )

        assert isinstance(result, EvaluationResult)
        assert result.model_name == "test_model"

    @pytest.mark.asyncio
    async def test_quick_compare(self, eval_service):
        """Test quick comparison."""
        result = await eval_service.quick_compare(
            model_a="model_a",
            model_b="model_b",
            prompts=["Test prompt 1", "Test prompt 2", "Test prompt 3"],
        )

        assert isinstance(result, ComparisonResult)
        assert result.model_a == "model_a"
        assert result.model_b == "model_b"

    @pytest.mark.asyncio
    async def test_statistical_test(self, eval_service):
        """Test statistical test."""
        sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]

        test = StatisticalTest(test_type="t_test")
        result = eval_service.run_statistical_test(sample_a, sample_b, test)

        assert isinstance(result, StatisticalResult)
        assert result.significant  # Should be significant


if __name__ == "__main__":
    pytest.main([__file__, "-v"])