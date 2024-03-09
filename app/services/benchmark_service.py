"""Benchmark service for managing evaluation tasks."""

import json
import logging
import random
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from app.core.config import settings
from app.models.schemas import (
    BenchmarkSuite,
    BenchmarkTask,
    BenchmarkType,
    EvaluationRequest,
    EvaluationResult,
    JudgeType,
)

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Service for managing benchmarks and evaluations."""

    def __init__(self) -> None:
        self._benchmarks: dict[UUID, BenchmarkSuite] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return

        settings.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        settings.results_dir.mkdir(parents=True, exist_ok=True)

        # Load built-in benchmarks
        self._load_builtin_benchmarks()

        self._initialized = True
        logger.info("Benchmark service initialized")

    def _load_builtin_benchmarks(self) -> None:
        """Load built-in benchmark suites."""
        # MMLU-style multiple choice
        mmlu = BenchmarkSuite(
            name="MMLU (Sample)",
            benchmark_type=BenchmarkType.MMLU,
            tasks=[
                BenchmarkTask(
                    id="mmlu_1",
                    prompt="What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Madrid",
                    expected="B",
                    category="geography",
                ),
                BenchmarkTask(
                    id="mmlu_2",
                    prompt="Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn",
                    expected="B",
                    category="astronomy",
                ),
                BenchmarkTask(
                    id="mmlu_3",
                    prompt="What is 2 + 2?\nA) 3\nB) 4\nC) 5\nD) 6",
                    expected="B",
                    category="math",
                ),
            ],
        )

        # HumanEval-style code generation
        humaneval = BenchmarkSuite(
            name="HumanEval (Sample)",
            benchmark_type=BenchmarkType.HUMANEVAL,
            tasks=[
                BenchmarkTask(
                    id="he_1",
                    prompt="Write a function that returns the sum of two numbers.",
                    expected="def add(a, b):\n    return a + b",
                    category="basic",
                ),
                BenchmarkTask(
                    id="he_2",
                    prompt="Write a function to reverse a string.",
                    expected="def reverse(s):\n    return s[::-1]",
                    category="strings",
                ),
            ],
        )

        # GSM8K-style math reasoning
        gsm8k = BenchmarkSuite(
            name="GSM8K (Sample)",
            benchmark_type=BenchmarkType.GSM8K,
            tasks=[
                BenchmarkTask(
                    id="gsm_1",
                    prompt="If John has 5 apples and gives 2 to Mary, how many does he have left?",
                    expected="3",
                    category="arithmetic",
                ),
                BenchmarkTask(
                    id="gsm_2",
                    prompt="A train travels 60 miles in 1.5 hours. What is its speed?",
                    expected="40 mph",
                    category="word_problem",
                ),
            ],
        )

        for benchmark in [mmlu, humaneval, gsm8k]:
            self._benchmarks[benchmark.id] = benchmark

    def list_benchmarks(self) -> list[BenchmarkSuite]:
        """List all available benchmarks."""
        if not self._initialized:
            self.initialize()
        return list(self._benchmarks.values())

    def get_benchmark(self, benchmark_id: UUID) -> BenchmarkSuite | None:
        """Get a benchmark by ID."""
        if not self._initialized:
            self.initialize()
        return self._benchmarks.get(benchmark_id)

    def create_custom_benchmark(self, name: str, tasks: list[BenchmarkTask]) -> BenchmarkSuite:
        """Create a custom benchmark suite."""
        if not self._initialized:
            self.initialize()

        benchmark = BenchmarkSuite(
            name=name,
            benchmark_type=BenchmarkType.CUSTOM,
            tasks=tasks,
        )
        self._benchmarks[benchmark.id] = benchmark

        # Save to disk
        self._save_benchmark(benchmark)

        return benchmark

    def _save_benchmark(self, benchmark: BenchmarkSuite) -> None:
        """Save benchmark to disk."""
        filepath = settings.benchmarks_dir / f"{benchmark.id}.json"
        with open(filepath, "w") as f:
            json.dump(benchmark.model_dump(), f, default=str, indent=2)

    async def evaluate(self, request: EvaluationRequest) -> list[EvaluationResult]:
        """Evaluate model outputs against a benchmark."""
        if not self._initialized:
            self.initialize()

        benchmark = self._benchmarks.get(request.benchmark_id)
        if not benchmark:
            raise ValueError(f"Benchmark not found: {request.benchmark_id}")

        results = []

        for model_name, outputs in request.model_outputs.items():
            if len(outputs) != len(benchmark.tasks):
                logger.warning(f"Output count mismatch for {model_name}")
                continue

            aggregate_scores, per_task_scores = await self._evaluate_model(benchmark, outputs, request)

            result = EvaluationResult(
                benchmark_id=benchmark.id,
                model_name=model_name,
                scores=aggregate_scores,
                judge_type=request.judge_type,
                judge_config=request.judge_config,
                per_task_scores=per_task_scores,
            )
            results.append(result)

            # Save result
            self._save_result(result)

        return results

    async def _evaluate_model(
        self,
        benchmark: BenchmarkSuite,
        outputs: list[str],
        request: EvaluationRequest,
    ) -> tuple[dict[str, float], list[float]]:
        """Evaluate a single model's outputs."""
        per_task_scores = []

        for task, output in zip(benchmark.tasks, outputs):
            score = await self._score_output(task, output, request)
            per_task_scores.append(score)

        # Aggregate scores
        aggregate_scores = {}
        if per_task_scores:
            aggregate_scores["mean"] = sum(per_task_scores) / len(per_task_scores)
            aggregate_scores["min"] = min(per_task_scores)
            aggregate_scores["max"] = max(per_task_scores)
            aggregate_scores["std"] = self._std_dev(per_task_scores)

        return aggregate_scores, per_task_scores

    async def _score_output(
        self,
        task: BenchmarkTask,
        output: str,
        request: EvaluationRequest,
    ) -> float:
        """Score a single output using the specified judge."""
        judge_type = request.judge_type

        if judge_type == JudgeType.EXACT_MATCH:
            return self._exact_match(task.expected or "", output)
        elif judge_type == JudgeType.LLM:
            return await self._llm_judge(task, output, request)
        elif judge_type == JudgeType.ROUGE:
            return self._rouge_score(task.expected or "", output)
        elif judge_type == JudgeType.BLEU:
            return self._bleu_score(task.expected or "", output)
        else:
            return 0.0

    def _exact_match(self, expected: str, output: str) -> float:
        """Exact match scoring."""
        return 1.0 if expected.strip().lower() == output.strip().lower() else 0.0

    async def _llm_judge(
        self,
        task: BenchmarkTask,
        output: str,
        request: EvaluationRequest,
    ) -> float:
        """LLM-based judging (placeholder)."""
        # In production, would call an LLM to judge the output
        # For now, return a mock score based on length similarity
        expected = task.expected or ""
        if not expected:
            return 0.5  # Neutral if no expected answer

        len_ratio = min(len(output), len(expected)) / max(len(output), len(expected), 1)
        return len_ratio * 0.8 + random.random() * 0.2

    def _rouge_score(self, expected: str, output: str) -> float:
        """ROUGE score (simplified)."""
        if not expected:
            return 0.0
        expected_words = set(expected.lower().split())
        output_words = set(output.lower().split())
        if not expected_words:
            return 0.0
        overlap = len(expected_words & output_words)
        return overlap / len(expected_words)

    def _bleu_score(self, expected: str, output: str) -> float:
        """BLEU score (simplified unigram precision)."""
        if not expected:
            return 0.0
        expected_words = expected.lower().split()
        output_words = output.lower().split()
        if not output_words:
            return 0.0
        matches = sum(1 for w in output_words if w in expected_words)
        return matches / len(output_words)

    def _std_dev(self, values: list[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _save_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to disk."""
        filepath = settings.results_dir / f"{result.id}.json"
        with open(filepath, "w") as f:
            json.dump(result.model_dump(), f, default=str, indent=2)


# Global instance
benchmark_service = BenchmarkService()