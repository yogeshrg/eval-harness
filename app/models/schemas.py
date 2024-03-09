"""Pydantic models for Eval Harness."""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4


class BenchmarkType(str, Enum):
    """Types of benchmarks."""

    MMLU = "mmlu"
    HUMANEVAL = "humaneval"
    GSM8K = "gsm8k"
    CUSTOM = "custom"


class JudgeType(str, Enum):
    """Types of judges."""

    LLM = "llm"
    EXACT_MATCH = "exact_match"
    ROUGE = "rouge"
    BLEU = "bleu"
    SEMANTIC = "semantic"


class ComparisonType(str, Enum):
    """Types of comparisons."""

    PAIRWISE = "pairwise"
    POINTWISE = "pointwise"
    RANKING = "ranking"


class BenchmarkTask(BaseModel):
    """A single benchmark task."""

    id: str
    prompt: str
    expected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    difficulty: str | None = None


class BenchmarkSuite(BaseModel):
    """A collection of benchmark tasks."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    benchmark_type: BenchmarkType
    tasks: list[BenchmarkTask] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRequest(BaseModel):
    """Request to evaluate model outputs."""

    benchmark_id: UUID
    model_outputs: dict[str, list[str]]  # model_name -> list of outputs
    judge_type: JudgeType = JudgeType.LLM
    judge_config: dict[str, Any] = Field(default_factory=dict)
    compare_with_baseline: bool = False
    baseline_model: str | None = None


class EvaluationResult(BaseModel):
    """Result of an evaluation."""

    id: UUID = Field(default_factory=uuid4)
    benchmark_id: UUID
    model_name: str
    scores: dict[str, float]  # metric_name -> score
    per_task_scores: list[float] | None = None
    judge_type: JudgeType
    judge_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class ComparisonRequest(BaseModel):
    """Request for pairwise comparison."""

    comparison_type: ComparisonType = ComparisonType.PAIRWISE
    model_a: str
    model_b: str
    prompts: list[str]
    judge_type: JudgeType = JudgeType.LLM
    judge_config: dict[str, Any] = Field(default_factory=dict)
    num_judges: int = 3


class ComparisonResult(BaseModel):
    """Result of a comparison."""

    id: UUID = Field(default_factory=uuid4)
    model_a: str
    model_b: str
    comparison_type: ComparisonType
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    win_rate_a: float = 0.0
    win_rate_b: float = 0.0
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    compared_at: datetime = Field(default_factory=datetime.utcnow)


class StatisticalTest(BaseModel):
    """Statistical test configuration."""

    test_type: str = "bootstrap"  # bootstrap, t_test, wilcoxon, permutation
    confidence_level: float = 0.95
    alternative: str = "two-sided"  # two-sided, less, greater
    num_iterations: int = 1000


class StatisticalResult(BaseModel):
    """Result of statistical significance testing."""

    test_type: str
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    significant: bool
    interpretation: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    benchmarks_available: int
    judge_ready: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""

    rank: int
    model_name: str
    score: float
    benchmark_id: UUID
    metadata: dict[str, Any] = Field(default_factory=dict)


class Leaderboard(BaseModel):
    """Leaderboard for a benchmark."""

    benchmark_id: UUID
    benchmark_name: str
    entries: list[LeaderboardEntry]
    generated_at: datetime = Field(default_factory=datetime.utcnow)