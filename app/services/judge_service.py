"""LLM Judge service for pairwise comparisons and rankings."""

import logging
import math
import random
from typing import Any
from uuid import UUID, uuid4

from app.core.config import settings
from app.models.schemas import (
    ComparisonRequest,
    ComparisonResult,
    ComparisonType,
    JudgeType,
)

logger = logging.getLogger(__name__)


class JudgeService:
    """Service for LLM-based pairwise comparisons."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("Judge service initialized")

    async def compare(self, request: ComparisonRequest) -> ComparisonResult:
        """Run pairwise comparison between two models."""
        if not self._initialized:
            self.initialize()

        if request.comparison_type == ComparisonType.PAIRWISE:
            return await self._pairwise_compare(request)
        elif request.comparison_type == ComparisonType.RANKING:
            return await self._ranking_compare(request)
        else:
            raise ValueError(f"Unsupported comparison type: {request.comparison_type}")

    async def _pairwise_compare(self, request: ComparisonRequest) -> ComparisonResult:
        """Run pairwise comparison."""
        wins_a = 0
        wins_b = 0
        ties = 0

        num_judges = request.num_judges

        for prompt in request.prompts:
            # Simulate multiple judges
            judge_votes = {"a": 0, "b": 0, "tie": 0}

            for _ in range(num_judges):
                vote = await self._single_judge(
                    request.model_a,
                    request.model_b,
                    prompt,
                    request.judge_type,
                    request.judge_config,
                )
                judge_votes[vote] += 1

            # Determine winner for this prompt
            if judge_votes["a"] > judge_votes["b"]:
                wins_a += 1
            elif judge_votes["b"] > judge_votes["a"]:
                wins_b += 1
            else:
                ties += 1

        total = wins_a + wins_b + ties
        win_rate_a = wins_a / total if total > 0 else 0.0
        win_rate_b = wins_b / total if total > 0 else 0.0

        # Statistical significance (placeholder)
        p_value = self._calculate_p_value(wins_a, wins_b, total)
        ci = self._wilson_ci(wins_a, total) if total > 0 else (0.0, 0.0)

        return ComparisonResult(
            model_a=request.model_a,
            model_b=request.model_b,
            comparison_type=ComparisonType.PAIRWISE,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            win_rate_a=win_rate_a,
            win_rate_b=win_rate_b,
            p_value=p_value,
            confidence_interval=ci,
            metadata={"num_judges": num_judges, "num_prompts": len(request.prompts)},
        )

    async def _ranking_compare(self, request: ComparisonRequest) -> ComparisonResult:
        """Run ranking comparison (placeholder)."""
        # For ranking, we'd need more than 2 models
        return await self._pairwise_compare(request)

    async def _single_judge(
        self,
        model_a: str,
        model_b: str,
        prompt: str,
        judge_type: JudgeType,
        judge_config: dict[str, Any],
    ) -> str:
        """Run a single judge comparison (placeholder)."""
        # In production, would call an LLM to judge
        # For now, random vote with slight bias
        r = random.random()
        if r < 0.45:
            return "a"
        elif r < 0.9:
            return "b"
        else:
            return "tie"

    def _calculate_p_value(self, wins_a: int, wins_b: int, total: int) -> float:
        """Calculate p-value for binomial test."""
        # Placeholder: simplified binomial test
        if total == 0:
            return 1.0
        # Use normal approximation
        p = 0.5
        expected = total * p
        variance = total * p * (1 - p)
        if variance == 0:
            return 1.0
        z = (wins_a - expected) / (variance ** 0.5)
        # Two-tailed p-value
        return 2 * (1 - self._phi(abs(z)))

    def _phi(self, z: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + self._erf(z / (2 ** 0.5)))

    def _erf(self, x: float) -> float:
        """Error function approximation."""
        # Abramowitz and Stegun approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return sign * y

    def _wilson_ci(self, wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval for binomial proportion."""
        if total == 0:
            return (0.0, 0.0)
        p = wins / total
        denominator = 1 + z * z / total
        centre = (p + z * z / (2 * total)) / denominator
        half = z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5 / denominator
        return (max(0, centre - half), min(1, centre + half))


# Global instance
judge_service = JudgeService()