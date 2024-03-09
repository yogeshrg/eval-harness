"""Statistical significance testing service."""

import logging
import math
import random
from typing import Any
from uuid import UUID, uuid4

from app.core.config import settings
from app.models.schemas import (
    StatisticalTest,
    StatisticalResult,
    ComparisonResult,
)

logger = logging.getLogger(__name__)


class StatisticsService:
    """Service for statistical significance testing."""

    def __init__(self) -> None:
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the service."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("Statistics service initialized")

    def bootstrap_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run bootstrap test for difference in means."""
        if not self._initialized:
            self.initialize()

        if len(sample_a) < 2 or len(sample_b) < 2:
            raise ValueError("Samples must have at least 2 elements")

        mean_a = sum(sample_a) / len(sample_a)
        mean_b = sum(sample_b) / len(sample_b)
        observed_diff = mean_a - mean_b

        # Bootstrap resampling
        bootstrap_diffs = []
        combined = sample_a + sample_b
        n_a = len(sample_a)

        for _ in range(test.num_iterations):
            # Resample with replacement
            resampled = [random.choice(combined) for _ in combined]
            boot_a = resampled[:n_a]
            boot_b = resampled[n_a:]
            boot_diff = sum(boot_a) / len(boot_a) - sum(boot_b) / len(boot_b)
            bootstrap_diffs.append(boot_diff)

        # Calculate p-value
        if test.alternative == "two-sided":
            extreme = sum(1 for d in bootstrap_diffs if abs(d) >= abs(observed_diff))
        elif test.alternative == "greater":
            extreme = sum(1 for d in bootstrap_diffs if d >= observed_diff)
        else:  # less
            extreme = sum(1 for d in bootstrap_diffs if d <= observed_diff)

        p_value = extreme / test.num_iterations

        # Confidence interval
        alpha = 1 - test.confidence_level
        lower_idx = int(alpha / 2 * test.num_iterations)
        upper_idx = int((1 - alpha / 2) * test.num_iterations)
        sorted_diffs = sorted(bootstrap_diffs)
        ci = (
            sorted_diffs[lower_idx] if lower_idx < len(sorted_diffs) else sorted_diffs[0],
            sorted_diffs[upper_idx] if upper_idx < len(sorted_diffs) else sorted_diffs[-1],
        )

        # Effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((len(sample_a) - 1) * self._variance(sample_a) + (len(sample_b) - 1) * self._variance(sample_b))
            / (len(sample_a) + len(sample_b) - 2)
        )
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        significant = p_value < (1 - test.confidence_level)

        if p_value < 0.001:
            interpretation = "Highly significant"
        elif p_value < 0.01:
            interpretation = "Very significant"
        elif p_value < 0.05:
            interpretation = "Significant"
        elif p_value < 0.1:
            interpretation = "Marginally significant"
        else:
            interpretation = "Not significant"

        return StatisticalResult(
            test_type="bootstrap",
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=significant,
            interpretation=f"{interpretation} (p={p_value:.4f}, d={effect_size:.3f})",
        )

    def t_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run independent samples t-test."""
        if not self._initialized:
            self.initialize()

        if len(sample_a) < 2 or len(sample_b) < 2:
            raise ValueError("Samples must have at least 2 elements")

        mean_a = sum(sample_a) / len(sample_a)
        mean_b = sum(sample_b) / len(sample_b)
        var_a = self._variance(sample_a)
        var_b = self._variance(sample_b)

        # Pooled standard deviation
        pooled_var = (
            (len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b
        ) / (len(sample_a) + len(sample_b) - 2)
        pooled_std = math.sqrt(pooled_var)

        # t-statistic
        diff = mean_a - mean_b
        se = pooled_std * math.sqrt(1 / len(sample_a) + 1 / len(sample_b))
        t_stat = diff / se if se > 0 else 0

        # Degrees of freedom
        df = len(sample_a) + len(sample_b) - 2

        # p-value (two-tailed)
        p_value = 2 * self._t_cdf(-abs(t_stat), df)

        # Confidence interval
        alpha = 1 - test.confidence_level
        t_critical = self._t_ppf(1 - alpha / 2, df)
        margin = t_critical * se
        ci = (diff - margin, diff + margin)

        # Effect size
        effect_size = diff / pooled_std if pooled_std > 0 else 0

        significant = p_value < (1 - test.confidence_level)

        interpretation = "Significant" if p_value < 0.05 else "Not significant"
        if p_value < 0.01:
            interpretation = "Very significant"
        elif p_value < 0.001:
            interpretation = "Highly significant"

        return StatisticalResult(
            test_type="t_test",
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=significant,
            interpretation=f"{interpretation} (p={p_value:.4f}, d={effect_size:.3f})",
        )

    def wilcoxon_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run Wilcoxon signed-rank test (paired)."""
        if not self._initialized:
            self.initialize()

        if len(sample_a) != len(sample_b):
            raise ValueError("Samples must have equal length for paired test")

        # Differences
        diffs = [a - b for a, b in zip(sample_a, sample_b)]
        diffs = [d for d in diffs if d != 0]  # Remove zeros

        if len(diffs) < 3:
            raise ValueError("Need at least 3 non-zero differences")

        # Rank absolute differences
        abs_diffs = [abs(d) for d in diffs]
        sorted_pairs = sorted(zip(abs_diffs, diffs), key=lambda x: x[0])

        # Handle ties in ranking
        ranks = []
        i = 0
        while i < len(sorted_pairs):
            j = i
            while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks.append((sorted_pairs[k][1], avg_rank))
            i = j

        # Sum positive and negative ranks
        pos_sum = sum(r for d, r in ranks if d > 0)
        neg_sum = sum(r for d, r in ranks if d < 0)
        w = min(pos_sum, neg_sum)

        # Normal approximation for large samples
        n = len(diffs)
        mean_w = n * (n + 1) / 4
        std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)

        z = (w - mean_w) / std_w if std_w > 0 else 0

        # Two-tailed p-value
        p_value = 2 * (1 - self._phi(abs(z)))

        significant = p_value < (1 - test.confidence_level)

        return StatisticalResult(
            test_type="wilcoxon",
            p_value=p_value,
            effect_size=0.0,  # Not directly applicable
            confidence_interval=(0.0, 0.0),
            significant=significant,
            interpretation=f"{'Significant' if significant else 'Not significant'} (p={p_value:.4f})",
        )

    def permutation_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run permutation test."""
        if not self._initialized:
            self.initialize()

        combined = sample_a + sample_b
        n_a = len(sample_a)
        observed_diff = sum(sample_a) / n_a - sum(sample_b) / len(sample_b)

        perm_diffs = []
        for _ in range(test.num_iterations):
            random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = sum(perm_a) / n_a - sum(perm_b) / len(perm_b)
            perm_diffs.append(perm_diff)

        if test.alternative == "two-sided":
            extreme = sum(1 for d in perm_diffs if abs(d) >= abs(observed_diff))
        elif test.alternative == "greater":
            extreme = sum(1 for d in perm_diffs if d >= observed_diff)
        else:
            extreme = sum(1 for d in perm_diffs if d <= observed_diff)

        p_value = extreme / test.num_iterations

        alpha = 1 - test.confidence_level
        lower_idx = int(alpha / 2 * test.num_iterations)
        upper_idx = int((1 - alpha / 2) * test.num_iterations)
        sorted_diffs = sorted(perm_diffs)
        ci = (sorted_diffs[lower_idx], sorted_diffs[upper_idx])

        pooled_std = math.sqrt(
            ((len(sample_a) - 1) * self._variance(sample_a) + (len(sample_b) - 1) * self._variance(sample_b))
            / (len(sample_a) + len(sample_b) - 2)
        )
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

        significant = p_value < (1 - test.confidence_level)

        return StatisticalResult(
            test_type="permutation",
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=significant,
            interpretation=f"{'Significant' if significant else 'Not significant'} (p={p_value:.4f})",
        )

    def run_test(
        self,
        sample_a: list[float],
        sample_b: list[float],
        test: StatisticalTest,
    ) -> StatisticalResult:
        """Run the specified statistical test."""
        if test.test_type == "bootstrap":
            return self.bootstrap_test(sample_a, sample_b, test)
        elif test.test_type == "t_test":
            return self.t_test(sample_a, sample_b, test)
        elif test.test_type == "wilcoxon":
            return self.wilcoxon_test(sample_a, sample_b, test)
        elif test.test_type == "permutation":
            return self.permutation_test(sample_a, sample_b, test)
        else:
            raise ValueError(f"Unknown test type: {test.test_type}")

    def compare_results(self, result_a: ComparisonResult, result_b: ComparisonResult) -> StatisticalResult:
        """Compare two comparison results."""
        # Create synthetic samples from win rates
        n = result_a.wins_a + result_a.wins_b + result_a.ties
        if n == 0:
            return StatisticalResult(
                test_type="comparison",
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                interpretation="No data to compare",
            )

        sample_a = [1] * result_a.wins_a + [0] * result_a.wins_b + [0.5] * result_a.ties
        sample_b = [1] * result_b.wins_a + [0] * result_b.wins_b + [0.5] * result_b.ties

        test = StatisticalTest(test_type="t_test", confidence_level=0.95)
        return self.t_test(sample_a, sample_b, test)

    # Helper methods
    def _variance(self, sample: list[float]) -> float:
        if len(sample) < 2:
            return 0.0
        mean = sum(sample) / len(sample)
        return sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)

    def _phi(self, x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + self._erf(x / math.sqrt(2)))

    def _erf(self, x: float) -> float:
        """Error function."""
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

    def _t_cdf(self, t: float, df: int) -> float:
        """t-distribution CDF approximation."""
        # Simple approximation for two-tailed
        if df > 30:
            return self._phi(t)
        # Use beta function approximation for small df
        x = df / (df + t * t)
        return 0.5 * (1 + math.copysign(1, t) * (1 - x ** (df / 2)))

    def _t_ppf(self, p: float, df: int) -> float:
        """t-distribution percent point function (inverse CDF)."""
        # Simple approximation
        if df > 30:
            return self._norm_ppf(p)
        # Wilson-Hilferty approximation
        return self._norm_ppf(p) * (1 + 1 / (4 * df))

    def _norm_ppf(self, p: float) -> float:
        """Normal percent point function."""
        # Rational approximation
        if p <= 0 or p >= 1:
            return 0
        q = p - 0.5
        r = q * q
        if abs(q) <= 0.425:
            r = q * q
            return q * (((((2.5066282746310002 * r + 18.61500062529271) * r + 41.39119773536445) * r + 25.44106049637372) * r + 7.802316052940563) * r + 0.9212431255670621) / \
                   ((((r + 6.38294727477829) * r + 21.05995884748129) * r + 17.73930742833831) * r + 7.858359150076201)
        r = math.sqrt(-2 * math.log(1 - p) if p > 0.5 else -2 * math.log(p))
        return -(((r - 1.64538004) * r - 4.71796937) * r + 2.1230145) / \
               ((r - 4.14456667) * r + 4.75548673) if q > 0 else \
               (((r - 1.64538004) * r - 4.71796937) * r + 2.1230145) / \
               ((r - 4.14456667) * r + 4.75548673)


# Global instance
statistics_service = StatisticsService()