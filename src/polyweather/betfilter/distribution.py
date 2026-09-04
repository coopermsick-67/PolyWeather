"""Turn a point forecast plus calibrated uncertainty into bucket probabilities.

The forecasting model answers "what is the high?" with a number and an
interval. A 2F market asks a different question: "what is the chance the
settled high lands in exactly this bucket?" This module is the bridge, and
it is deliberately the only place that knows about settlement rounding.

Two backends, in order of preference:

1. ``empirical`` -- shift the station's own historical residual distribution
   around the point forecast. This inherits real skewness and fat tails
   instead of assuming them away, and is used whenever enough residuals for
   that station are available.
2. ``split_normal`` -- a two-piece normal matched to the model's calibrated
   asymmetric interval. Used when residual history is too thin to trust.

Never assume a symmetric normal from a half-width alone: the residual model
has genuinely asymmetric bounds and collapsing them loses the asymmetry that
matters most near a bucket edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this many station residuals the empirical distribution is too lumpy
# to assign 2F-bucket probabilities; fall back to the calibrated parametric
# form rather than reporting a spuriously precise number.
MIN_EMPIRICAL_RESIDUALS = 120


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _probit(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("Probit requires a probability strictly inside (0, 1).")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def settlement_window(lower_f: int, upper_f: int, rounding: str = "nearest") -> tuple[float, float]:
    """Continuous temperature window that settles into an integer bucket.

    With ``nearest`` rounding a 94-95 bucket is won by any true high in
    [93.5, 95.5) -- a full degree wider than the naive [94, 95] read, and
    the difference is decisive for a 2F market. ``truncate`` models a
    floor-based settlement rule; ``exact`` treats the printed bounds as the
    literal continuous interval.
    """
    if upper_f < lower_f:
        raise ValueError("Bucket upper bound must not be below its lower bound.")
    if rounding == "nearest":
        return (lower_f - 0.5, upper_f + 0.5)
    if rounding == "truncate":
        return (float(lower_f), float(upper_f) + 1.0)
    if rounding == "exact":
        return (float(lower_f), float(upper_f))
    raise ValueError(f"Unknown settlement rounding rule: {rounding}")


@dataclass(frozen=True)
class BucketProbability:
    lower_f: int
    upper_f: int
    probability: float
    window_lower_f: float
    window_upper_f: float
    distance_to_lower_edge_f: float
    distance_to_upper_edge_f: float
    minimum_edge_distance_f: float
    normalized_boundary_safety: float

    @property
    def label(self) -> str:
        return f"{self.lower_f}-{self.upper_f}"


@dataclass(frozen=True)
class TemperatureDistribution:
    """A calibrated predictive distribution for one station-day."""

    expected_high_f: float
    sigma_f: float
    sigma_lower_f: float
    sigma_upper_f: float
    method: str
    sample_size: int
    residuals: np.ndarray | None = None
    # A same-day high that has already been observed truncates the
    # distribution from below: the final high cannot come in under it.
    observed_high_floor_f: float | None = None

    def cdf(self, value: float) -> float:
        floor = self.observed_high_floor_f
        if floor is not None and value < floor:
            return 0.0
        raw = self._raw_cdf(value)
        if floor is None:
            return raw
        # Renormalize over the surviving support rather than reporting mass
        # that today's observations have already ruled out.
        floor_mass = self._raw_cdf(floor)
        if floor_mass >= 1.0 - 1e-12:
            return 1.0
        return (raw - floor_mass) / (1.0 - floor_mass)

    def _raw_cdf(self, value: float) -> float:
        if self.residuals is not None and self.residuals.size:
            return float(np.mean(self.residuals + self.expected_high_f <= value))
        s1, s2 = self.sigma_lower_f, self.sigma_upper_f
        total = s1 + s2
        if total <= 0:
            return 1.0 if value >= self.expected_high_f else 0.0
        if value < self.expected_high_f:
            return (2.0 * s1 / total) * _phi((value - self.expected_high_f) / s1)
        return s1 / total + (2.0 * s2 / total) * (_phi((value - self.expected_high_f) / s2) - 0.5)

    def probability_between(self, lower: float, upper: float) -> float:
        return max(0.0, self.cdf(upper) - self.cdf(lower))

    def bucket_probability(
        self, lower_f: int, upper_f: int, rounding: str = "nearest"
    ) -> BucketProbability:
        window_lower, window_upper = settlement_window(lower_f, upper_f, rounding)
        probability = self.probability_between(window_lower, window_upper)
        to_lower = self.expected_high_f - window_lower
        to_upper = window_upper - self.expected_high_f
        minimum_edge = min(to_lower, to_upper)
        # Distance is only meaningful relative to this forecast's own spread.
        normalized = minimum_edge / self.sigma_f if self.sigma_f > 0 else 0.0
        return BucketProbability(
            lower_f=lower_f,
            upper_f=upper_f,
            probability=probability,
            window_lower_f=window_lower,
            window_upper_f=window_upper,
            distance_to_lower_edge_f=to_lower,
            distance_to_upper_edge_f=to_upper,
            minimum_edge_distance_f=minimum_edge,
            normalized_boundary_safety=normalized,
        )

    def quantile(self, probability: float) -> float:
        if self.residuals is not None and self.residuals.size:
            return float(self.expected_high_f + np.quantile(self.residuals, probability))
        s1, s2 = self.sigma_lower_f, self.sigma_upper_f
        total = s1 + s2
        split = s1 / total if total > 0 else 0.5
        if probability < split:
            return self.expected_high_f + s1 * _probit(probability * total / (2.0 * s1))
        return self.expected_high_f + s2 * _probit(
            0.5 + (probability - split) * total / (2.0 * s2)
        )


def from_calibrated_interval(
    expected_high_f: float,
    interval_lower_f: float,
    interval_upper_f: float,
    nominal_coverage: float,
    observed_high_floor_f: float | None = None,
) -> TemperatureDistribution:
    """Fit a two-piece normal that reproduces the model's own interval.

    Matching each side separately preserves the asymmetry the conformal
    calibration measured. The resulting split-normal reproduces exactly
    ``nominal_coverage`` between the two endpoints regardless of how
    asymmetric they are.
    """
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("Nominal coverage must be strictly between 0 and 1.")
    if not (math.isfinite(expected_high_f) and math.isfinite(interval_lower_f) and math.isfinite(interval_upper_f)):
        raise ValueError("Cannot build a distribution from non-finite forecast inputs.")
    z = _probit(0.5 + nominal_coverage / 2.0)
    sigma_lower = max((expected_high_f - interval_lower_f) / z, 1e-6)
    sigma_upper = max((interval_upper_f - expected_high_f) / z, 1e-6)
    return TemperatureDistribution(
        expected_high_f=expected_high_f,
        # A single summary scale for boundary normalization: the average of
        # the two sides, so neither tail alone drives the safety metric.
        sigma_f=(sigma_lower + sigma_upper) / 2.0,
        sigma_lower_f=sigma_lower,
        sigma_upper_f=sigma_upper,
        method="split_normal",
        sample_size=0,
        observed_high_floor_f=observed_high_floor_f,
    )


def from_residual_history(
    expected_high_f: float,
    residuals: np.ndarray,
    observed_high_floor_f: float | None = None,
) -> TemperatureDistribution:
    """Use a station's own realized errors as the predictive distribution.

    ``residuals`` must be ``actual - predicted`` from forecasts that were
    made strictly before the day being predicted. Passing residuals that
    include the target day is leakage and will silently overstate every
    probability this module reports.
    """
    clean = np.asarray(residuals, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < MIN_EMPIRICAL_RESIDUALS:
        raise ValueError(
            f"Empirical distribution needs at least {MIN_EMPIRICAL_RESIDUALS} residuals; got {clean.size}."
        )
    sigma = float(np.std(clean, ddof=1))
    lower_spread = float(np.abs(np.quantile(clean, 0.10)))
    upper_spread = float(np.abs(np.quantile(clean, 0.90)))
    return TemperatureDistribution(
        expected_high_f=expected_high_f,
        sigma_f=max(sigma, 1e-6),
        sigma_lower_f=max(lower_spread / 1.2816, 1e-6),
        sigma_upper_f=max(upper_spread / 1.2816, 1e-6),
        method="empirical",
        sample_size=int(clean.size),
        residuals=clean,
        observed_high_floor_f=observed_high_floor_f,
    )


def candidate_buckets(
    distribution: TemperatureDistribution,
    bucket_width_f: int = 2,
    span_f: int = 8,
    rounding: str = "nearest",
    anchor_f: int | None = None,
) -> list[BucketProbability]:
    """Enumerate the market buckets around the forecast, most likely first.

    ``anchor_f`` pins the bucket grid to the market's actual offsets; a
    94-95 board and a 93-94 board are different grids and scoring the wrong
    one produces a confident answer to a question nobody asked.
    """
    if bucket_width_f < 1:
        raise ValueError("Bucket width must be at least one degree.")
    center = int(round(distribution.expected_high_f))
    anchor = center if anchor_f is None else anchor_f
    # Align the grid so `anchor` is the lower edge of one of the buckets.
    offset = (center - anchor) % bucket_width_f
    start = center - offset - span_f
    buckets = [
        distribution.bucket_probability(lower, lower + bucket_width_f - 1, rounding)
        for lower in range(start, center + span_f + 1, bucket_width_f)
    ]
    return sorted(buckets, key=lambda bucket: bucket.probability, reverse=True)


def probability_gap(buckets: list[BucketProbability]) -> float:
    """Margin between the best and second-best bucket.

    A 56/39 split and a 45/43 split can both name the same favorite while
    describing completely different levels of evidence.
    """
    if len(buckets) < 2:
        return 0.0
    return float(buckets[0].probability - buckets[1].probability)


def normalized_entropy(buckets: list[BucketProbability]) -> float:
    """0 = all mass in one bucket, 1 = spread evenly. Lower is better."""
    probabilities = np.asarray([bucket.probability for bucket in buckets], dtype=float)
    probabilities = probabilities[probabilities > 0]
    if probabilities.size < 2:
        return 0.0
    probabilities = probabilities / probabilities.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(probabilities.size)
