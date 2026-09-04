"""Cross-source agreement analysis.

Averaging four guidance sources into one number throws away the single most
useful thing they collectively know: whether they agree. Four models within
0.8F and four models spanning 3F can produce the same mean and completely
different bet quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Median absolute deviations from the median before a source is called an
# outlier. 3.0 is the conventional robust-statistics cutoff.
OUTLIER_MAD_THRESHOLD = 3.0


@dataclass(frozen=True)
class EnsembleAnalysis:
    sources: dict[str, float]
    mean_f: float
    median_f: float
    minimum_f: float
    maximum_f: float
    spread_f: float
    weighted_mean_f: float
    standard_deviation_f: float
    agreement_score: float
    outliers: list[str] = field(default_factory=list)

    @property
    def source_count(self) -> int:
        return len(self.sources)


def analyze(
    sources: dict[str, float | None],
    weights: dict[str, float] | None = None,
    reference_spread_f: float = 4.0,
) -> EnsembleAnalysis:
    """Summarize guidance agreement for one station-day.

    ``weights`` should come from each source's measured historical error at
    this station and horizon. Absent that evidence every source is weighted
    equally -- an equal weight is an admission of ignorance, not a claim
    that the sources are equally good.
    """
    usable = {
        name: float(value)
        for name, value in sources.items()
        if value is not None and np.isfinite(value)
    }
    if not usable:
        return EnsembleAnalysis(
            sources={}, mean_f=float("nan"), median_f=float("nan"),
            minimum_f=float("nan"), maximum_f=float("nan"), spread_f=float("nan"),
            weighted_mean_f=float("nan"), standard_deviation_f=float("nan"),
            agreement_score=0.0, outliers=[],
        )
    values = np.asarray(list(usable.values()), dtype=float)
    names = list(usable.keys())
    median = float(np.median(values))
    spread = float(values.max() - values.min())
    if weights:
        applied = np.asarray([max(0.0, float(weights.get(name, 0.0))) for name in names], dtype=float)
        weighted = float(np.average(values, weights=applied)) if applied.sum() > 0 else float(values.mean())
    else:
        weighted = float(values.mean())
    # Median absolute deviation flags a single divergent source without
    # letting it drag the consensus, which a plain mean would.
    deviations = np.abs(values - median)
    mad = float(np.median(deviations))
    outliers: list[str] = []
    if mad > 1e-9 and values.size >= 3:
        outliers = [
            name for name, deviation in zip(names, deviations, strict=True)
            if deviation / (1.4826 * mad) > OUTLIER_MAD_THRESHOLD
        ]
    # Agreement decays linearly with spread and is floored at zero, so a
    # single wild source cannot produce a negative score that later
    # arithmetic would treat as a bonus.
    agreement = max(0.0, 1.0 - spread / reference_spread_f) if values.size >= 2 else 0.0
    return EnsembleAnalysis(
        sources=usable,
        mean_f=float(values.mean()),
        median_f=median,
        minimum_f=float(values.min()),
        maximum_f=float(values.max()),
        spread_f=spread,
        weighted_mean_f=weighted,
        standard_deviation_f=float(values.std(ddof=1)) if values.size >= 2 else 0.0,
        agreement_score=agreement,
        outliers=outliers,
    )


def convergence(spread_history_f: list[float]) -> float:
    """Positive when sources are converging over successive runs.

    Models tightening toward each other through the morning is genuine
    evidence; models fanning out is a warning that the earlier agreement was
    coincidence.
    """
    finite = [value for value in spread_history_f if value is not None and np.isfinite(value)]
    if len(finite) < 2:
        return 0.0
    return float(finite[0] - finite[-1])
