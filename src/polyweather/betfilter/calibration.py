"""Make stated probabilities mean what they say.

A raw bucket probability derived from the model's interval is not a
calibrated probability. Measured on held-out history, markets this system
labelled 64% actually settled 51% of the time. Every gate in
``decision.py`` is expressed as a probability threshold, so an
overconfident probability silently moves every gate: asking for 63% while
receiving 51% is not a conservative filter, it is a broken one.

This module fits the correction on resolved history only, and refuses to
apply a correction it does not have enough evidence to support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # pragma: no cover - sklearn ships with the project.
    IsotonicRegression = None  # type: ignore[assignment,misc]

# Isotonic regression is nonparametric and will happily carve a step
# function out of noise. Below this many resolved markets the honest answer
# is to leave the probability uncorrected and say so.
MIN_CALIBRATION_ROWS = 300
# Half-width of the neighbourhood used to judge how much evidence supports a
# given point on the probability scale, and the prior strength the local
# count is weighed against.
LOCAL_SUPPORT_WINDOW = 0.05
LOCAL_SUPPORT_PRIOR = 60


@dataclass(frozen=True)
class ProbabilityCalibrator:
    """Monotone mapping from stated probability to realized frequency."""

    knots_x: tuple[float, ...]
    knots_y: tuple[float, ...]
    rows: int
    fitted_through: str | None = None

    def apply(self, probability: float) -> float:
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be finite and in [0, 1].")
        if probability == 0.0:
            return 0.0
        if not self.knots_x:
            return probability
        corrected = float(np.clip(np.interp(probability, self.knots_x, self.knots_y), 0.0, 1.0))
        # `np.interp` clamps outside the fitted range, so a probability above
        # anything the calibrator ever saw picks up the last knot's value.
        # When that knot sits above the input, the clamp silently *raises*
        # confidence in exactly the region with no evidence behind it. Beyond
        # the fitted range the correction may only ever be conservative.
        if probability > self.knots_x[-1] or probability < self.knots_x[0]:
            return min(corrected, probability)
        return corrected

    def to_dict(self) -> dict[str, object]:
        return {
            "knotsX": list(self.knots_x),
            "knotsY": list(self.knots_y),
            "rows": self.rows,
            "fittedThrough": self.fitted_through,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProbabilityCalibrator":
        return cls(
            knots_x=tuple(float(value) for value in payload.get("knotsX", [])),  # type: ignore[union-attr]
            knots_y=tuple(float(value) for value in payload.get("knotsY", [])),  # type: ignore[union-attr]
            rows=int(payload.get("rows", 0)),  # type: ignore[arg-type]
            fitted_through=payload.get("fittedThrough"),  # type: ignore[arg-type]
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


IDENTITY = ProbabilityCalibrator(knots_x=(), knots_y=(), rows=0)


def fit(
    stated_probability: np.ndarray,
    settled: np.ndarray,
    fitted_through: str | None = None,
    minimum_rows: int = MIN_CALIBRATION_ROWS,
) -> ProbabilityCalibrator:
    """Fit an isotonic correction from resolved markets.

    ``stated_probability`` and ``settled`` must come from markets that
    resolved strictly before the period this calibrator will be applied to.
    Fitting on the same rows the calibrator is later scored against turns a
    measured correction into a self-fulfilling one.

    Isotonic (monotone) rather than a free-form fit: a market the model
    thinks is more likely must never come out less likely after correction,
    or the ordering the gates rely on stops meaning anything.
    """
    probabilities = np.asarray(stated_probability, dtype=float)
    outcomes = np.asarray(settled, dtype=float)
    if probabilities.ndim != 1 or probabilities.shape != outcomes.shape:
        raise ValueError("Calibration requires equally sized one-dimensional arrays.")
    mask = np.isfinite(probabilities) & np.isfinite(outcomes)
    probabilities, outcomes = probabilities[mask], outcomes[mask]
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.isin(outcomes, [0, 1]).all():
        raise ValueError("Calibration requires probabilities in [0, 1] and binary outcomes.")
    if probabilities.size < minimum_rows or IsotonicRegression is None:
        return IDENTITY
    if len(np.unique(outcomes)) < 2:
        return IDENTITY
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(probabilities, outcomes)
    grid = np.linspace(float(probabilities.min()), float(probabilities.max()), 50)
    fitted = model.predict(grid)
    # Isotonic regression is nonparametric and the top of the probability
    # range is always the thinnest part of the sample. Left alone it will
    # happily map a stated 0.75 -- supported by a handful of markets that
    # all happened to hit -- onto a claimed certainty of 1.0. That is the
    # one direction this correction must never move: the whole point is to
    # remove overconfidence, not manufacture it. Where the local sample is
    # thin, shrink back toward the uncorrected value in proportion to how
    # little evidence sits nearby.
    corrected = []
    for point, value in zip(grid, fitted, strict=True):
        local = int(np.sum(np.abs(probabilities - point) <= LOCAL_SUPPORT_WINDOW))
        trust = local / (local + LOCAL_SUPPORT_PRIOR)
        corrected.append(float(point + (value - point) * trust))
    return ProbabilityCalibrator(
        knots_x=tuple(float(value) for value in grid),
        # Shrinking each point independently can break monotonicity, and the
        # gates rely on ordering being preserved.
        knots_y=tuple(float(value) for value in np.maximum.accumulate(corrected)),
        rows=int(probabilities.size),
        fitted_through=fitted_through,
    )


def reliability_gap(stated_probability: np.ndarray, settled: np.ndarray) -> float:
    """Mean signed error of the probabilities. Negative means overconfident."""
    probabilities = np.asarray(stated_probability, dtype=float)
    outcomes = np.asarray(settled, dtype=float)
    mask = np.isfinite(probabilities) & np.isfinite(outcomes)
    if not mask.any():
        return float("nan")
    return float(np.mean(outcomes[mask] - probabilities[mask]))
