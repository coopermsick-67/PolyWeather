"""Machine-readable reason codes for every decision the filter makes.

Reasons are emitted for approvals as well as rejections. A recommendation
that cannot say why it is a recommendation is a confidence badge, not a
decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["critical", "high", "medium", "low", "positive"]


@dataclass(frozen=True)
class Reason:
    code: str
    severity: Severity
    message: str
    value: float | None = None
    threshold: float | None = None

    @property
    def is_blocking(self) -> bool:
        return self.severity == "critical"

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "value": None if self.value is None else round(float(self.value), 4),
            "threshold": None if self.threshold is None else round(float(self.threshold), 4),
        }


REJECTION_CODES = (
    "LOW_RANGE_PROBABILITY",
    "LOW_PROBABILITY_GAP",
    "HIGH_BOUNDARY_RISK",
    "MODEL_DISAGREEMENT",
    "HIGH_ENSEMBLE_SPREAD",
    "FORECAST_UNSTABLE",
    "PREDICTION_TOO_VOLATILE",
    "BUCKET_FLIP_RISK",
    "LOW_STATION_RELIABILITY",
    "INSUFFICIENT_HISTORY",
    "OBSERVATIONS_RUNNING_HOT",
    "OBSERVATIONS_RUNNING_COLD",
    "BUCKET_ALREADY_IMPOSSIBLE",
    "STALE_DATA",
    "MISSING_CRITICAL_DATA",
    "INSUFFICIENT_DATA_SOURCES",
    "LOW_FEATURE_COMPLETENESS",
    "UNSUPPORTED_HORIZON",
    "SETTLEMENT_STATION_UNVERIFIED",
    "UNCALIBRATED_STATION",
    "BELOW_QUALITY_THRESHOLD",
)

POSITIVE_CODES = (
    "HIGH_ENSEMBLE_AGREEMENT",
    "LOW_ENSEMBLE_SPREAD",
    "STABLE_FORECAST",
    "SAFE_BUCKET_POSITION",
    "STRONG_STATION_HISTORY",
    "OBSERVATIONS_ALIGNED",
    "CONCENTRATED_DISTRIBUTION",
)


@dataclass
class ReasonLog:
    """Collects reasons while the gates and scorer run."""

    reasons: list[Reason] = field(default_factory=list)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        value: float | None = None,
        threshold: float | None = None,
    ) -> None:
        self.reasons.append(Reason(code, severity, message, value, threshold))

    @property
    def blocking(self) -> list[Reason]:
        return [reason for reason in self.reasons if reason.is_blocking]

    @property
    def negative(self) -> list[Reason]:
        return [reason for reason in self.reasons if reason.severity != "positive"]

    @property
    def positive(self) -> list[Reason]:
        return [reason for reason in self.reasons if reason.severity == "positive"]

    def to_list(self) -> list[dict[str, object]]:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "positive": 4}
        return [reason.to_dict() for reason in sorted(self.reasons, key=lambda item: order[item.severity])]
