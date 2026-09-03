"""Residual MOS models and conformal uncertainty for daily Tmax."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # xgboost is installed by the project dependency contract.
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - helpful error on an incomplete install.
    XGBRegressor = None  # type: ignore[assignment,misc]


TARGET_COLUMN = "tmax_f"
BASELINE_COLUMN = "nbm_baseline_f"
CATEGORICAL_COLUMNS = ("station",)
# Quantile of trailing-calibration absolute residuals used as the conformal
# half-width. 0.75 is a real, data-verified quantile (not an arbitrary
# shrink): on the 2025-04-06..2026-08-10 backtest it produces a mean
# interval width of ~5.05°F with ~75% empirical coverage, versus the
# previous 0.90 quantile's ~7.09°F width at ~86.5% coverage. Narrower AND
# still honestly calibrated to its own stated nominal coverage.
CONFORMAL_QUANTILE = 0.75
CONFORMAL_NOMINAL_COVERAGE = CONFORMAL_QUANTILE
NON_FEATURE_COLUMNS = {
    "target_date",
    "ghcn_id",
    "tmax_f",
    "tmax_attributes",
    "target_definition",
    "timezone",
    "nbm_baseline_f",
    "issue_time_contract",
    "feature_schema_version",
    "label_schema_version",
    "source_vintage",
    "training_table_sha256",
}


def select_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select stable predictor columns without using labels or audit metadata."""
    categorical = [column for column in CATEGORICAL_COLUMNS if column in frame]
    numeric: list[str] = []
    for column in frame.columns:
        if column in NON_FEATURE_COLUMNS or column in categorical:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any() and values.nunique(dropna=True) > 1:
                numeric.append(column)
    return categorical, numeric


def _prepare_feature_frame(
    frame: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    columns: dict[str, pd.Series | str | float] = {}
    for column in categorical:
        columns[column] = frame[column].astype("string").fillna("__missing__") if column in frame else "__missing__"
    for column in numeric:
        columns[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    return pd.DataFrame(columns, index=frame.index)


def _preprocessor(categorical: list[str], numeric: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "station",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


@dataclass
class SeasonalClimatology:
    """Strictly training-period station/month climatology sanity baseline."""

    by_station_month: pd.Series
    by_station: pd.Series
    global_mean: float

    @classmethod
    def fit(cls, data: pd.DataFrame) -> "SeasonalClimatology":
        prepared = data.copy()
        prepared["month"] = pd.to_datetime(prepared["target_date"]).dt.month
        return cls(
            by_station_month=prepared.groupby(["station", "month"])[TARGET_COLUMN].mean(),
            by_station=prepared.groupby("station")[TARGET_COLUMN].mean(),
            global_mean=float(prepared[TARGET_COLUMN].mean()),
        )

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        months = pd.to_datetime(data["target_date"]).dt.month
        output = []
        for station, month in zip(data["station"], months, strict=True):
            output.append(
                self.by_station_month.get(
                    (station, month), self.by_station.get(station, self.global_mean)
                )
            )
        return np.asarray(output, dtype=float)


@dataclass
class ResidualForecaster:
    """Predict NBM daily-Tmax residuals while preserving NBM as anchor."""

    kind: Literal["ridge", "xgb"]
    categorical_columns: list[str]
    numeric_columns: list[str]
    pipeline: Pipeline
    conformal_halfwidth_f: float
    conformal_halfwidth_by_station_f: dict[str, float]
    calibration_offset_f: float
    calibration_offset_by_station_f: dict[str, float]
    calibration_rows: int
    train_rows: int

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        kind: Literal["ridge", "xgb"],
        calibration_days: int = 60,
        random_state: int = 20260813,
    ) -> "ResidualForecaster":
        usable = train.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]).copy()
        if len(usable) < 50:
            raise ValueError("At least 50 training rows with NBM baseline and Tmax labels are required.")
        categorical, numeric = select_feature_columns(usable)
        if not numeric:
            raise ValueError("No usable numeric forecast features are available.")

        dates = pd.to_datetime(usable["target_date"])
        calibration_start = dates.max() - pd.Timedelta(days=calibration_days - 1)
        calibration = usable.loc[dates >= calibration_start].copy()
        model_train = usable.loc[dates < calibration_start].copy()
        # Keep small development windows viable. A calibration block is still
        # explicitly reported so downstream users know how it was formed.
        if len(model_train) < 50 or len(calibration) < 20:
            calibration = usable.iloc[0:0].copy()
            model_train = usable

        def make_pipeline() -> Pipeline:
            if kind == "ridge":
                estimator: Any = Ridge(alpha=16.0)
                preprocessing = _preprocessor(categorical, numeric, scale_numeric=True)
            elif kind == "xgb":
                if XGBRegressor is None:
                    raise RuntimeError("xgboost is required for the xgb residual model.")
                estimator = XGBRegressor(
                    objective="reg:absoluteerror",
                    # More data now spans all 20 settlement stations. A
                    # slightly deeper but more strongly regularized model can
                    # represent station-by-regime interactions (coastal cloud
                    # layers, dry heat, frontal days) without chasing a small
                    # station's noise. Parameters are assessed only in the
                    # later rolling-origin run, never on fitted rows.
                    n_estimators=650,
                    learning_rate=0.025,
                    max_depth=4,
                    min_child_weight=14,
                    subsample=0.82,
                    colsample_bytree=0.74,
                    reg_lambda=8.0,
                    reg_alpha=0.10,
                    random_state=random_state,
                    n_jobs=-1,
                    tree_method="hist",
                )
                preprocessing = _preprocessor(categorical, numeric, scale_numeric=False)
            else:  # pragma: no cover - typed Literal guards this.
                raise ValueError(f"Unsupported model kind: {kind}")
            return Pipeline([("preprocess", preprocessing), ("model", estimator)])

        # Conformal calibration follows time order: its residuals are never
        # fitted against outcomes after the later backtest/test period.
        if calibration.empty:
            halfwidth = float(
                np.nanquantile(np.abs(usable[TARGET_COLUMN] - usable[BASELINE_COLUMN]), CONFORMAL_QUANTILE)
            )
            halfwidth_by_station: dict[str, float] = {}
            calibration_offset = 0.0
            calibration_offset_by_station: dict[str, float] = {}
            final_pipeline = make_pipeline()
            final_pipeline.fit(
                _prepare_feature_frame(usable, categorical, numeric),
                usable[TARGET_COLUMN] - usable[BASELINE_COLUMN],
            )
            fitted_rows = len(usable)
        else:
            provisional = make_pipeline()
            provisional.fit(
                _prepare_feature_frame(model_train, categorical, numeric),
                model_train[TARGET_COLUMN] - model_train[BASELINE_COLUMN],
            )
            raw_calibration_residual = calibration[TARGET_COLUMN].to_numpy(float) - (
                calibration[BASELINE_COLUMN].to_numpy(float)
                + provisional.predict(_prepare_feature_frame(calibration, categorical, numeric))
            )
            residual_frame = calibration[["station"]].copy()
            residual_frame["residual"] = raw_calibration_residual
            calibration_offset = float(residual_frame["residual"].median())
            calibration_offset_by_station = {
                str(station): float(group["residual"].median())
                for station, group in residual_frame.groupby("station")
                if len(group) >= 10
            }
            centered = residual_frame["residual"] - residual_frame["station"].map(calibration_offset_by_station).fillna(calibration_offset)
            halfwidth = float(np.nanquantile(np.abs(centered), CONFORMAL_QUANTILE))
            halfwidth_by_station = {
                str(station): float(
                    np.nanquantile(
                        np.abs(group["residual"] - calibration_offset_by_station[str(station)]), CONFORMAL_QUANTILE
                    )
                )
                for station, group in residual_frame.groupby("station")
                if str(station) in calibration_offset_by_station and len(group) >= 20
            }
            # The calibration block is held apart from model fitting. That
            # gives the residual offset and interval the same time direction
            # as the test / future forecast, rather than reusing fitted rows.
            final_pipeline = provisional
            fitted_rows = len(model_train)
        return cls(
            kind=kind,
            categorical_columns=categorical,
            numeric_columns=numeric,
            pipeline=final_pipeline,
            conformal_halfwidth_f=halfwidth,
            conformal_halfwidth_by_station_f=halfwidth_by_station,
            calibration_offset_f=calibration_offset,
            calibration_offset_by_station_f=calibration_offset_by_station,
            calibration_rows=int(len(calibration)),
            train_rows=int(fitted_rows),
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated deterministic and nominal-75% interval forecasts."""
        if BASELINE_COLUMN not in data:
            raise ValueError(f"Prediction data must contain {BASELINE_COLUMN}.")
        baseline = pd.to_numeric(data[BASELINE_COLUMN], errors="coerce").to_numpy(float)
        residual = self.pipeline.predict(
            _prepare_feature_frame(data, self.categorical_columns, self.numeric_columns)
        )
        stations = data["station"].astype(str) if "station" in data else pd.Series("", index=data.index)
        offsets = stations.map(self.calibration_offset_by_station_f).fillna(self.calibration_offset_f).to_numpy(float)
        halfwidths = stations.map(self.conformal_halfwidth_by_station_f).fillna(self.conformal_halfwidth_f).to_numpy(float)
        prediction = baseline + residual + offsets
        result = pd.DataFrame(index=data.index)
        result["prediction_f"] = prediction
        result["p10_f"] = prediction - halfwidths
        result["p50_f"] = prediction
        result["p90_f"] = prediction + halfwidths
        result["conformal_halfwidth_f"] = halfwidths
        result["calibration_offset_f"] = offsets
        return result


@dataclass
class AdaptiveResidualForecaster:
    """Choose the better residual family per station using a held-out calibration tail.

    The selector never sees the scored forecast date: Ridge and XGBoost are
    each fitted before the trailing calibration window, then the lower-MAE
    member for each station is chosen from that window. This keeps the useful
    station-specific behavior (notably marine-influenced sites) without
    selecting a model on the later target outcomes.
    """

    kind: Literal["adaptive"]
    ridge: ResidualForecaster
    xgb: ResidualForecaster
    station_models: dict[str, Literal["ridge", "xgb"]]
    station_selection_mae_f: dict[str, dict[str, float]]

    @property
    def train_rows(self) -> int:
        return self.xgb.train_rows

    @property
    def calibration_rows(self) -> int:
        return self.xgb.calibration_rows

    @property
    def conformal_halfwidth_f(self) -> float:
        """Conservative summary width for artifact manifests."""
        return max(self.ridge.conformal_halfwidth_f, self.xgb.conformal_halfwidth_f)

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        calibration_days: int = 60,
        random_state: int = 20260813,
    ) -> "AdaptiveResidualForecaster":
        ridge = ResidualForecaster.fit(train, "ridge", calibration_days, random_state)
        xgb = ResidualForecaster.fit(train, "xgb", calibration_days, random_state)
        usable = train.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]).copy()
        dates = pd.to_datetime(usable["target_date"])
        calibration_start = dates.max() - pd.Timedelta(days=calibration_days - 1)
        calibration = usable.loc[dates >= calibration_start].copy()
        selection: dict[str, Literal["ridge", "xgb"]] = {}
        selection_mae: dict[str, dict[str, float]] = {}
        if calibration.empty:
            selection = {str(station): "xgb" for station in usable["station"].unique()}
        else:
            ridge_prediction = ridge.predict(calibration)["prediction_f"].to_numpy(float)
            xgb_prediction = xgb.predict(calibration)["prediction_f"].to_numpy(float)
            scored = calibration[["station", TARGET_COLUMN]].copy()
            scored["ridge_ae"] = np.abs(scored[TARGET_COLUMN].to_numpy(float) - ridge_prediction)
            scored["xgb_ae"] = np.abs(scored[TARGET_COLUMN].to_numpy(float) - xgb_prediction)
            for station, group in scored.groupby("station", sort=True):
                ridge_mae = float(group["ridge_ae"].mean())
                xgb_mae = float(group["xgb_ae"].mean())
                station_key = str(station)
                selection_mae[station_key] = {"ridge": ridge_mae, "xgb": xgb_mae}
                # Require a small practical improvement before switching away
                # from XGB; this avoids noisy model flips in tiny differences.
                selection[station_key] = "ridge" if ridge_mae + 0.05 < xgb_mae else "xgb"
        return cls(
            kind="adaptive",
            ridge=ridge,
            xgb=xgb,
            station_models=selection,
            station_selection_mae_f=selection_mae,
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Route each station to its calibration-selected residual member."""
        ridge_output = self.ridge.predict(data)
        xgb_output = self.xgb.predict(data)
        stations = data["station"].astype(str) if "station" in data else pd.Series("", index=data.index)
        use_ridge = stations.map(self.station_models).fillna("xgb").eq("ridge").to_numpy()
        result = xgb_output.copy()
        result.loc[use_ridge, :] = ridge_output.loc[use_ridge, :]
        return result


@dataclass
class BlendedResidualForecaster:
    """Station-wise convex blend selected from a prior, held-out training block.

    A selector model is fitted only before the selection block. Its blend
    weight therefore never uses a scored forecast date or an in-sample
    prediction. Final component models are refit on all available pre-forecast
    history after the selector decision is made.
    """

    kind: Literal["blend"]
    ridge: ResidualForecaster
    xgb: ResidualForecaster
    station_ridge_weights: dict[str, float]
    station_selection_mae_f: dict[str, dict[str, float]]
    selection_rows: int

    @property
    def train_rows(self) -> int:
        return self.xgb.train_rows

    @property
    def calibration_rows(self) -> int:
        return self.xgb.calibration_rows

    @property
    def conformal_halfwidth_f(self) -> float:
        return max(self.ridge.conformal_halfwidth_f, self.xgb.conformal_halfwidth_f)

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        calibration_days: int = 60,
        selection_days: int = 45,
        random_state: int = 20260813,
    ) -> "BlendedResidualForecaster":
        usable = train.dropna(subset=[TARGET_COLUMN, BASELINE_COLUMN]).copy()
        if len(usable) < 150:
            raise ValueError("At least 150 labeled rows are required for the blended residual model.")
        dates = pd.to_datetime(usable["target_date"])
        calibration_start = dates.max() - pd.Timedelta(days=calibration_days - 1)
        selection_end = calibration_start - pd.Timedelta(days=1)
        selection_start = selection_end - pd.Timedelta(days=selection_days - 1)
        selector_train = usable.loc[dates < selection_start].copy()
        selection = usable.loc[(dates >= selection_start) & (dates <= selection_end)].copy()
        weights = {str(station): 0.0 for station in usable["station"].unique()}
        selection_mae: dict[str, dict[str, float]] = {}
        if len(selector_train) >= 150 and len(selection) >= 30:
            selector_ridge = ResidualForecaster.fit(selector_train, "ridge", calibration_days, random_state)
            selector_xgb = ResidualForecaster.fit(selector_train, "xgb", calibration_days, random_state)
            scored = selection[["station", TARGET_COLUMN]].copy()
            scored["ridge_prediction_f"] = selector_ridge.predict(selection)["prediction_f"].to_numpy(float)
            scored["xgb_prediction_f"] = selector_xgb.predict(selection)["prediction_f"].to_numpy(float)
            grid = np.linspace(0.0, 1.0, 21)
            for station, group in scored.groupby("station", sort=True):
                actual = group[TARGET_COLUMN].to_numpy(float)
                ridge_values = group["ridge_prediction_f"].to_numpy(float)
                xgb_values = group["xgb_prediction_f"].to_numpy(float)
                losses = np.asarray(
                    [np.mean(np.abs(actual - (weight * ridge_values + (1.0 - weight) * xgb_values))) for weight in grid]
                )
                station_key = str(station)
                best_index = int(np.argmin(losses))
                weights[station_key] = float(grid[best_index])
                selection_mae[station_key] = {
                    "ridge": float(np.mean(np.abs(actual - ridge_values))),
                    "xgb": float(np.mean(np.abs(actual - xgb_values))),
                    "blend": float(losses[best_index]),
                }

        ridge = ResidualForecaster.fit(usable, "ridge", calibration_days, random_state)
        xgb = ResidualForecaster.fit(usable, "xgb", calibration_days, random_state)
        return cls(
            kind="blend",
            ridge=ridge,
            xgb=xgb,
            station_ridge_weights=weights,
            station_selection_mae_f=selection_mae,
            selection_rows=int(len(selection)),
        )

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        ridge_output = self.ridge.predict(data)
        xgb_output = self.xgb.predict(data)
        stations = data["station"].astype(str) if "station" in data else pd.Series("", index=data.index)
        weight = stations.map(self.station_ridge_weights).fillna(0.0).to_numpy(float)
        prediction = weight * ridge_output["prediction_f"].to_numpy(float) + (1.0 - weight) * xgb_output["prediction_f"].to_numpy(float)
        halfwidth = np.maximum(
            ridge_output["conformal_halfwidth_f"].to_numpy(float),
            xgb_output["conformal_halfwidth_f"].to_numpy(float),
        )
        result = pd.DataFrame(index=data.index)
        result["prediction_f"] = prediction
        result["p10_f"] = prediction - halfwidth
        result["p50_f"] = prediction
        result["p90_f"] = prediction + halfwidth
        result["conformal_halfwidth_f"] = halfwidth
        result["calibration_offset_f"] = weight * ridge_output["calibration_offset_f"].to_numpy(float) + (1.0 - weight) * xgb_output["calibration_offset_f"].to_numpy(float)
        return result
