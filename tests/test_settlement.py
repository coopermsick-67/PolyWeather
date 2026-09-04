import ast
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import polyweather.settlement as settlement
from polyweather.markets import normal_bracket_probabilities
from polyweather.settlement import (
    EXAMPLE_BUCKET_WIDTHS,
    MarketContract,
    bucket_hit_rate,
    illustrative_bucket_hit_rate,
    load_market_contracts,
)


def _sample_predictions() -> pd.DataFrame:
    # Mirrors artifacts/backtest_20_enhanced/rolling_predictions.parquet's shape:
    # station, target_date, xgb_prediction_f, tmax_f, p10_f, p90_f.
    return pd.DataFrame(
        [
            {"station": "KNYC", "target_date": date(2026, 1, 1), "xgb_prediction_f": 80.0, "tmax_f": 80.4, "p10_f": 77.0, "p90_f": 83.0},
            {"station": "KNYC", "target_date": date(2026, 1, 2), "xgb_prediction_f": 60.0, "tmax_f": 90.0, "p10_f": 57.0, "p90_f": 63.0},
            {"station": "KMDW", "target_date": date(2026, 1, 1), "xgb_prediction_f": 40.0, "tmax_f": 40.2, "p10_f": 37.0, "p90_f": 43.0},
        ]
    )


def _verified_contract(**overrides) -> MarketContract:
    fields = {
        "station_id": "KNYC",
        "provider": "ExampleProvider",
        "settlement_source": "nws_climate_report",
        "bucket_definition": ((78, 79), (80, 81), (82, 83)),
        "rounding_rule": "round half up",
        "contract_version": "v1",
        "verified_at": date(2026, 1, 1),
        "verified_by": "coopermsick@gmail.com",
    }
    fields.update(overrides)
    return MarketContract(**fields)


def test_verified_contract_can_be_scored():
    contract = _verified_contract()
    predictions = _sample_predictions()
    result = bucket_hit_rate(predictions, contract)
    assert result["station_id"] == "KNYC"
    assert result["contract_version"] == "v1"
    assert result["n_predictions"] == 2
    # Row 1: prediction 80 -> bucket 80-81, actual 80.4 -> rounds to 80 -> same bucket -> hit.
    # Row 2: prediction 60 -> falls outside all buckets -> no hit.
    assert result["bucket_hit_rate"] == pytest.approx(0.5)
    assert result["mean_claimed_probability_on_winning_bucket"] is not None


def test_unverified_contract_is_rejected_at_construction():
    with pytest.raises(ValueError, match="verified_at and verified_by"):
        MarketContract(
            station_id="KNYC",
            provider="ExampleProvider",
            settlement_source="nws_climate_report",
            bucket_definition=((80, 81),),
            rounding_rule="round half up",
            contract_version="v1",
            verified_at=None,
            verified_by=None,
        )


def test_malformed_contract_is_rejected_not_guessed():
    # Bad settlement_source: not one of the station's own source_priority entries.
    with pytest.raises(ValueError, match="source_priority"):
        _verified_contract(settlement_source="a-source-nobody-verified")
    # Bad rounding rule: must be one of the explicit supported set, never assumed.
    with pytest.raises(ValueError, match="rounding_rule"):
        _verified_contract(rounding_rule="round to nearest degree, whatever that means")
    # Non-integer bucket bounds.
    with pytest.raises(ValueError, match="integer degrees"):
        _verified_contract(bucket_definition=((78.5, 79.5),))


def test_load_market_contracts_rejects_unverified_or_malformed_records(tmp_path):
    path = tmp_path / "contracts.json"
    path.write_text(
        '['
        '{"station_id": "KNYC", "provider": "P", "settlement_source": "nws_climate_report", '
        '"bucket_definition": [[80, 81]], "rounding_rule": "round half up", "contract_version": "v1"}'
        ']',
        encoding="utf-8",
    )
    # verified_at/verified_by are absent entirely -> must be rejected, not defaulted.
    with pytest.raises(ValueError):
        load_market_contracts(path)


def test_load_market_contracts_accepts_a_valid_verified_record(tmp_path):
    path = tmp_path / "contracts.json"
    path.write_text(
        '['
        '{"station_id": "KNYC", "provider": "P", "settlement_source": "nws_climate_report", '
        '"bucket_definition": [[80, 81]], "rounding_rule": "round half up", "contract_version": "v1", '
        '"verified_at": "2026-01-01", "verified_by": "coopermsick@gmail.com"}'
        ']',
        encoding="utf-8",
    )
    loaded = load_market_contracts(path)
    assert len(loaded) == 1
    assert loaded[0].is_verified


def test_load_market_contracts_accepts_an_empty_registry(tmp_path):
    # Shipping with zero populated real contracts is expected and fine.
    path = tmp_path / "contracts.json"
    path.write_text("[]", encoding="utf-8")
    assert load_market_contracts(path) == []


def test_bucket_probability_math_matches_normal_bracket_probabilities():
    mean, std = 80.0, 2.0
    buckets = [(78, 79), (80, 81)]
    direct = normal_bracket_probabilities(mean, std, buckets)

    predictions = pd.DataFrame(
        [{"station": "KNYC", "target_date": date(2026, 1, 1), "xgb_prediction_f": mean, "tmax_f": mean, "p10_f": mean - std * settlement._Z_90, "p90_f": mean + std * settlement._Z_90}]
    )
    result = illustrative_bucket_hit_rate(predictions, station_id="KNYC", bucket_widths=tuple(buckets))
    assert result["illustrative_mean_claimed_probability_on_winning_bucket"] == pytest.approx(direct["80-81"])


def test_illustrative_function_uses_example_bucket_widths_by_default():
    predictions = _sample_predictions()
    result = illustrative_bucket_hit_rate(predictions)
    assert result["illustrative_bucket_width_count"] == len(EXAMPLE_BUCKET_WIDTHS)
    assert "warning" in result
    assert "UNVERIFIED" in result["warning"]


def _dict_key_field_names_by_enclosing_function(tree: ast.AST) -> list[tuple[str, str | None]]:
    """Collect output *field names* (dict-literal keys), not doc prose.

    We deliberately look at `ast.Dict` keys rather than every string constant
    in the module -- a docstring that merely mentions `bucket_hit_rate` by
    name in prose is not an output field and shouldn't trip this guard; an
    actual `{"...hit_rate...": value}` key in a returned report dict is
    exactly the kind of field name the ground rules are protecting.
    """
    results: list[tuple[str, str | None]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Dict(self, node: ast.Dict) -> None:
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    results.append((key.value, self.stack[-1] if self.stack else None))
            self.generic_visit(node)

    Visitor().visit(tree)
    return results


def test_illustrative_vs_verified_naming_distinction_is_enforced():
    """Lint-style guard: any 'hit_rate' field name must say 'illustrative' or
    live inside a function that actually checks contract verification.

    This protects the ground rule that an illustrative/unverified number can
    never be confused with a real, contract-verified one -- if a future edit
    added a bare `"hit_rate"` output field outside a verified-contract-gated
    function, this test fails.
    """
    source = Path(settlement.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    verified_guarded_functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "hit_rate" in node.name.casefold() and "illustrative" not in node.name.casefold():
            body_source = ast.get_source_segment(source, node) or ""
            assert "is_verified" in body_source, f"{node.name} produces a real hit-rate value but has no is_verified guard in its body"
            verified_guarded_functions.add(node.name)

    assert verified_guarded_functions, "expected at least one verified-contract-gated hit-rate function (bucket_hit_rate)"

    for field_name, enclosing in _dict_key_field_names_by_enclosing_function(tree):
        if "hit_rate" not in field_name.casefold():
            continue
        if "illustrative" in field_name.casefold():
            continue
        assert enclosing in verified_guarded_functions, (
            f"output field name {field_name!r} mentions hit_rate without 'illustrative' and is not inside a "
            f"verified-contract-guarded function (found in {enclosing!r})"
        )
