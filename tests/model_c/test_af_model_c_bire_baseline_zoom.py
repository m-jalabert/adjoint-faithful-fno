from __future__ import annotations

from pathlib import Path

import numpy as np

from bire_repro.af_model_c_bire_baseline_zoom import (
    CSV_NAME,
    FIELDS,
    FIGURE_NAME,
    METHODS,
    curve_summary,
    first_worse_lead,
    load_contract,
    plot_companion,
    write_csv,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "config" / "model_c_bire_baseline_zoom_v1.json"


def _arrays() -> dict[str, np.ndarray]:
    leads = np.arange(0, 201, 10, dtype=np.int16)
    arrays: dict[str, np.ndarray] = {"lead_days": leads}
    for field_index, field in enumerate(FIELDS, start=1):
        climatology = np.full(leads.size, 0.02 * field_index)
        persistence = np.linspace(0.0, 0.03 * field_index, leads.size)
        model = np.linspace(0.0, 0.12 * field_index, leads.size)
        for method, curve in (
            ("model", model),
            ("climatology", climatology),
            ("persistence", persistence),
        ):
            arrays[f"{method}__rmse__{field}"] = np.tile(curve, (15, 1))
    return arrays


def test_frozen_baseline_zoom_contract() -> None:
    contract, path, digest = load_contract(CONTRACT)
    assert path == CONTRACT
    assert len(digest) == 64
    assert contract["decision_contract"]["descriptive_only"] is True


def test_first_worse_lead_and_curve_summary() -> None:
    arrays = _arrays()
    leads = arrays["lead_days"]
    crossing = first_worse_lead(
        arrays["model__rmse__sst"],
        arrays["persistence__rmse__sst"],
        leads,
    )
    assert crossing == 10
    summary = curve_summary(arrays)
    assert summary["sst"]["crossings"]["persistence"] == 10
    assert summary["phihyd_surface"]["crossings"]["climatology"] == 40
    assert set(summary["sst"]["selected_leads"]["200"]) == set(METHODS)


def test_companion_plot_and_csv(tmp_path: Path) -> None:
    arrays = _arrays()
    plot_companion(
        tmp_path,
        arrays,
        {"sst": [0.0, 0.08], "phihyd_surface": [0.0, 0.10]},
    )
    write_csv(tmp_path, arrays)
    assert (tmp_path / FIGURE_NAME).stat().st_size > 0
    rows = (tmp_path / CSV_NAME).read_text().splitlines()
    assert rows[0] == "field,method,lead_days,mean,p10,p90"
    assert len(rows) == 1 + len(FIELDS) * len(METHODS) * 21
