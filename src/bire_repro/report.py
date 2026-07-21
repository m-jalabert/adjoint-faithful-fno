"""Self-contained HTML/JSON audit report for a reproduction run."""

from __future__ import annotations

import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import config_sha256
from .data import canonical_store_path, validate_store


def _git_sha(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_files(root: Path) -> list[dict[str, Any]]:
    records = []
    if not root.exists():
        return records
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
            records.append({"path": str(path), "payload": payload})
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"path": str(path), "error": str(exc)})
    return records


def collect(config: Mapping[str, Any]) -> dict[str, Any]:
    project = Path(config["paths"]["project_root"])
    scratch = Path(config["paths"]["scratch_root"])
    store = canonical_store_path(config)
    try:
        data_validation: dict[str, Any] = validate_store(config, store)
    except Exception as exc:  # report must remain useful before production exists
        data_validation = {"valid": False, "unavailable": str(exc), "path": str(store)}
    figure_dir = Path(config["paths"]["figures"])
    figures = {
        str(number): {
            "png": (figure_dir / f"figure{number:02d}.png").is_file(),
            "pdf": (figure_dir / f"figure{number:02d}.pdf").is_file(),
        }
        for number in range(2, 12)
    }
    expected_mit = config["upstream"]["mitgcm_sha"]
    expected_code = config["upstream"]["paper_code_sha"]
    actual_mit = _git_sha(Path(config["paths"]["mitgcm_source"]))
    actual_code = _git_sha(Path(config["paths"]["paper_code"]))
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config.get("_config_path"),
        "config_sha256": config_sha256(config),
        "upstream": {
            "MITgcm": {"expected": expected_mit, "actual": actual_mit, "matches": actual_mit == expected_mit},
            "oceanfourcast": {
                "expected": expected_code,
                "actual": actual_code,
                "matches": actual_code == expected_code,
            },
        },
        "data": data_validation,
        "figures": figures,
        "project_manifests": _json_files(project / "manifests"),
        "scratch_manifests": _json_files(scratch / "manifests"),
        "known_limitations": [
            "The paper's exact MITgcm namelists and raw data were not published.",
            "The locked 15-layer interpretation is 1800 m because the paper's vertical specifications conflict.",
            "The validation and final-1000 inference ranges overlap by design.",
            "Recovered evaluation indices are absolute and overlap the declared training range; this is flagged data leakage.",
            "Primary RMSE is true area-weighted RMSE; archived unweighted MSE is exported only as a legacy diagnostic.",
        ],
    }


def _status(value: bool) -> str:
    return '<span class="ok">PASS</span>' if value else '<span class="bad">MISSING/FAIL</span>'


def render_html(report: Mapping[str, Any]) -> str:
    upstream_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(str(item['expected']))}</code></td>"
        f"<td><code>{html.escape(str(item['actual']))}</code></td><td>{_status(item['matches'])}</td></tr>"
        for name, item in report["upstream"].items()
    )
    figure_rows = "".join(
        f"<tr><td>Figure {number}</td><td>{_status(item['png'])}</td><td>{_status(item['pdf'])}</td></tr>"
        for number, item in report["figures"].items()
    )
    limitations = "".join(f"<li>{html.escape(text)}</li>" for text in report["known_limitations"])
    data_json = html.escape(json.dumps(report["data"], indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Bire et al. JAMES 2025 reproduction</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #18202a; }}
code, pre {{ background: #f3f5f7; }} pre {{ padding: 1rem; overflow: auto; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ border: 1px solid #ccd3da; padding: .45rem; text-align: left; }}
.ok {{ color: #087830; font-weight: 700; }} .bad {{ color: #b31b1b; font-weight: 700; }}
</style></head><body>
<h1>Bire et al. (2025) double-gyre reproduction audit</h1>
<p>Generated {html.escape(str(report['generated_utc']))}. Configuration digest:
<code>{html.escape(str(report['config_sha256']))}</code>.</p>
<h2>Upstream provenance</h2><table><thead><tr><th>Artifact</th><th>Expected</th><th>Actual</th><th>Status</th></tr></thead><tbody>{upstream_rows}</tbody></table>
<h2>Canonical data validation</h2><p>{_status(bool(report['data'].get('valid')))}</p><pre>{data_json}</pre>
<h2>Paper figures</h2><table><thead><tr><th>Panel</th><th>PNG</th><th>PDF</th></tr></thead><tbody>{figure_rows}</tbody></table>
<h2>Known limitations and declared deviations</h2><ul>{limitations}</ul>
<p>See <code>docs/bire_a0_reconstruction.md</code> and all JSON manifests for the complete audit trail.</p>
</body></html>"""


def generate_report(
    config: Mapping[str, Any], output_dir: str | Path | None = None
) -> tuple[Path, Path]:
    output = Path(output_dir or config["paths"]["reports"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = collect(config)
    json_path = output / "reproduction-report.json"
    html_path = output / "reproduction-report.html"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    html_path.write_text(render_html(payload))
    return html_path, json_path
