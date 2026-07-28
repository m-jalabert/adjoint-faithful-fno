from pathlib import Path

import pytest

from bire_repro.core.plots import PlotError, RolloutCatalog


def test_empty_rollout_catalog_is_actionable(tmp_path: Path):
    with pytest.raises(PlotError, match="no rollout Zarr groups"):
        RolloutCatalog(tmp_path)
