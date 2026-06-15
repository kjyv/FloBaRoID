"""Tests for measurement-data loading and multi-file bookkeeping."""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MEAS = "data/THREELINK/SIM/measurements_opt1_fb.npz"


def _opt(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    import yaml

    with open("configs/threeLinks.yaml") as f:
        opt = yaml.safe_load(f)
    opt["startOffset"] = 0
    opt["skipSamples"] = 0
    opt["selectBlocksFromMeasurements"] = 0
    opt["verbose"] = 0
    opt["showTiming"] = 0
    return opt


def test_file_boundaries_single_file(monkeypatch):
    """A single measurement file yields boundaries [0, n] (no per-trajectory weighting)."""
    import numpy as np

    from identification.data import Data

    data = Data(_opt(monkeypatch))
    data.init_from_files([[MEAS]])
    n = np.load(MEAS)["positions"].shape[0]
    assert data.file_boundaries == [0, n]
    assert len(data.file_boundaries) <= 2  # the weighting code only activates for >2


def test_file_boundaries_multi_file(monkeypatch):
    """Two measurement files yield cumulative boundaries [0, n, 2n] for per-file weighting."""
    import numpy as np

    from identification.data import Data

    data = Data(_opt(monkeypatch))
    data.init_from_files([[MEAS], [MEAS]])
    n = np.load(MEAS)["positions"].shape[0]
    assert data.file_boundaries == [0, n, 2 * n]
