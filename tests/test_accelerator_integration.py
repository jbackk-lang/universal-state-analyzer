"""Integracyjny test przykladu 'akcelerator' - potwierdza, ze timdr_core
faktycznie dziala na wyjsciu z symulacji lattice/glueball (nie tylko na
syntetycznych tablicach w test_core.py), i ze wstrzykniety skok zostaje
zlapany na prawdziwej sciezce (import + wywolanie funkcji z
glueball_mass.py/lattice_demo.py, nie mock)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "accelerator"))

import numpy as np

from analyze_trajectory import build_trajectory
from timdr_core import TIMDRCore


def test_build_trajectory_shapes():
    series = build_trajectory(T=10, N=6, mode="basic")
    assert set(series.keys()) == {"glueball_C", "wilson_plaq", "su3_action"}
    for arr in series.values():
        assert len(arr) == 10
        assert np.all(np.isfinite(arr))


def test_injected_spike_detected_on_real_trajectory():
    np.random.seed(7)
    series = build_trajectory(T=15, N=6, mode="basic")
    spike_idx = 8
    series["glueball_C"][spike_idx] = series["glueball_C"][spike_idx] + 50 * (np.std(series["glueball_C"]) or 1.0)

    core = TIMDRCore()
    t = np.arange(15)
    result = core.analyze_multi(t, series)
    assert spike_idx in result["anomaly_idx"]["glueball_C"]
