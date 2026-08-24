"""Testy timdr_core.core - konwencja jak test_timdr_core_finance.py:
odporność na n=0/1/2, kontrola pułapki zero-inflation (MAD=0/spread=0),
kontrola liczenia gradientu względem CZASU (nie indeksu) na danych z luką,
i sprawdzenie, że wstrzyknięta anomalia faktycznie zostaje złapana.
"""
import numpy as np
import pytest

from timdr_core import TIMDRCore

core = TIMDRCore()


@pytest.mark.parametrize("n", [0, 1, 2])
def test_trm_n_male(n):
    t = np.arange(n, dtype=float)
    s = np.arange(n, dtype=float)
    assert len(core.trm(t, s)) == n


@pytest.mark.parametrize("n", [0, 1, 2])
def test_flow_n_male(n):
    t = np.arange(n, dtype=float)
    s = np.arange(n, dtype=float)
    assert len(core.flow(t, s)) == n


@pytest.mark.parametrize("n", [0, 1, 2])
def test_twist_n_male(n):
    idx, dg = core.twist(np.arange(n, dtype=float), np.arange(n, dtype=float))
    assert len(idx) == 0


@pytest.mark.parametrize("n", [0, 1, 2])
def test_anomalies_n_male(n):
    idx, z, th = core.anomalies(np.arange(n, dtype=float), np.arange(n, dtype=float))
    assert len(z) == n


@pytest.mark.parametrize("n", [0, 1])
def test_defekt_n_male(n):
    idx, diffs = core.defekt(np.arange(n, dtype=float))
    assert len(idx) == 0


@pytest.mark.parametrize("n", [0, 1])
def test_rhythm_n_male(n):
    periods, score = core.rhythm(np.arange(n, dtype=float))
    assert periods == []
    assert score == 0.0


def test_anomalies_finds_injected_spike():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, size=100)
    s[50] = 1000.0  # jawny, ogromny skok
    idx, z, thr = core.anomalies(np.arange(100), s, factor=3.0)
    assert 50 in idx


def test_anomalies_mad_zero_floor_does_not_crash():
    # seria stała (MAD=0, std=0) - powinno dać skończony, niezerowy prog
    # z podłogi (floor_frac), nie ZeroDivisionError/NaN wszędzie
    s = np.full(20, 5.0)
    idx, z, thr = core.anomalies(np.arange(20), s)
    assert np.all(np.isfinite(z))
    assert thr > 0


def test_defekt_zero_spread_floor_does_not_crash():
    s = np.zeros(20)
    idx, diffs = core.defekt(s)
    assert np.all(np.isfinite(diffs))
    # same zera -> brak realnych skoków mimo podlogi
    assert len(idx) == 0


def test_defekt_finds_injected_jump():
    s = np.concatenate([np.zeros(10), np.full(10, 100.0)])
    idx, diffs = core.defekt(s, factor=0.3)
    assert 10 in idx  # indeks NOWEJ probki po skoku


def test_flow_uses_time_not_index_with_gap():
    # t ma luke (brakuje probek 3..7) - jesli flow liczyloby po indeksie
    # zamiast po t, tempo zmiany bylby zawyzone/zle skalowane na luce
    t = np.array([0, 1, 2, 8, 9, 10], dtype=float)
    s = np.array([0, 1, 2, 8, 9, 10], dtype=float)  # nachylenie = 1 wszedzie wzgledem t
    flow = core.flow(t, s, window=3)
    assert np.allclose(flow, 1.0, atol=1e-6)


def test_rhythm_detects_period_after_detrend():
    t = np.arange(200, dtype=float)
    # sygnal okresowy + trend liniowy - rhythm() musi odjac trend zeby wykryc okres
    s = np.sin(2 * np.pi * t / 10) + 0.05 * t
    periods, score = core.rhythm(s, max_lag=48, power_thresh=0.3)
    assert len(periods) > 0
    assert any(abs(p - 10) <= 2 for p in periods)


def test_rezonans_requires_min_count():
    n = 10
    idx_a = np.array([3, 5])
    idx_b = np.array([5, 7])
    idx_c = np.array([5])
    idx, counts = core.rezonans([idx_a, idx_b, idx_c], n=n, min_count=3)
    assert list(idx) == [5]
    assert counts[5] == 3
    assert counts[3] == 1


def test_analyze_multi_end_to_end():
    rng = np.random.default_rng(1)
    t = np.arange(60, dtype=float)
    params = {
        "a": rng.normal(0, 1, 60),
        "b": rng.normal(0, 1, 60),
        "c": rng.normal(0, 1, 60),
    }
    # wstrzyknij jednoczesna anomalie we wszystkich 3 na tym samym t
    for name in params:
        params[name][30] = 50.0
    result = core.analyze_multi(t, params, rezonans_min=3)
    assert 30 in result["rezonans_idx"]
    for name in params:
        assert 30 in result["anomaly_idx"][name]
