"""Testy timdr_core.ringdown - detekcja OSCYLACYJNEGO (rezonansowego) vs
MONOTONICZNEGO powrotu do poziomu odniesienia po zdarzeniu. Weryfikacja na
syntetykach ze ZNANYM parametrem fizycznym (częstotliwość, stała czasowa
tłumienia) - patrz docstring modułu po wzór na teoretyczny damping_ratio.
"""
import numpy as np
import pytest

from timdr_core import TIMDRCore, ringdown_resonance


def _damped_oscillator(t, event_idx, f0, tau, amplitude=5.0, noise_sigma=0.0, seed=0):
    """t obejmuje CISZĘ przed event_idx (do estymacji baseline/szumu) +
    tłumiony oscylator PO event_idx: A*exp(-(t-t0)/tau)*cos(2*pi*f0*(t-t0))."""
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    post = t[event_idx:] - t[event_idx]
    x[event_idx:] = amplitude * np.exp(-post / tau) * np.cos(2 * np.pi * f0 * post)
    if noise_sigma:
        x = x + rng.normal(0, noise_sigma, len(t))
    return x


def _monotonic_decay(t, event_idx, tau, amplitude=5.0, noise_sigma=0.0, seed=0):
    """Jak wyżej, ale BEZ oscylacji - czysty zanik wykładniczy (przetłumiony
    oscylator, brak rezonansu do zaobserwowania)."""
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    post = t[event_idx:] - t[event_idx]
    x[event_idx:] = amplitude * np.exp(-post / tau)
    if noise_sigma:
        x = x + rng.normal(0, noise_sigma, len(t))
    return x


# ---------------------------------------------------------------------
# Główna walidacja: znany f0/tau -> odzyskana częstotliwość/tłumienie
# ---------------------------------------------------------------------

def test_underdamped_recovers_known_frequency_and_damping():
    fs = 100.0
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    f0, tau = 1.5, 1.2
    x = _damped_oscillator(t, event_idx, f0, tau, noise_sigma=0.05, seed=0)

    res = ringdown_resonance(t, x, event_idx=event_idx, pre_event_window=event_idx)

    assert res["is_oscillatory"] is True
    assert res["frequency_hz"] == pytest.approx(f0, rel=0.05)

    zeta_theory = 1.0 / np.sqrt((2 * np.pi * f0 * tau) ** 2 + 1)
    assert res["damping_ratio"] == pytest.approx(zeta_theory, rel=0.3)
    assert res["n_peaks_used"] >= 4  # kilka pełnych okresów ponad progiem szumu


def test_overdamped_monotonic_decay_is_not_oscillatory():
    """Czysty zanik wykładniczy (bez oscylacji) - klucz do rozróżnienia
    'rezonans' (oscylacyjny powrót) od zwykłego powrotu do równowagi."""
    fs = 100.0
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    x = _monotonic_decay(t, event_idx, tau=0.8, noise_sigma=0.05, seed=1)

    res = ringdown_resonance(t, x, event_idx=event_idx, pre_event_window=event_idx)

    assert res["is_oscillatory"] is False
    assert res["frequency_hz"] is None
    assert res["damping_ratio"] is None


def test_bug_niefiltrowany_szum_dawal_falszywy_rezonans():
    """Regresja: bez progu szumu (noise_floor_factor), szum w ogonie
    sygnału (gdzie amplituda już wygasła) generował dziesiątki fałszywych
    przejść przez baseline - zanik wykładniczy wychodził jako
    'is_oscillatory=True' z bezsensowną częstotliwością. Sprawdzone też z
    noise_floor_factor=0 (wyłączony filtr), żeby udokumentować, że błąd
    faktycznie by się odtworzył bez naprawy."""
    fs = 100.0
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    x = _monotonic_decay(t, event_idx, tau=0.8, noise_sigma=0.05, seed=1)

    res_filtered = ringdown_resonance(t, x, event_idx=event_idx, pre_event_window=event_idx)
    res_unfiltered = ringdown_resonance(
        t, x, event_idx=event_idx, pre_event_window=event_idx, noise_floor_factor=0.0,
    )

    assert res_filtered["is_oscillatory"] is False
    # bez filtra odtwarza się dokładnie ten błąd, który filtr ma naprawiać -
    # potwierdza, że test faktycznie sprawdza mechanizm naprawy, nie
    # przypadkowo zawsze-negatywny wynik
    assert res_unfiltered["n_crossings"] > 10


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_underdamped_frequency_stable_across_seeds(seed):
    """Ten sam sygnał, różne realizacje szumu - częstotliwość powinna
    wyjść blisko prawdy za każdym razem, nie tylko dla jednego wybranego
    ziarna (unikamy przypadkowego 'trafienia')."""
    fs = 100.0
    t = np.arange(0, 8.0, 1 / fs)
    event_idx = int(2.0 * fs)
    f0 = 2.0
    x = _damped_oscillator(t, event_idx, f0, tau=1.0, noise_sigma=0.05, seed=seed)
    res = ringdown_resonance(t, x, event_idx=event_idx, pre_event_window=event_idx)
    assert res["is_oscillatory"] is True
    assert res["frequency_hz"] == pytest.approx(f0, rel=0.08)


# ---------------------------------------------------------------------
# Przypadki brzegowe
# ---------------------------------------------------------------------

def test_too_short_window_returns_safe_default():
    t = np.array([0.0, 1.0])
    s = np.array([1.0, 2.0])
    res = ringdown_resonance(t, s, event_idx=0)
    assert res["is_oscillatory"] is False
    assert res["period_s"] is None


def test_event_idx_out_of_range_raises():
    t = np.arange(10, dtype=float)
    s = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        ringdown_resonance(t, s, event_idx=99)


def test_event_idx_zero_no_pre_history_falls_back_to_unfiltered():
    """Bez historii przed zdarzeniem (event_idx=0) nie da się oszacować
    szumu - udokumentowane ograniczenie: noise_floor=0 (brak filtrowania),
    NIE cichy crash ani zgadywanie."""
    fs = 100.0
    t = np.arange(0, 3.0, 1 / fs)
    x = 5.0 * np.exp(-t / 0.5) * np.cos(2 * np.pi * 2.0 * t)
    res = ringdown_resonance(t, x, event_idx=0, baseline=0.0)
    assert res["noise_floor"] == 0.0
    assert res["is_oscillatory"] is True  # bez szumu w tym czystym sygnale, wciąż poprawnie


def test_explicit_baseline_overrides_pre_event_mean():
    t = np.arange(0, 5.0, 0.1)
    event_idx = 20
    x = np.full_like(t, 10.0)
    x[event_idx:] = 10.0 + 2.0 * np.exp(-(t[event_idx:] - t[event_idx]))
    res = ringdown_resonance(t, x, event_idx=event_idx, baseline=10.0)
    assert res["baseline"] == 10.0


# ---------------------------------------------------------------------
# Integracja: zdarzenie znalezione przez defekt() -> analiza ringdown
# ---------------------------------------------------------------------

def test_analyze_ringdown_end_to_end_with_defekt():
    """Realistyczny przepływ: TIMDRCore.defekt() znajduje skok, potem
    analyze_ringdown() ocenia, czy powrót po nim jest oscylacyjny."""
    core = TIMDRCore()
    fs = 100.0
    t = np.arange(0, 6.0, 1 / fs)
    event_idx = int(2.0 * fs)

    s = np.full_like(t, 3.0)
    post = t[event_idx:] - t[event_idx]
    s[event_idx:] = 3.0 + 4.0 * np.exp(-post / 1.0) * np.cos(2 * np.pi * 1.8 * post)
    rng = np.random.default_rng(7)
    s_noisy = s + rng.normal(0, 0.05, len(s))

    de_idx, _ = core.defekt(s_noisy, factor=0.3)
    assert len(de_idx) > 0
    # bierzemy pierwszy wykryty skok blisko wstrzykniętego zdarzenia
    candidate = de_idx[np.argmin(np.abs(de_idx - event_idx))]
    assert abs(candidate - event_idx) <= 3

    results = core.analyze_ringdown(t, s_noisy, [candidate], pre_event_window=event_idx)
    assert len(results) == 1
    assert results[0]["is_oscillatory"] is True
    assert results[0]["frequency_hz"] == pytest.approx(1.8, rel=0.1)
