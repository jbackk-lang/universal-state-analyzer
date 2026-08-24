"""
test_selfbaseline_recovery.py -- pytanie analogiczne do test_recovery.py z
siostrzanego repo TIMDR-Crypto-Graph: gdy anomalia w sygnale sie konczy i
kolejne odczyty wracaja do normy, czy anomalies() (self-baseline, bez
baseline=) dalej falszywie flaguje NOWE, normalne probki tylko dlatego, ze
stare anomalne probki wciaz siedza w oknie referencyjnym?

WYNIK (rozny od Crypto-Graph, gdzie state to EMA i powrot jest STOPNIOWY -
tu z jest liczone na nowo z median/MAD kazdego wywolania, wiec jest tylko
JEDNO pytanie: czy median/MAD calego okna sa zanieczyszczone anomalia):

- Gdy anomalia to MNIEJSZOSC okna (sprawdzone <=50% W): odzysk jest PRAWIE
  NATYCHMIASTOWY, nie stopniowy - juz PIERWSZA normalna probka po evencie
  ma male |z|, mimo ze okno referencyjne WCIAZ zawiera anomalne probki.
  Powod: median() i MAD sa odporne na mniejszosciowe wartosci odstajace
  (punkt zalamania mediany to 50%) - nie trzeba czekac, az anomalia
  "wypadnie" z okna, tak jak trzeba by czekac na zanik EMA w Crypto-Graph.
- Gdy anomalia zblizona/przekracza 50% okna: median/MAD zaczynaja sie
  przesuwac w strone anomalii, normalne probki tuz po evencie wychodza
  jako SYSTEMATYCZNIE przesuniete (nieprzypadkowe |z|~2, nie szum) - to
  ten sam mechanizm co udokumentowany "slepy punkt self-baseline" w
  README/test_baseline.py (sygnal nietypowy PRZEZ CALE okno = niewidoczny/
  wypaczony), tu obserwowany w wersji czesciowej/przejsciowej.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from timdr_core import TIMDRCore

core = TIMDRCore()
W = 30  # rozmiar trailing window symulujacy typowe uzycie strumieniowe


def _z_of_current_sample(window_vals):
    _, z, _ = core.anomalies(np.arange(len(window_vals)), window_vals, factor=3.0)
    return z[-1]


def test_z_recovers_immediately_gdy_anomalia_to_mniejszosc_okna():
    """anomalia = 3 probki (10% z W=30) - sprawdzone na 5 niezaleznych
    ziarnach: |z| pierwszej normalnej probki PO evencie < 2.5 za kazdym
    razem (typowy szumowy zakres, nie falszywa flaga), mimo ze okno
    referencyjne wciaz zawiera te 3 anomalne probki."""
    n_pre, n_anom = 60, 3
    for seed in range(5):
        rng = np.random.default_rng(seed)
        full = np.concatenate([
            rng.normal(0, 1, n_pre),
            np.full(n_anom, 1000.0),
            rng.normal(0, 1, 20),
        ])
        event_end = n_pre + n_anom
        post_event_z = [
            _z_of_current_sample(full[max(0, i - W):i + 1])
            for i in range(event_end, event_end + 5)
        ]
        assert all(abs(z) < 2.5 for z in post_event_z), (
            f"seed={seed}: probki tuz po evencie wciaz falszywie odstaja "
            f"({post_event_z}), mimo ze anomalia to tylko 10% okna"
        )


def test_z_recovery_degraduje_gdy_anomalia_zblizona_do_polowy_okna():
    """Gdy anomalia siega ~73% okna (22/30), normalne probki PO evencie
    wychodza SYSTEMATYCZNIE przesuniete (|z| konsekwentnie ~2, nie losowy
    szum kolo 0) - median/MAD sa juz wypaczone przez wiekszosciowa
    kontaminacje okna. To NIE jest flagowane jako anomalia (wciaz < factor
    domyslny 3.0), ale przestaje byc "czystym" odczytem normalnosci -
    ten sam ślepy punkt co self-baseline na calym oknie, w wersji
    czesciowej. Udokumentowane, nie naprawiane tutaj (patrz README)."""
    n_pre, n_anom = 60, 22
    rng = np.random.default_rng(0)
    full = np.concatenate([
        rng.normal(0, 1, n_pre),
        np.full(n_anom, 1000.0),
        rng.normal(0, 1, 20),
    ])
    event_end = n_pre + n_anom
    post_event_z = [
        _z_of_current_sample(full[max(0, i - W):i + 1])
        for i in range(event_end, event_end + 5)
    ]
    # systematyczne przesuniecie (nie losowy szum ~N(0,1)), ale nadal
    # ponizej progu flagowania anomalies() (factor=3.0)
    assert all(1.0 < abs(z) < 3.0 for z in post_event_z), (
        f"oczekiwano systematycznego, ale niesflagowanego przesuniecia "
        f"przy 73% kontaminacji okna, dostano {post_event_z}"
    )
