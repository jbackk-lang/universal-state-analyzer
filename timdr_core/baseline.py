"""
timdr_core/baseline.py — median/MAD z OKRESU KALIBRACJI, osobno od okna
ocenianego, do przekazania jako `baseline=` do TIMDRCore.anomalies() (i
`baseline_spread=` do TIMDRCore.defekt()).

PO CO TO JEST: bez tego, anomalies()/defekt() liczą swój punkt odniesienia
z TEGO SAMEGO okna, które oceniają ("self"). To ma znany, przetestowany
ślepy punkt (patrz README, sekcja "Ograniczenia", i test_baseline.py):
sygnał, który jest nietypowy PRZEZ CAŁE obserwowane okno, wychodzi jako
statystycznie normalny, bo nie ma z czym go porównać w danych, które widzi.
To dokładnie ten sam błąd, który w TIMDR-Crypto-Graph nazwano "ślepym
punktem self-eq" (izolowany, chronicznie zły klaster niewidoczny dla
`omega_hotspot()`, bo jego "norma" to on sam).

Dwie funkcje, dwie strategie odniesienia (mirror `test_eq_definitions.py`/
`test_eq_cohort.py` z TIMDR-Crypto-Graph, tu dla pojedynczego szeregu S(t)
zamiast grafu):

- `baseline_from_calibration(calib_values)` — (median, mad) z JEDNEGO
  osobnego okresu kalibracji tego samego bytu ("kim ten sygnał był,
  zanim zaczęło się coś podejrzanego"). Wymaga, żeby okres kalibracji
  faktycznie poprzedzał to, co oceniasz, i był wolny od badanej anomalii
  — inaczej to nic nie rozwiązuje (patrz README).

- `cohort_baseline(calib_values_per_entity, cohort_of)` — (median, mad) per
  byt, liczone z MEDIANY median-ów kalibracyjnych bytów w TEJ SAMEJ
  kohorcie. Rozwiązuje przypadek, którego `baseline_from_calibration()`
  NIE rozwiązuje: byt nietypowy JUŻ W OKRESIE KALIBRACJI (bo wtedy jego
  własna kalibracja też jest "zła"). WAŻNE ZAŁOŻENIE, zweryfikowane w
  TIMDR-Crypto-Graph (`test_eq_definitions.py`, `test_kmeans_cohort_risk.py`):
  `cohort_of` MUSI pochodzić z etykiety NIEZALEŻNEJ od ocenianego sygnału
  i od tego, kto z kim jest powiązany. Najbardziej oczywista literalna
  implementacja "peer-group" — sąsiedzi w grafie/sieci połączeń — jest
  PUŁAPKĄ, nie rozwiązaniem: dla odizolowanej, wewnętrznie spójnej grupy
  (np. zmowy) sąsiedzi TO współoskarżeni, więc ich mediana odtwarza ten
  sam ślepy punkt co self-baseline, czasem gorzej. Tak samo automatyczne
  klastrowanie (k-means) na cechach może odtworzyć tę pułapkę, jeśli grupa
  jest zdominowana (>50%) we własnym automatycznie znalezionym skupieniu
  — żadnego takiego automatycznego klastrowania NIE MA w tym module,
  świadomie: `cohort_of` musi dostarczyć wywołujący, z etykiety spoza tego
  sygnału (segment/typ/kategoria), nie z algorytmu klastrującego dane,
  które ocenia.

ZNANE OGRANICZENIE (zweryfikowane w tests/test_baseline.py, opisane też w
README): przy MAŁEJ liczbie członków kohorty (sprawdzone przy n=5) mad
między-bytowych średnich jest statystycznie niestabilne - zwykły normalny
byt, który przez przypadek wylosował wartość odstającą od reszty swojej
małej kohorty, może dostać masowe fałszywe flagi w anomalies(), mimo że
nic nietypowego nie robi. To nie jest specyficzne dla kolegujących się
grup - to zwykła niestabilność MAD przy małej próbie, więc większe
kohorty (i/lub minimalna liczba członków) są bezpieczniejsze, choć nie
ma tu (jeszcze) wbudowanego progu bezpieczeństwa - patrz README,
"Kierunki rozwoju".
"""
from __future__ import annotations

import numpy as np


def baseline_from_calibration(calib_values, mad_scale: float = 1.4826) -> tuple[float, float]:
    """calib_values: tablica z OSOBNEGO okresu kalibracji (nie z okna,
    które potem ocenisz w anomalies()/defekt()). Zwraca (median, mad).
    Pusta/cała-NaN tablica -> (0.0, 0.0), wywołujący dostanie mad=0 i
    anomalies() sam zastosuje podłogę (floor_frac) na tej podstawie."""
    calib = np.asarray(calib_values, float)
    calib = calib[np.isfinite(calib)]
    if len(calib) == 0:
        return 0.0, 0.0
    med = float(np.median(calib))
    mad = float(np.median(np.abs(calib - med)) * mad_scale)
    return med, mad


def cohort_baseline(calib_values_per_entity: dict, cohort_of: dict, mad_scale: float = 1.4826) -> dict:
    """calib_values_per_entity: {entity_id: tablica z okresu kalibracji
    TEGO bytu}. cohort_of: {entity_id: etykieta_kohorty} - dowolny
    hashowalny label, NIEZALEŻNY od grafu/sygnału (patrz zastrzeżenie w
    docstringu modułu). Zwraca {entity_id: (median, mad)} - median/mad
    kohorty, do której należy dany byt, liczone z median-ów
    kalibracyjnych WSZYSTKICH bytów tej kohorty (włącznie z samym bytem -
    to celowe, mediana jest odporna, więc pojedynczy nietypowy byt w
    kohorcie z >=3-4 normalnymi wciąż nie zdominuje mediany kohorty).
    Byty bez pokrywającego się wpisu w obu słownikach są pomijane w
    wyniku - wywołujący dostaje z powrotem mniej kluczy niż wejściowych
    bytów, jeśli dane są niepełne, nie błąd."""
    entity_med: dict = {}
    for eid, vals in calib_values_per_entity.items():
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        entity_med[eid] = float(np.median(vals))

    cohorts: dict = {}
    for eid, c in cohort_of.items():
        if eid in entity_med:
            cohorts.setdefault(c, []).append(eid)

    result: dict = {}
    for members in cohorts.values():
        meds = np.array([entity_med[e] for e in members])
        cohort_med = float(np.median(meds))
        cohort_mad = float(np.median(np.abs(meds - cohort_med)) * mad_scale)
        for e in members:
            result[e] = (cohort_med, cohort_mad)
    return result
