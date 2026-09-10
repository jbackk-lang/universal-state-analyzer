"""timdr_core/hypothesis_testing.py -- środowisko do testowania nowych
hipotez detekcyjnych na operatorach timdr_core (anomalia/defekt/rezonans/
skręt/ringdown), zbudowane wokół rzetelnego testu statystycznego zamiast
"na oko wygląda sensownie".

GENEZA (2026-09-08): to NIE jest nowa implementacja. Ten moduł
sibling-importuje `timdr_formalism.pipeline` z `TIMDR-Math-Formalism`
(repo-siostra pod tym samym katalogiem nadrzędnym) zamiast duplikować tę
matematykę trzeci raz w tym ekosystemie -- ten sam protokół (pre-
rejestracja, Mann-Whitney U, rozmiar efektu rank-biserial, kontrola
pozytywna/negatywna, korekta Bonferroniego) już istniał dwa razy:
    1. `TIMDR-Math-Formalism/timdr_formalism/pipeline.py` -- pełna,
       ogólna implementacja (pre-rejestracja + kontrole + Bonferroni),
       pierwotnie zbudowana do testowania hipotez numerologicznych/
       formalizmu matematycznego.
    2. `TIMDR-Earthquake-Core/precursor_validation.py` -- lżejsza,
       węższa kopia (tylko Mann-Whitney + próg efektu), zbudowana do
       testowania jednej konkretnej hipotezy (czy ringdown_resonance()
       jest sygnałem precursorem trzęsień ziemi).
(2) została naprawiona w tej samej sesji, w której powstał ten plik, żeby
sibling-importować z (1) zamiast trzymać własną kopię -- ten moduł
stosuje dokładnie tę samą naprawę na starcie, zamiast tworzyć TRZECIĄ
kopię tu, w silniku domenowo-niezależnym, gdzie duplikacja byłaby
najbardziej kosztowna (to repo jest współdzielonym rdzeniem dla wielu
domen -- patrz __init__.py).

Import jest bezpieczny nawet gdy scipy jest niedostępne/ryzykowne do
zaimportowania (patrz nagłówek pipeline.py -- historia Windows Device
Guard blokującego DLL-e scipy): `mann_whitney_test(..., backend="numpy")`
nigdy nie dotyka scipy w trakcie liczenia, a sam import `pipeline.py` ma
`try/except` wokół `from scipy import stats`.

Typowy przepływ testowania NOWEJ hipotezy detekcyjnej w tym repo:
    1. Zbuduj kandydacką metrykę na bazie operatorów timdr_core (np.
       nowa kombinacja anomalie()/rezonans()/ringdown_resonance()).
    2. `Hypothesis(...)` + `Preregistration.create(...)` PRZED
       dotknięciem realnych danych.
    3. `run_controls(metric_fn, positive_injector, negative_generator_a,
       negative_generator_b)` -- bramka pozytywna/negatywna na danych
       syntetycznych, PRZED testem głównym.
    4. Jeśli bramka przeszła: `mann_whitney_test(test_values,
       background_values)` na realnych/głównych danych.
    5. `format_report(...)` -- czytelny raport, honorujący wynik
       negatywny jako pełną odpowiedź (patrz TestResult.verdict()).

Ten plik sam niczego nie definiuje na nowo -- jest cienką warstwą
re-eksportu + `_ensure_sibling_on_path()`, tak żeby `from
timdr_core.hypothesis_testing import ...` działało bez ręcznego
dopisywania sys.path przez każdego wywołującego.
"""
from __future__ import annotations

# ZWENDOROWANE 2026-09-10 (patrz naglowek timdr_core/_vendor_timdr_formalism_pipeline.py
# dla pelnego uzasadnienia): ten modul ladowal wczesniej TIMDR-Math-Formalism
# przez sys.path sibling-import z folderu-siostry na dysku. Zamienione na
# lokalna, zwendorowana kopie (import relatywny w obrebie tego pakietu),
# zeby to repo dzialalo samodzielnie po sklonowaniu WYLACZNIE siebie
# (decyzja na wyrazna prosbe: "repozytoria kodu maja byc niezalezne od
# siebie"). Zachowanie/matematyka bez zmian.
from ._vendor_timdr_formalism_pipeline import (
    Hypothesis,
    Preregistration,
    TestResult,
    ControlResult,
    mann_whitney_test,
    run_controls,
    rank_biserial_effect_size,
    effect_size_label,
    bonferroni_correct,
    format_report,
    sieve_of_eratosthenes,
    random_background,
    ar1_noise,
)

__all__ = [
    "Hypothesis",
    "Preregistration",
    "TestResult",
    "ControlResult",
    "mann_whitney_test",
    "run_controls",
    "rank_biserial_effect_size",
    "effect_size_label",
    "bonferroni_correct",
    "format_report",
    "sieve_of_eratosthenes",
    "random_background",
    "ar1_noise",
]
