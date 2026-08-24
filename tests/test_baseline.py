"""
test_baseline.py -- replikuje na tym silniku ten sam problem i tę samą
poprawkę, jaka została znaleziona i zweryfikowana w TIMDR-Crypto-Graph
(test_v1_blind_spot.py -> test_v2_fix_verified.py -> test_eq_definitions.py
-> test_eq_cohort.py): self-baseline (median/MAD z tego samego okna, które
jest oceniane) jest ślepy na sygnał nietypowy PRZEZ CAŁE obserwowane okno.

Rosnąca trudność:
1. Bez baseline= w ogóle: pokazuje sam problem (ślepy punkt istnieje).
2. Z baseline_from_calibration() z osobnego, wcześniejszego okresu:
   pokazuje najprostszą poprawkę - działa, gdy jest gdzieś "normalny"
   fragment historii DANEGO bytu sprzed zmiany.
3. Z cohort_baseline(): pokazuje przypadek, którego (2) NIE rozwiązuje -
   byt nietypowy JUŻ W OKRESIE KALIBRACJI (więc jego własna kalibracja
   też jest "zła").

Do tego DWA znaleziska z pisania tych testów (nie założone z góry, odkryte
przy weryfikacji), oba udokumentowane też w README ("Ograniczenia"):

- Zła etykieta kohorty (zbudowana z samego ocenianego sygnału) NIE daje
  "niewidocznego" ringu, jak PEER_NB w TIMDR-Crypto-Graph - daje
  DEGENEROWANE (bardzo małe) mad kohorty, bo czysto-RING-owa kohorta ma
  bardzo zgodne między sobą średnie. To wciąż flaguje RING, ale przez
  artefakt statystyczny, nie przez poprawne porównanie z populacją.
- Nawet przy DOBRZE dobranej etykiecie kohorty, mała liczba członków (tu
  n=5) daje niestabilne mad między-bytowe - normalny byt, który przez
  przypadek wylosował wartość odstającą od reszty SWOJEJ małej kohorty,
  może dostać dziesiątki-setki fałszywych flag, mimo że nic nietypowego
  nie robi.
"""
import numpy as np
import pytest

from timdr_core import TIMDRCore
from timdr_core.baseline import baseline_from_calibration, cohort_baseline

core = TIMDRCore()


def test_self_baseline_blind_to_signal_chronically_shifted_for_whole_window():
    """Cały obserwowany odcinek jest chronicznie podniesiony (zmiana
    zaszła PRZED tym, co widzimy - w danych nie ma śladu "starego"
    poziomu). Bez baseline=, anomalies() normalizuje się do własnej
    mediany i nie widzi nic - to jest ten sam błąd, co self-eq w
    TIMDR-Crypto-Graph v1."""
    rng = np.random.default_rng(0)
    chronically_shifted = 50.0 + rng.normal(0, 1, 60)
    idx, z, thr = core.anomalies(np.arange(60), chronically_shifted, factor=3.0)
    assert len(idx) == 0


def test_baseline_from_calibration_catches_the_same_shift():
    """Ten sam sygnał, ale z baseline= policzonym przez
    baseline_from_calibration() z OSOBNEGO okresu SPRZED zmiany. Teraz
    odniesienie jest do dawnego, normalnego poziomu - cały post-okres
    wychodzi jako anomalia."""
    rng = np.random.default_rng(0)
    calib = rng.normal(0, 1, 200)  # normalny okres, przed zmianą, NIEUŻYWANY do liczenia mediany "self"
    post_change = 50.0 + rng.normal(0, 1, 60)
    baseline = baseline_from_calibration(calib)
    idx, z, thr = core.anomalies(np.arange(60), post_change, factor=3.0, baseline=baseline)
    assert len(idx) == 60


def test_baseline_from_calibration_does_not_help_if_shift_already_in_calibration():
    """Kontrola: jeśli zmiana zaszła JUŻ W OKRESIE KALIBRACJI (byt jest
    chronicznie inny od samego początku), baseline_from_calibration() z
    WŁASNEJ historii bytu nie pomaga - to nie jest błąd tej funkcji, to
    jest dokładnie granica, którą ma pokonać cohort_baseline() niżej."""
    rng = np.random.default_rng(1)
    calib_already_shifted = 50.0 + rng.normal(0, 1, 200)  # chronicznie zly JUZ w kalibracji
    post = 50.0 + rng.normal(0, 1, 60)
    baseline = baseline_from_calibration(calib_already_shifted)
    idx, z, thr = core.anomalies(np.arange(60), post, factor=3.0, baseline=baseline)
    assert len(idx) == 0  # nadal niewidoczny - self-kalibracja tez byla "zla"


def test_cohort_baseline_catches_group_chronic_from_start_of_calibration():
    """Replika test_eq_cohort.py Scenariusza A: N bytow, mala grupa RING
    jest chronicznie podniesiona JUZ W OKRESIE KALIBRACJI. cohort_of
    rozprasza czlonkow RING miedzy kohortami z normalnymi bytami (etykieta
    NIEZALEZNA od poziomu sygnalu - tu: numer bytu modulo 6, tak jak w
    TIMDR-Crypto-Graph). Mediana kohorty kotwiczy sie w normalnej
    wiekszosci, wiec RING wychodzi jako anomalia mimo ze self-baseline
    (i baseline_from_calibration z wlasnej historii) by go nie zlapaly."""
    rng = np.random.default_rng(2)
    N = 30
    RING = [0, 1, 2, 3, 4]
    calib_values = {}
    for i in range(N):
        level = 9.0 if i in RING else rng.uniform(0.5, 2.0)
        calib_values[i] = level + rng.normal(0, 0.2, 200)

    cohort_of = {i: i % 6 for i in range(N)}  # NIEZALEZNA od poziomu sygnalu
    baselines = cohort_baseline(calib_values, cohort_of)

    for i in RING:
        med, mad = baselines[i]
        idx, z, thr = core.anomalies(np.arange(200), calib_values[i], factor=3.0, baseline=(med, mad))
        assert len(idx) > 150  # niemal caly odcinek zaflagowany

    normal_id = next(i for i in range(N) if i not in RING)
    med, mad = baselines[normal_id]
    idx, z, thr = core.anomalies(np.arange(200), calib_values[normal_id], factor=3.0, baseline=(med, mad))
    assert len(idx) < 20  # normalny byt NIE zaflagowany przez wlasna kohorte


def test_cohort_baseline_small_cohort_can_false_flag_a_normal_member():
    """ZNALEZIONE PRZY PISANIU test_cohort_baseline_catches_group... (wyzej),
    NIE zalozone z gory: powyzszy test sprawdza tylko JEDEN normalny byt
    jako kontrole i przechodzi - ale nie wszystkie normalne byty maja tak
    niskie liczby. Node 12 (kohorta i%6==0, razem z RING-owym node 0 i
    trzema normalnymi) w tym samym scenariuszu dostaje 187/200 falszywych
    flag, mimo ze jego wlasny poziom (uniform 0.5-2.0) jest zupelnie
    normalny. Przyczyna: przy n=5 czlonkow kohorty, mad miedzy-bytowych
    srednich jest bardzo NIESTABILNE (male-probkowy szum) - w tym
    konkretnym losowaniu 3 z 4 normalnych czlonkow wypadly przypadkiem
    blisko siebie (mad~0.09), wiec czwarty normalny czlonek (node 12),
    ktory po prostu wylosowal nizszy poziom w swoim wlasnym uniform(0.5,2.0),
    wyszedl pozornie "odstajacy" wzgledem tej przypadkowo ciasnej grupki -
    nie dlatego, ze cokolwiek zrobil nietypowego. To NIE jest kontaminacja
    przez RING (RING jest odrzucany przez mediane poprawnie) - to czysta
    niestabilnosc statystyki przy malej probie. Udokumentowane jako znane
    ograniczenie w README, nie naprawione tutaj (wymagaloby innej definicji
    mad kohorty, np. z polaczonych surowych odczytow zamiast median-of-means
    - patrz "Kierunki rozwoju")."""
    rng = np.random.default_rng(2)
    N = 30
    RING = [0, 1, 2, 3, 4]
    calib_values = {}
    for i in range(N):
        level = 9.0 if i in RING else rng.uniform(0.5, 2.0)
        calib_values[i] = level + rng.normal(0, 0.2, 200)

    cohort_of = {i: i % 6 for i in range(N)}
    baselines = cohort_baseline(calib_values, cohort_of)
    med, mad = baselines[12]
    idx, z, thr = core.anomalies(np.arange(200), calib_values[12], factor=3.0, baseline=(med, mad))
    assert len(idx) > 150  # reprodukuje znalezisko: normalny byt, prawie caly czas "zaflagowany"


def test_cohort_baseline_small_pure_cohort_gives_degenerate_mad_not_invisibility():
    """UWAGA - to NIE jest replika PEER_NB 1:1. Pierwotna hipoteza przy
    pisaniu tego testu byla taka sama jak PEER_NB w TIMDR-Crypto-Graph:
    "zla etykieta -> RING trafia do WLASNEJ kohorty -> RING niewidoczny".
    Zmierzone zachowanie jest INNE, i ciekawsze: gdy kohorta sklada sie
    (prawie) wylacznie z czlonkow RING, ich srednie kalibracyjne sa do
    siebie BARDZO podobne (male mad MIEDZY-bytowe), wiec baseline ma
    male mad - i to male mad, podzielone przez WLASNY, normalny szum
    odczytu (0.2), daje EKSTREMALNE z-score, nie male. RING zostaje
    zaflagowany (nie niewidoczny) - ale z powodu degenerowanego mad, nie
    dlatego, ze porownanie z populacja bylo poprawne. To jest ODREBNE,
    rowniez realne ryzyko cohort_baseline() przy MALYCH kohortach - patrz
    test nizej i README ("Ograniczenia")."""
    rng = np.random.default_rng(2)
    N = 30
    RING = [0, 1, 2, 3, 4]
    calib_values = {}
    for i in range(N):
        level = 9.0 if i in RING else rng.uniform(0.5, 2.0)
        calib_values[i] = level + rng.normal(0, 0.2, 200)

    hist_mean = {i: float(np.mean(calib_values[i])) for i in range(N)}
    cohort_of_leaky = {i: ("wysoki" if hist_mean[i] > 5.0 else "niski") for i in range(N)}
    assert set(i for i in range(N) if cohort_of_leaky[i] == "wysoki") == set(RING)

    baselines = cohort_baseline(calib_values, cohort_of_leaky)
    for i in RING:
        med, mad = baselines[i]
        assert mad < 0.05  # miedzy-bytowe mad w czysto-RING kohorcie jest male
        idx, z, thr = core.anomalies(np.arange(200), calib_values[i], factor=3.0, baseline=(med, mad))
        assert len(idx) > 150  # RING nadal zaflagowany - ale przez degenerowane mad


def test_defekt_baseline_spread_param_is_used():
    """defekt() ma ten sam baseline_spread= co anomalies() ma baseline=,
    ale (w odroznieniu od anomalies()) NIE ma dla niego osobnego testu
    na realistycznym scenariuszu typu "self kontra kalibracja" - to
    jest jedyny test pokrywajacy w ogole ten parametr, i sprawdza tylko,
    ze jest FAKTYCZNIE UZYWANY (mniejszy prog -> wiecej wykrytych skokow),
    nie ze rozwiazuje jakis konkretny, znaleziony problem. Patrz
    README "Ograniczenia" - to jest swiadomie oznaczone jako slabiej
    zweryfikowane niz baseline= w anomalies()."""
    s = np.concatenate([np.zeros(10), np.full(10, 1.0)])  # maly skok
    idx_self, _ = core.defekt(s, factor=0.3)  # rozrzut liczony z tego samego s (0..1) -> zwykle nie zlapie tak malego skoku wzgledem wlasnego rozrzutu
    idx_tight_baseline, _ = core.defekt(s, factor=0.3, baseline_spread=0.01)  # bardzo ciasny rozrzut z zewnatrz -> ten sam skok teraz przekracza prog
    assert len(idx_tight_baseline) >= len(idx_self)
    assert 10 in idx_tight_baseline


def test_analyze_multi_accepts_baselines_dict():
    """analyze_multi() faktycznie uzywa baselines= per parametr, nie
    tylko przyjmuje argument bez efektu."""
    rng = np.random.default_rng(3)
    t = np.arange(60)
    params = {
        "a": 50.0 + rng.normal(0, 1, 60),  # chronicznie podniesiony
        "b": rng.normal(0, 1, 60),          # normalny
    }
    baselines = {"a": baseline_from_calibration(rng.normal(0, 1, 200))}

    result_without = core.analyze_multi(t, params)
    result_with = core.analyze_multi(t, params, baselines=baselines)

    # bez baseline=, self-mediana normalizuje sie do wlasnego (podniesionego)
    # poziomu "a" - co najwyzej pojedyncze punkty wpadaja ponad prog przez
    # zwykly szum (MAD na n=60 nie jest idealnie stabilny), NIE caly odcinek
    assert len(result_without["anomaly_idx"]["a"]) < 5
    assert len(result_with["anomaly_idx"]["a"]) == 60
    # parametr bez baseline w slowniku dziala jak dotad (self)
    assert list(result_without["anomaly_idx"]["b"]) == list(result_with["anomaly_idx"]["b"])
