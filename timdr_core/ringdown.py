"""
timdr_core/ringdown.py — czy powrót do równowagi jest REZONANSOWY (oscylacyjny)
================================================================================
Kontekst: `core.py::rezonans()` to licznik koincydencji (ile kanałów flaguje
anomalię w tej samej chwili) — nazwa pożyczona z fizyki, ale mechanizm inny.
Ten moduł liczy coś, co faktycznie odpowiada fizycznemu rezonansowi: po
zdarzeniu (anomalii/defekcie/twiście) sygnał wraca do poziomu odniesienia
(swojego "otoczenia"/stanu sprzed zaburzenia) — pytanie brzmi, CZY ten
powrót jest OSCYLACYJNY (układ "dzwoni" z powrotem do równowagi na
charakterystycznej częstotliwości, czyli rezonans/synchronizacja z
otoczeniem w sensie fizycznym: tłumiony oscylator) czy MONOTONICZNY (układ
przetłumiony — wraca bez dzwonienia, brak rezonansu do zaobserwowania).

Nie każdy powrót do równowagi jest rezonansem — to jest właśnie punkt
tego modułu. Rezonans to SPECYFICZNIE ten przypadek, gdy powrót ma
charakter oscylacyjny (przynajmniej jedno pełne przewahnięcie przez
poziom odniesienia w obie strony).

Metoda: przejścia przez zero (zero-crossing) względem poziomu odniesienia
+ logarithmic decrement między szczytami tego samego znaku — standardowa,
podręcznikowa technika inżynierska do szacowania częstotliwości i
tłumienia z REALNYCH, zaszumionych danych (używana np. przy szacowaniu
tłumienia konstrukcji, obwodów RLC). Świadomie NIE dopasowanie nieliniowe
(fit tłumionego oscylatora metodą najmniejszych kwadratów) — fit
nieliniowy jest numerycznie kruchy na krótkich, zaszumionych oknach
(łatwo o rozbieżność/lokalne minimum); zero-crossing + log-decrement jest
prostszy, stabilniejszy i równie standardowy.

Zweryfikowane numerycznie na syntetykach ze znanym parametrem (patrz
tests/test_ringdown.py): tłumiony oscylator o znanej częstotliwości/stałej
czasowej -> odzyskana częstotliwość i współczynnik tłumienia zgodne z
wartością teoretyczną (ζ = 1/sqrt((2π·f0·τ)² + 1)) w granicach tolerancji
szumu pomiarowego; czysty zanik wykładniczy (bez oscylacji) -> poprawnie
rozpoznany jako NIE-rezonansowy.
"""
from __future__ import annotations

import numpy as np


def ringdown_resonance(
    t,
    s,
    event_idx: int,
    baseline: float | None = None,
    pre_event_window: int = 10,
    max_lookahead: int | None = None,
    noise_floor_factor: float = 3.0,
) -> dict:
    """Analizuje powrót `s` do poziomu odniesienia PO indeksie `event_idx`.

    baseline: poziom odniesienia ("stan równowagi z otoczeniem"). Jeśli
        None, liczony jako średnia `s` z `pre_event_window` próbek PRZED
        zdarzeniem (stan sprzed zaburzenia) — ten sam duch co
        `baseline_from_calibration()` w baseline.py, tylko lokalnie, z
        historii tego samego kanału tuż przed zdarzeniem.
    pre_event_window: ile próbek przed zdarzeniem użyć do wyliczenia
        baseline ORAZ poziomu szumu (patrz noise_floor_factor), gdy nie
        podano ich jawnie.
    max_lookahead: ile próbek PO zdarzeniu analizować (None = do końca
        serii) — ogranicza okno, żeby nie złapać zupełnie innego,
        późniejszego zdarzenia jako część tego samego "powrotu".
    noise_floor_factor: PIERWOTNA WERSJA tej funkcji (bez tego progu) na
        prawdziwie zaszumionych danych myliła szum wokół baseline z
        oscylacją - zweryfikowano to na syntetycznym, czysto wykładniczym
        (nieoscylacyjnym) zaniku z szumem: dawał fałszywie
        `is_oscillatory=True` z kilkudziesięcioma "przejściami" będącymi
        czystym szumem w ogonie sygnału (patrz test_ringdown.py,
        `test_bug_niefiltrowany_szum_dawal_falszywy_rezonans`). Naprawa:
        próg szumu = `noise_floor_factor * std(s[pre_event_window])` -
        szczyty poniżej tego progu (i wszystko po pierwszym takim szczycie)
        są ODRZUCANE z analizy oscylacyjności/częstotliwości/tłumienia,
        zanim zdążą wygenerować fałszywe przejścia przez zero.

    DRUGI, ODDZIELNY błąd znaleziony przy portowaniu tej funkcji do sygnału
    o wysokiej częstotliwości próbkowania względem poziomu szumu (sieć
    energetyczna, ~1000 próbek/s): PRAWDZIWE przejście przez zero też
    generowało kilka-kilkanaście "przejść" z rzędu, bo próbka szumu tuż
    PRZY samym przejściu (gdzie sygnał i tak jest bliski zeru) potrafi
    kilkukrotnie zmienić znak, zanim sygnał wyraźnie odejdzie na nową
    stronę ("drganie"/chatter, dokładnie ten sam problem co w realnych
    komparatorach analogowych). Naprawa: histereza metodą Schmitta
    zastosowana NA WYKRYWANIU STANU (nie doklejona po fakcie do już
    policzonych szczytów) — stan HIGH/LOW jest "potwierdzany" dopiero gdy
    |sygnał - baseline| > noise_floor, a przejście liczy się dopiero przy
    faktycznym przełączeniu na przeciwny, potwierdzony stan. Próbki w
    paśmie ±noise_floor (chatter przy prawdziwym przejściu ORAZ szum w
    zanikłym ogonie sygnału - oba błędy tym samym mechanizmem) nigdy nie
    potwierdzają nowego stanu, więc żaden z nich nie generuje fałszywego
    przejścia. (Pierwsza próba naprawy robiła to po fakcie, osobnym,
    luźniejszym progiem histerezy — to dawało obciążenie doboru:
    "przetrwałe" szczyty były systematycznie zawyżone, część z nich
    przypadkiem przekraczała potem próg odcięcia ogona i dawała fałszywe
    is_oscillatory=True na czysto monotonicznym zaniku; naprawione przez
    przejście na jeden, wspólny próg na poziomie stanu, patrz historia
    tego pliku.)

    Zwraca dict:
      baseline, noise_floor, is_oscillatory (bool),
      n_crossings, n_peaks_used (ile przejść/szczytów POWYŻEJ progu szumu
          wzięto pod uwagę - to jest "trusted" podzbiór, nie surowa liczba
          wszystkich przejść w oknie),
      period_s, frequency_hz (None jeśli nieoscylacyjny lub za mało przejść),
      log_decrement, damping_ratio (None jeśli nie da się policzyć —
          potrzeba >=2 szczytów TEGO SAMEGO znaku, czyli >=1.5 okresu),
      peak_times, peak_amplitudes (TYLKO szczyty powyżej progu szumu, do
          wykresu/diagnostyki).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    n = len(s)
    if n == 0 or not (0 <= event_idx < n):
        raise ValueError(f"event_idx={event_idx} poza zakresem serii o długości {n}")

    pre_start = max(0, event_idx - pre_event_window)
    pre_samples = s[pre_start:event_idx]

    if baseline is None:
        baseline = float(np.mean(pre_samples)) if len(pre_samples) else float(s[event_idx])

    # Poziom szumu z historii PRZED zdarzeniem - jeśli jej brak (event_idx=0,
    # nic wcześniej), nie da się go wiarygodnie oszacować; floor=0 (brak
    # filtrowania) w takim przypadku jest udokumentowanym ograniczeniem, nie
    # cichym zgadywaniem.
    noise_std = float(np.std(pre_samples)) if len(pre_samples) >= 2 else 0.0
    noise_floor = noise_floor_factor * noise_std

    end = n if max_lookahead is None else min(n, event_idx + max_lookahead)
    t_post = t[event_idx:end]
    d = s[event_idx:end] - baseline

    result: dict = {
        "baseline": float(baseline),
        "noise_floor": float(noise_floor),
        "is_oscillatory": False,
        "n_crossings": 0,
        "n_peaks_used": 0,
        "period_s": None,
        "frequency_hz": None,
        "log_decrement": None,
        "damping_ratio": None,
        "peak_times": [],
        "peak_amplitudes": [],
    }

    if len(d) < 3:
        return result

    # --- histereza Schmitta NA STANIE, nie doklejona po fakcie do już
    # policzonych szczytów: stan HIGH/LOW jest "potwierdzany" dopiero gdy
    # |d| > noise_floor; przejście zapisujemy dopiero gdy stan faktycznie
    # PRZEŁĄCZY się na przeciwny, potwierdzony stan. Próbki w paśmie
    # [-noise_floor, noise_floor] nigdy nie potwierdzają nowego stanu, więc
    # nie generują fałszywych przejść — to standardowy komparator z
    # histerezą (Schmitt trigger) zastosowany wprost do detekcji stanu.
    # (Wcześniejsza wersja robiła to po fakcie, osobnym progiem histerezy
    # luźniejszym niż noise_floor — to wprowadzało obciążenie doboru:
    # "przetrwałe" szczyty były systematycznie zawyżone i część z nich
    # przypadkiem przekraczała potem próg odcięcia ogona, dając fałszywe
    # is_oscillatory=True. Jeden próg dla obu ról usuwa tę asymetrię.)
    band = noise_floor
    confirmed_idx: list[int] = []
    state = 0
    for i in range(len(d)):
        if d[i] > band:
            new_state = 1
        elif d[i] < -band:
            new_state = -1
        else:
            continue
        if new_state != state:
            confirmed_idx.append(i)
            state = new_state

    # przejścia = surowy moment zmiany znaku d (interpolowany), leżący
    # MIĘDZY dwoma kolejnymi potwierdzonymi punktami przeciwnego stanu — to
    # on odpowiada faktycznej chwili, w której sygnał zaczął zmieniać
    # stronę (potwierdzenie histerezą przychodzi chwilę później, gdy sygnał
    # wyraźnie odjedzie od zera).
    crossing_times: list[float] = []
    for prev_i, cur_i in zip(confirmed_idx[:-1], confirmed_idx[1:]):
        found = None
        for k in range(prev_i, cur_i):
            if d[k] == 0 or (d[k] > 0) != (d[k + 1] > 0):
                frac = 0.0 if d[k] == 0 else -d[k] / (d[k + 1] - d[k])
                found = float(t_post[k] + frac * (t_post[k + 1] - t_post[k]))
                break
        if found is None:
            found = float((t_post[prev_i] + t_post[cur_i]) / 2.0)
        crossing_times.append(found)

    # szczyty: lokalne ekstremum |d| w segmentach ograniczonych
    # potwierdzonymi punktami stanu (nie surowymi przejściami) — każdy
    # segment z definicji zawiera punkt przekraczający próg szumu, więc
    # każdy zwrócony szczyt jest już "zaufany" (>= noise_floor); osobne
    # obcinanie ogona po fakcie nie jest już potrzebne.
    # (deduplikacja: gdy sygnał PRZEKRACZA próg już w pierwszej/ostatniej
    # próbce okna, confirmed_idx[0]/[-1] pokrywa się z granicą 0/len(d)-1 -
    # bez `set()` dawałoby to zdegenerowany, jednopunktowy segment i
    # podwójnie liczony ten sam fizyczny szczyt, patrz historia tego pliku)
    bounds_idx = sorted(set([0] + confirmed_idx + [len(d) - 1]))
    peak_times: list[float] = []
    peak_amps: list[float] = []
    for a, b in zip(bounds_idx[:-1], bounds_idx[1:]):
        if b < a:
            continue
        seg = d[a:b + 1]
        local_idx = int(np.argmax(np.abs(seg)))
        peak_times.append(float(t_post[a + local_idx]))
        peak_amps.append(float(seg[local_idx]))

    used_crossings = crossing_times

    result["n_crossings"] = len(used_crossings)
    result["n_peaks_used"] = len(peak_amps)
    result["peak_times"] = peak_times
    result["peak_amplitudes"] = peak_amps

    # Oscylacyjny = przynajmniej 2 (zaufane) przejścia przez baseline I
    # przynajmniej 2 (zaufane) szczyty naprzemiennych znaków - sygnał
    # realnie "przewahnął" przez poziom odniesienia w obie strony, ponad
    # poziom szumu, nie tylko go raz musnął (co równie dobrze mogłoby być
    # szumem na monotonicznym powrocie).
    if len(used_crossings) >= 2 and len(peak_amps) >= 2:
        result["is_oscillatory"] = True

        # mediana, nie średnia: ostatni(e) potwierdzony(e) półokres(y) bywa(ją)
        # tuż nad progiem szumu (amplituda oscylacji zdążyła już mocno
        # zaniknąć) - jego dokładny czas potwierdzenia jest wtedy niepewny
        # (pojedyncza próbka szumu decyduje o momencie przekroczenia progu),
        # co potrafi go skrócić/wydłużyć i wypaczyć ŚREDNIĄ różnicę między
        # przejściami. Mediana jest odporna na taki pojedynczy zanieczyszczony
        # półokres bez potrzeby osobnego, ręcznie dobieranego progu odcięcia
        # (zaobserwowane empirycznie przy porcie do TIMDR-Grid-Monitor, fs=1000Hz:
        # bez mediany częstotliwość wychodziła zawyżona o ~20%).
        crossing_diffs = np.diff(used_crossings)
        if len(crossing_diffs) and np.median(crossing_diffs) > 0:
            period = 2.0 * float(np.median(crossing_diffs))  # przejście 2x na okres
            result["period_s"] = period
            result["frequency_hz"] = 1.0 / period

        # logarithmic decrement: między szczytami TEGO SAMEGO znaku,
        # oddalonymi o jeden pełny okres (peak[i] vs peak[i+2])
        log_ratios = []
        for i in range(len(peak_amps) - 2):
            a, b = peak_amps[i], peak_amps[i + 2]
            if np.sign(a) == np.sign(b) and a != 0 and b != 0:
                ratio = abs(a) / abs(b)
                if ratio > 0:
                    log_ratios.append(np.log(ratio))
        if log_ratios:
            delta = float(np.mean(log_ratios))
            result["log_decrement"] = delta
            result["damping_ratio"] = float(delta / np.sqrt(4 * np.pi ** 2 + delta ** 2))

    return result
