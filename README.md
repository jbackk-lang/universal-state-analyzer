# timdr_core — uniwersalny silnik analizy stanu (anomalia/defekt/rezonans/skręt)

Domenowo niezależny silnik do wykrywania nietypowych odczytów w dowolnym
szeregu czasowym `S(t)` — bez żadnych założeń o tym, co to za sygnał.

## Aktualny stan

Pięć operatorów, wszystkie w `timdr_core/core.py`, wszystkie działające na
gołych tablicach `(t, s)`:

| Operator | Co robi | Funkcja |
|---|---|---|
| TRM | Mediana krocząca (wygładzenie szumu) | `trm()` |
| FLOW | Lokalny gradient, liczony WZGLĘDEM CZASU (LSQ na oknie), nie indeksu próbki | `flow()` |
| TWIST | Nagła zmiana kierunku FLOW, różniczkowana względem czasu | `twist()` |
| ANOMALIE | MAD-owy z-score względem mediany, z podłogą na serie "prawie stałe" | `anomalies()` |
| REZONANS | Licznik: ile kanałów flaguje `anomalies()` w tej samej chwili | `rezonans()` |

Do tego osobna funkcja `defekt()` (nagły skok między dwoma bezpośrednio
kolejnymi odczytami, próg z rozrzutu p90-p10 tego samego okna — inny sygnał
niż `anomalies()`, nie synonim), `rhythm()` (autokorelacja na wartości ze
znakiem po odjęciu trendu liniowego) i pipeline `analyze_multi()` spinający
wszystko dla wielu kanałów naraz.

`REZONANS` jest licznikiem koincydencji (ile kanałów flaguje jednocześnie),
NIE jest sumowany do żadnego wyniku anomalii — świadoma decyzja, bo w innych
projektach tej rodziny (TIMDR-Quantum-Lattice, TIMDR-Earthquake-Core,
TIMDR-Crypto-Graph) dodawanie rezonansu/kompozytowego wskaźnika do wyniku
predykcyjnego za każdym razem pogarszało wynik.

**UWAGA na nazewnictwo:** `REZONANS`/`rezonans()` powyżej to licznik
koincydencji — nazwa pożyczona z fizyki, ale mechanizm inny (zgodność
WIELU kanałów W TEJ SAMEJ CHWILI, nie zjawisko oscylacyjne). Fizyczny
rezonans — układ wracający do stanu równowagi z otoczeniem przez
oscylacyjne "dzwonienie" na charakterystycznej częstotliwości — jest
osobno w `timdr_core/ringdown.py::ringdown_resonance()`, patrz niżej.

### RINGDOWN — czy powrót do równowagi PO zdarzeniu jest oscylacyjny

`ringdown_resonance(t, s, event_idx, ...)` (i wygodne opakowanie
`TIMDRCore.analyze_ringdown(t, s, event_indices, ...)`) analizuje, co się
dzieje z kanałem PO zdarzeniu (anomalii/defekcie/twiście, znalezionym np.
przez `analyze_multi()`): czy powrót do poziomu sprzed zaburzenia jest
OSCYLACYJNY (sygnał przewahnął przez ten poziom w obie strony — prawdziwy
rezonans/synchronizacja z otoczeniem w sensie fizycznym) czy MONOTONICZNY
(przetłumiony powrót — brak rezonansu do zaobserwowania). Nie każdy powrót
do równowagi jest rezonansem; to rozróżnienie jest sensem tej funkcji.

Metoda: histereza Schmitta NA WYKRYWANIU STANU (stan HIGH/LOW potwierdzany
dopiero gdy |sygnał - baseline| > `noise_floor_factor × std(szum przed
zdarzeniem)`, przejście liczone dopiero przy faktycznym przełączeniu na
przeciwny, potwierdzony stan) + interpolowane przejścia przez poziom
odniesienia + logarithmic decrement między szczytami tego samego znaku —
standardowa, podręcznikowa technika inżynierska szacowania
częstotliwości/tłumienia z realnych danych (nie dopasowanie nieliniowe —
za kruche numerycznie na krótkich, zaszumionych oknach).

**Zweryfikowane liczbowo, nie tylko "wygląda sensownie"** (`tests/test_ringdown.py`,
13 testów): tłumiony oscylator o ZNANEJ częstotliwości f0=1.5Hz i stałej
czasowej τ=1.2s → odzyskana częstotliwość 1,498Hz (błąd 0,15%), odzyskany
współczynnik tłumienia 0,075 vs teoretyczny 0,088 (wzór
ζ=1/√((2π·f0·τ)²+1), błąd w granicach tolerancji testu rel=0.3); stabilne
na 5 różnych realizacjach szumu. Czysty zanik wykładniczy (bez oscylacji)
poprawnie rozpoznany jako `is_oscillatory=False`.

**Trzy błędy znalezione i naprawione przy budowie** (w kolejności, w jakiej
wypłynęły — każdy następny ujawnił się dopiero po naprawie poprzedniego):

1. *Szum w ogonie brany za oscylację.* Pierwsza wersja (bez progu szumu) na
   zaszumionym, nieoscylacyjnym zaniku dawała fałszywie
   `is_oscillatory=True` — szum w ogonie sygnału (gdzie prawdziwa amplituda
   już wygasła) generował dziesiątki przypadkowych przejść przez poziom
   odniesienia. Naprawa: próg szumu `noise_floor_factor × std(...)`.

2. *Drganie ("chatter") tuż przy prawdziwym przejściu, przy wysokiej
   częstotliwości próbkowania.* Wykryte przy portowaniu do TIMDR-Grid-Monitor
   (sieć energetyczna, ~1000 próbek/s): PRAWDZIWE przejście przez zero też
   generowało kilkanaście-kilkaset "przejść" z rzędu, bo próbka szumu tuż
   przy samym przejściu potrafi kilkukrotnie zmienić znak, zanim sygnał
   wyraźnie odjedzie na nową stronę. Pierwsza próba naprawy (doklejona po
   fakcie do już policzonych szczytów, osobnym, luźniejszym progiem)
   wprowadziła REGRESJĘ na błędzie #1 (obciążenie doboru zawyżało
   "przetrwałe" szczyty). Ostateczna naprawa: histereza Schmitta zastosowana
   wprost NA WYKRYWANIU STANU, jednym wspólnym progiem (`noise_floor`) dla
   obu ról — usuwa oba błędy tym samym mechanizmem.
3. *Zdegenerowany, podwójnie liczony pierwszy/ostatni szczyt* — gdy sygnał
   przekracza próg już w pierwszej/ostatniej próbce okna, granica okna i
   punkt potwierdzenia stanu pokrywały się, dając dwa niemal identyczne
   "szczyty" w tym samym miejscu (widoczne jako zawyżona o ~20% odzyskana
   częstotliwość na teście specyficznym dla sieci energetycznej — patrz
   TIMDR-Grid-Monitor). Naprawa: deduplikacja granic segmentów.

Dodatkowo: częstotliwość liczona jest z **mediany**, nie średniej, odstępów
między przejściami — ostatni potwierdzony półokres bywa tuż nad progiem
szumu (niepewny dokładny czas), a mediana jest odporna na taki pojedynczy
zanieczyszczony półokres bez osobnego progu odcięcia.

**Ograniczenie:** `noise_floor_factor=3.0` to wartość ustalona ręcznie,
zweryfikowana wyłącznie na syntetykach z tego repo — nie skalibrowana na
żadnych realnych danych. Bez historii przed zdarzeniem (`event_idx=0`)
nie da się oszacować szumu — `noise_floor=0` (brak filtrowania) w takim
przypadku, udokumentowane, nie ciche zgadywanie.

Dodatkowe moduły:
- `timdr_core/volatility.py` — `detect_jump()` + trwały stan między
  uruchomieniami na dysku (nie w pamięci procesu — restart procesu inaczej
  cicho zeruje historię).
- `timdr_core/bias_correction.py` — prosta korekta obciążenia z par
  (prognoza, rzeczywistość), grupowana po `lead`; to nie jest model uczenia
  maszynowego, tylko średni błąd per grupa z progiem minimalnej liczby próbek.
- `timdr_core/baseline.py` — `baseline_from_calibration()` i
  `cohort_baseline()`: pozwalają `anomalies()`/`defekt()` liczyć punkt
  odniesienia z OSOBNEGO okresu kalibracji albo z kohorty, zamiast domyślnie
  z tego samego okna, które jest oceniane. Patrz "Ograniczenia" niżej po to,
  jaki dokładnie problem to rozwiązuje i czego nie rozwiązuje.
- `timdr_core/trigger.py` — **czujnik sygnałowy** (NIE model, NIE
  predyktor): `TIMDRTrigger`, dispatcher nad `analyze_multi()` — mówi
  który typ zdarzenia się odpalił, w którym kanale i gdzie: `RESONANCE`
  (>=rezonans_min kanałów naraz) > `STRUCTURE` (twist w dowolnym kanale) >
  `DEFEKT` (nagły skok w dowolnym kanale) > `SCALE` (pojedyncza anomalia)
  > `NONE`. Sam nie liczy statystyki, tylko woła już przetestowany
  pipeline. Wpięty do `examples/accelerator/analyze_trajectory.py`. Testy:
  `tests/test_trigger.py` (66/66 łącznie z resztą).

  ```python
  from timdr_core import TIMDRTrigger

  trigger = TIMDRTrigger(rezonans_min=3)
  result = trigger.analyze(t, params)
  print(result.trigger_type, result.location, result.channel, result.message)
  ```

Przykład domeny spoza pogody/finansów: `examples/accelerator/` — analiza
trajektorii z (jawnie uproszczonej, patrz zastrzeżenie niżej) symulacji
lattice QCD / masy glueballa, tym samym silnikiem.

**Testy: 59, wszystkie przechodzą** (`pytest tests/ -q`) — brzegowe
przypadki n=0/1/2, podłoga na zero-inflation (MAD=0/rozrzut=0), gradient
liczony względem czasu (nie indeksu) na danych z luką, wykrywanie
wstrzykniętej anomalii/skoku na syntetykach i na nie-mockowanej ścieżce
integracyjnej z przykładu akceleratora (`test_accelerator_integration.py`),
`test_baseline.py` (8 testów) opisany w sekcji "Ograniczenia", oraz
`test_ringdown.py` (13 testów) — walidacja `ringdown_resonance()` na
tłumionym oscylatorze o ZNANEJ częstotliwości/tłumieniu, patrz sekcja
RINGDOWN wyżej.

### Użycie

```python
import numpy as np
from timdr_core import TIMDRCore

core = TIMDRCore()
t = np.arange(100)
params = {"kanal_a": ..., "kanal_b": ..., "kanal_c": ...}  # dowolne tablice tej samej długości

result = core.analyze_multi(t, params, rezonans_min=3)
result["anomaly_idx"]["kanal_a"]   # indeksy t z anomalią w kanale a
result["defekt_idx"]["kanal_a"]    # indeksy nagłych skoków
result["rezonans_idx"]              # indeksy, gdzie >=3 kanały naraz flagują anomalię

# Rezonans w sensie fizycznym (oscylacyjny powrót do równowagi) - osobno:
ring = core.analyze_ringdown(t, params["kanal_a"], result["defekt_idx"]["kanal_a"])
ring[0]["is_oscillatory"]           # czy powrót po pierwszym wykrytym skoku "dzwoni"
ring[0]["frequency_hz"]             # jeśli tak - na jakiej częstotliwości
```

Z odniesieniem do osobnego okresu kalibracji zamiast tego samego okna
(patrz "Ograniczenia" — dlaczego to bywa potrzebne):

```python
from timdr_core.baseline import baseline_from_calibration

baseline = baseline_from_calibration(dane_z_okresu_bez_anomalii["kanal_a"])
result = core.analyze_multi(t, params, baselines={"kanal_a": baseline})
```

### Uruchomienie

```
pip install -r requirements.txt
pytest tests/ -q
cd examples/accelerator && python analyze_trajectory.py --T 40 --N 12 --inject-anomaly-at 20
```

### Struktura

```
timdr_core/
  core.py             — TIMDRCore: trm/flow/twist/anomalies/defekt/rhythm/rezonans/analyze_multi/analyze_ringdown
  ringdown.py          — ringdown_resonance(): rezonans w sensie fizycznym (oscylacyjny powrót do równowagi)
  baseline.py          — baseline_from_calibration/cohort_baseline
  volatility.py        — detect_jump + load/save/clear_state (dysk, nie pamięć procesu)
  bias_correction.py   — compute_lead_bias/apply_bias_correction/badge
examples/accelerator/
  glueball_mass.py     — symulacja masy glueballa (oryginalny skrypt)
  lattice_demo.py       — mini demo lattice QCD: Wilson loops, U(1) 4D, Metropolis, SU(3) mock (oryginalny skrypt)
  analyze_trajectory.py — podłącza timdr_core do trajektorii z powyższych dwóch skryptów
tests/
  test_core.py, test_baseline.py, test_volatility.py, test_bias_correction.py, test_accelerator_integration.py, test_ringdown.py
```

## Ograniczenia

**Ślepy punkt self-baseline (bez `baseline=`).** Domyślnie `anomalies()`/
`defekt()` liczą medianę/MAD/rozrzut z TEGO SAMEGO okna `s`, które jest
oceniane. Sygnał nietypowy PRZEZ CAŁE obserwowane okno (bez wcześniejszego,
"normalnego" fragmentu w danych) wychodzi jako statystycznie normalny — bo
nie ma z czym go porównać. To ten sam błąd, który w `TIMDR-Crypto-Graph`
nazwano "ślepym punktem self-eq", tu zreplikowany i potwierdzony w
`test_baseline.py::test_self_baseline_blind_to_signal_chronically_shifted_for_whole_window`.

**`baseline_from_calibration()` nie pomaga, jeśli zmiana zaszła już w
okresie kalibracji.** Jeśli byt jest chronicznie inny od samego początku
danych, jakie masz (nie ma w historii momentu "przed"), własna kalibracja
tego bytu jest tak samo ślepa jak self-baseline — zweryfikowane w
`test_baseline_from_calibration_does_not_help_if_shift_already_in_calibration`.
Jedyne rozwiązanie tego konkretnego przypadku, jakie tu jest, to
`cohort_baseline()` (odniesienie do INNYCH bytów, nie do własnej historii).

**`cohort_baseline()` wymaga etykiety kohorty niezależnej od ocenianego
sygnału — i nawet wtedy ma dwa udokumentowane, zweryfikowane ryzyka:**
- Zła etykieta (zbudowana z samego ocenianego sygnału, np. próg na
  poziomie samej wartości) NIE daje "niewidocznego" bytu, jak można by
  się spodziewać po analogii z pułapką sąsiadów-w-grafie z
  `TIMDR-Crypto-Graph` — daje DEGENEROWANE (bardzo małe) mad kohorty, bo
  grupa złożona wyłącznie z podobnych do siebie bytów ma małe
  między-bytowe zróżnicowanie. Byt nadal wychodzi zaflagowany, ale przez
  artefakt statystyczny, nie przez poprawne porównanie z populacją
  (`test_cohort_baseline_small_pure_cohort_gives_degenerate_mad_not_invisibility`).
- Nawet przy DOBRZE dobranej etykiecie, mała liczba członków kohorty
  (sprawdzone przy n=5) daje niestabilne mad między-bytowych średnich —
  zwykły normalny byt, który przez przypadek wylosował wartość odstającą
  od reszty swojej małej kohorty, może dostać dziesiątki-setki
  fałszywych flag, mimo że nic nietypowego nie robi
  (`test_cohort_baseline_small_cohort_can_false_flag_a_normal_member`,
  zaobserwowane: 187/200 fałszywych flag dla jednego normalnego bytu w
  kohorcie liczącej 5 członków).

**Brak automatycznego wykrywania kohort.** Świadomie — automatyczne
klastrowanie na cechach ma w `TIMDR-Crypto-Graph` udokumentowany,
zweryfikowany tryb awarii (odtwarza pułapkę kolegujących się bytów przez
inne drzwi). `cohort_of` tutaj musi dostarczyć wywołujący, z etykiety spoza
ocenianego sygnału.

**`defekt(baseline_spread=...)` jest słabiej zweryfikowany niż
`anomalies(baseline=...)`.** Ma jeden test (`test_defekt_baseline_spread_param_is_used`),
sprawdzający tylko, że parametr jest faktycznie używany — nie ma dla niego
osobnego testu na realistycznym scenariuszu w stylu self-vs-kalibracja.

**`examples/accelerator/` to jawnie uproszczona/mock symulacja**, nie
zwalidowane obliczenie lattice QCD — poza trybem `metropolis` każdy krok
losuje pole/macierze od nowa, niezależnie od poprzedniego (brak faktycznej
ewolucji Monte Carlo sprzężonej działaniem). `defekt`/`twist` odpalają się
niemal na każdym kroku, bo dane rzeczywiście skaczą losowo krok do kroku —
to poprawny odczyt tego sygnału, nie błąd silnika, ale też nie dowód, że
silnik radzi sobie z prawdziwą fizyczną trajektorią.

**Brak walidacji na jakichkolwiek realnych (nie-syntetycznych,
nie-mockowych) danych w tym repozytorium.** Wszystkie testy — łącznie z
`test_baseline.py` — działają na syntetycznie wygenerowanych szeregach.
Repozytoria, z których ten rdzeń został wydestylowany (Synoptyk-v2.0 —
realne dane Open-Meteo; `timdr_core_finance.py` — świece OHLCV), mają
walidację na realnych danych, ale ten pakiet, w obecnej formie, jej nie ma.

**`examples/accelerator/analyze_trajectory.py` nie korzysta jeszcze z
`baseline=`/`baselines=`** — używa wyłącznie domyślnego trybu self, mimo
że mechanizm kalibracji jest już dostępny w `analyze_multi()`.

**`ringdown_resonance()` zwalidowany wyłącznie na syntetycznym, czystym
modelu tłumionego oscylatora.** `noise_floor_factor=3.0` (próg odcięcia
szumu) jest ustalony ręcznie, nie skalibrowany na realnych danych. Metoda
zero-crossing/log-decrement zakłada dobrze odseparowane, wyraźne szczyty —
na sygnale z kilkoma nakładającymi się częstotliwościami naraz
(polirytmia) da prawdopodobnie mylącą, uśrednioną częstotliwość zamiast
błędu — to nie zostało przetestowane.

**Nie jest to model uczenia maszynowego** (`bias_correction` to zwykła
średnia błędu per grupa, nie trening) ani zwalidowane narzędzie
fizyczne/finansowe — wykrywa nietypowe odczyty w szeregu czasowym, nic
więcej. Trafność każdego sygnału (czy "anomalia" faktycznie znaczy coś
ważnego w Twojej domenie) zależy od tego, co podłączysz jako `params`, i
wymaga własnej weryfikacji względem rzeczywistości.

## Kierunki rozwoju

- **Mad kohorty odporne na małą próbę.** Obecne `cohort_baseline()` liczy
  mad z median-of-means (jedna liczba na byt) — niestabilne przy małych
  kohortach (patrz "Ograniczenia"). Kandydat: liczyć rozrzut z połączonych
  surowych odczytów całej kohorty względem jej mediany, zamiast z median
  poszczególnych bytów, i/lub wymusić minimalną liczbę członków z
  jawnym ostrzeżeniem poniżej progu.
- **Wariant GLOBAL.** `TIMDR-Crypto-Graph` testował trzeci punkt
  odniesienia — medianę CAŁEJ populacji, bez podziału na kohorty —
  jako pośredni krok między self a cohort (łapie chroniczne przypadki
  jak cohort, ale z realnym kosztem fałszywych alarmów na legalnie
  odmiennych podgrupach). Tu jeszcze nie przeniesiony.
- **Walidacja na realnych danych.** Podłączyć `timdr_core` pod co
  najmniej jedno źródło danych spoza syntetyków/mocków (np. te same dane
  Open-Meteo co Synoptyk-v2.0, albo świece OHLCV) i sprawdzić, czy
  `anomalies()`/`baseline=`/`cohort_baseline()` dają się sensownie
  skalibrować poza kontrolowanym eksperymentem.
- **Podłączyć `baselines=` do przykładu akceleratora**, żeby
  `examples/accelerator/` demonstrował nie tylko self-mode.
- **Automatyczne wykrywanie okresu kalibracji.** Obecnie użytkownik sam
  musi wskazać, który fragment danych jest wolny od badanej anomalii —
  brak jakiejkolwiek heurystyki wspomagającej ten wybór.
