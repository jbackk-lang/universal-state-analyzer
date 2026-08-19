# timdr_core — uniwersalny silnik analizy stanu (anomalia/defekt/rezonans/skręt)

Domenowo niezależny silnik do wykrywania nietypowych odczytów w dowolnym
szeregu czasowym: `S(t)` — bez żadnych założeń o tym, co to za sygnał.
Wspólny rdzeń z projektami TIMDR już obecnymi w tym środowisku (pogoda —
Synoptyk-v2.0, finanse — `deliverable_timdr_finanse/timdr_core_finance.py`),
tu wydzielony jako osobny, przenośny pakiet.

Pierwszy dodatkowy przykład użycia: symulacje lattice QCD / masa glueballa
(`examples/accelerator/`).

## Co to robi

Cztery sygnały, generyczne dla każdej domeny:

- **anomalia** — pojedynczy odczyt poza statystyczną normą (z-score na
  medianie/MAD, z podłogą dla serii "prawie stałych", żeby uniknąć
  dzielenia przez ~0).
- **defekt** — nagły skok między kolejnymi odczytami tego samego parametru,
  próg liczony z rozrzutu p90-p10 tego samego okna.
- **rezonans** — kilka parametrów flaguje anomalię w tej samej chwili —
  silniejszy, bardziej wiarygodny sygnał niż pojedyncza anomalia.
- **skręt (twist)** — nagła zmiana kierunku lokalnego trendu (flow),
  liczona względem CZASU, nie indeksu próbki (żeby luki w danych nie dawały
  fałszywych alarmów).

Do tego: `rhythm()` (wykrywanie okresowości przez autokorelację, po
odjęciu trendu liniowego), `detect_jump()` + persystencja stanu na dysku
(wykrywanie skoku między kolejnymi uruchomieniami tego samego celu — stacja,
sensor, cokolwiek), i `compute_lead_bias()`/`badge()` (prosta, przejrzysta
korekta obciążenia z sparowanych (prognoza, rzeczywistość), z progiem
minimalnej liczby próbek i znacznikiem 🔴/🟠/🟢 pokazującym pewność).

## Struktura

```
timdr_core/
  core.py             — TIMDRCore: trm/flow/twist/anomalies/defekt/rhythm/rezonans/analyze_multi
  volatility.py        — detect_jump + load/save/clear_state (dysk, nie pamięć procesu)
  bias_correction.py   — compute_lead_bias/apply_bias_correction/badge
examples/accelerator/
  glueball_mass.py     — symulacja masy glueballa (Twój oryginalny skrypt)
  lattice_demo.py       — mini demo lattice QCD: Wilson loops, U(1) 4D, Metropolis, SU(3) mock (Twój oryginalny skrypt)
  analyze_trajectory.py — NOWE: podłącza timdr_core do trajektorii z powyższych dwóch skryptów
tests/
  test_core.py, test_volatility.py, test_bias_correction.py, test_accelerator_integration.py
```

## Użycie — dowolny projekt

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
```

Żadna z tych funkcji nie zakłada nazw kolumn ani jednostek — podłączasz
własne dane pod dowolnym kluczem w `params`.

## Przykład: akcelerator (lattice QCD / glueball)

```bash
cd examples/accelerator
python analyze_trajectory.py --T 40 --N 12
python analyze_trajectory.py --T 40 --N 12 --inject-anomaly-at 20   # demonstracja wykrywania
```

Zbiera 3 niezależne kanały (korelacja glueballa `C(t)`, plaquette Wilsona
U(1) 2D, mock "action" SU(3)) i analizuje je tym samym silnikiem.

**Ważne zastrzeżenie**: `glueball_mass.py` i `lattice_demo.py` to jawnie
uproszczone/mock symulacje (Twoje własne komentarze w kodzie: "su2_mock",
"nieortodoksyjny, ale wystarczy jako mock", "tylko symbolicznie") — poza
trybem `metropolis` każdy krok `t` losuje pole/macierze od nowa, niezależnie
od poprzedniego kroku (brak faktycznej ewolucji Monte Carlo sprzężonej
działaniem). To ma bezpośrednią konsekwencję dla wyników: `defekt`/`skręt`
odpalają się na niemal każdym kroku, bo dane rzeczywiście "skaczą" losowo
krok do kroku — to poprawny odczyt TEGO sygnału, nie błąd silnika. Podłączenie
do prawdziwie ewoluującego łańcucha (np. `metropolis`, gdzie kolejne stany
faktycznie wynikają jeden z drugiego) dałoby bardziej znaczące `defekt`.
Trzy kanały są też generowane z niezależnych wywołań RNG — `rezonans` w tym
przykładzie demonstruje MECHANIZM (czy silnik poprawnie łapie jednoczesne
odchylenia), nie odkrywa faktycznej korelacji fizycznej między nimi.

## Testy

```bash
pip install -r requirements.txt
pytest tests/ -q
```

38 testów: odporność na krótkie serie (n=0/1/2), pułapka zero-inflation
(MAD=0 / spread=0 → podłoga, nie crash/NaN), gradient liczony względem
czasu (nie indeksu) na danych z luką, wykrywanie wstrzykniętej anomalii/
skoku zarówno na syntetycznych danych, jak i na prawdziwej trajektorii z
`examples/accelerator/`.

## Czego to NIE jest

Nie jest to model uczenia maszynowego (bias-correction to zwykła średnia
błędu per grupa, nie trening) ani zwalidowane narzędzie fizyczne/finansowe —
wykrywa nietypowe odczyty w szeregu czasowym, nic więcej. Trafność każdego
sygnału (czy "anomalia" faktycznie znaczy coś ważnego w Twojej domenie)
zależy od tego, co podłączysz jako `params`, i wymaga własnej weryfikacji
względem rzeczywistości — dokładnie tak samo jak w Synoptyku (korekta
obciążenia aktywuje się dopiero po zebraniu sparowanych obserwacji
prognoza/rzeczywistość).
