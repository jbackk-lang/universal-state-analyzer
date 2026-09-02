"""
test_trigger.py — testy timdr_core/trigger.py (TIMDRTrigger).

Ten plik NIE re-weryfikuje matematyki TIMDRCore.anomalies()/defekt()/
twist()/rezonans() (już przetestowane w test_core.py) - to nie jest
robota dispatchera. Dwa rodzaje testów:

1. test_resonance_na_realnych_kanalach - JEDEN test integracyjny na
   prawdziwym TIMDRCore (bez mockowania), z recznie wyprowadzonymi z-score
   (ten sam wzorzec MAD=0 -> fallback std, ktory jest juz zweryfikowany w
   deliverable_timdr_finanse/test_timdr_finance_trigger.py - tu te same
   trzy szeregi liczbowe, inna domena/nazewnictwo kanalow).
2. Reszta testow wstrzykuje falszywy `core` (stub zwracajacy ustalony
   slownik, ta sama struktura co TIMDRCore.analyze_multi()) - testujemy
   WYLACZNIE logike priorytetow/mapowania dispatchera.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from timdr_core.trigger import TIMDRTrigger, SignalTriggerType


# ----------------------------------------------------------------------
# 1) Test integracyjny na realnych (recznie wyprowadzonych) kanalach
# ----------------------------------------------------------------------

def test_resonance_na_realnych_kanalach():
    """
    3 kanaly x 12 probek, wszystkie plaskie oprocz WSPOLNEGO skoku w idx=6:
      chan_a: 100 wszedzie, 130 w idx6 -> mad_raw=0 -> fallback std=8.2916,
              z(idx6)=30/8.2916=3.618 > 3.0
      chan_b: 10 wszedzie, 500 w idx6  -> fallback std=135.43,
              z(idx6)=449.167/135.43=3.317 > 3.0
      chan_c: 0 wszedzie, 500 w idx6   -> fallback std=138.19,
              z(idx6)=458.333/138.19=3.317 > 3.0
    (identyczna arytmetyka jak w deliverable_timdr_finanse/
    test_timdr_finance_trigger.py::test_resonance_wins_na_realnych_swiecach,
    ten sam ksztalt danych, inne nazwy kanalow.)
    Wszystkie 3 kanaly anomalne w idx6 = rezonans_min (domyslnie 3) ->
    RESONANCE w lokalizacji 6, niezaleznie od tego, co twist()/defekt()
    tam zglaszaja.
    """
    n = 12
    t = list(range(n))
    chan_a = [100.0] * n
    chan_b = [10.0] * n
    chan_c = [0.0] * n
    chan_a[6] = 130.0
    chan_b[6] = 500.0
    chan_c[6] = 500.0

    trigger = TIMDRTrigger()
    result = trigger.analyze(t, {"chan_a": chan_a, "chan_b": chan_b, "chan_c": chan_c})

    assert result.triggered is True
    assert result.trigger_type == SignalTriggerType.RESONANCE
    assert result.location == 6
    assert result.channel is None  # RESONANCE nie wskazuje pojedynczego kanalu
    assert "3" in result.message


# ----------------------------------------------------------------------
# 2) Testy priorytetow/mapowania z wstrzyknietym core (stub)
# ----------------------------------------------------------------------

class _FakeCore:
    """Stub o tym samym kontrakcie co TIMDRCore.analyze_multi(): zwraca
    ustalony słownik niezależnie od danych wejściowych."""

    def __init__(self, result_dict):
        self._result = result_dict

    def analyze_multi(self, *args, **kwargs):
        return self._result


def _empty_result(**overrides):
    base = dict(
        anomaly_idx={}, defekt_idx={}, twist_idx={},
        rezonans_idx=[], rezonans_counts=[0] * 20,
    )
    base.update(overrides)
    return base


def _dummy_args():
    n = 5
    return (list(range(n)), {"x": [0.0] * n})


def test_priorytet_resonance_nad_wszystkim():
    counts = [0] * 20
    counts[8] = 4
    fake = _FakeCore(_empty_result(
        rezonans_idx=[8], rezonans_counts=counts,
        twist_idx={"x": [3]}, defekt_idx={"x": [5]}, anomaly_idx={"x": [1]},
    ))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.trigger_type == SignalTriggerType.RESONANCE
    assert result.location == 8
    assert "4" in result.message


def test_priorytet_structure_nad_defekt_i_scale():
    fake = _FakeCore(_empty_result(
        twist_idx={"x": [5]}, defekt_idx={"x": [7]}, anomaly_idx={"x": [1]},
    ))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.trigger_type == SignalTriggerType.STRUCTURE
    assert result.location == 5
    assert result.channel == "x"


def test_priorytet_defekt_nad_scale():
    fake = _FakeCore(_empty_result(
        defekt_idx={"x": [9]}, anomaly_idx={"x": [1]},
    ))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.trigger_type == SignalTriggerType.DEFEKT
    assert result.location == 9


def test_scale_gdy_tylko_jeden_kanal_anomalny():
    fake = _FakeCore(_empty_result(anomaly_idx={"x": [3]}))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.triggered is True
    assert result.trigger_type == SignalTriggerType.SCALE
    assert result.location == 3
    assert result.channel == "x"


def test_najmniejszy_indeks_wygrywa_miedzy_kanalami():
    """Gdy dwa RÓŻNE kanały mają defekt w różnych indeksach, dispatcher
    wskazuje najwcześniejszy indeks czasowy, niezależnie od kolejności
    kanałów w słowniku."""
    fake = _FakeCore(_empty_result(
        defekt_idx={"b": [9], "a": [4]},
    ))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.trigger_type == SignalTriggerType.DEFEKT
    assert result.location == 4
    assert result.channel == "a"


def test_none_gdy_wszystko_puste():
    fake = _FakeCore(_empty_result())
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert result.triggered is False
    assert result.trigger_type == SignalTriggerType.NONE
    assert result.location is None
    assert result.channel is None


def test_get_last_zwraca_ostatni_wynik():
    fake = _FakeCore(_empty_result(defekt_idx={"x": [4]}))
    trigger = TIMDRTrigger(core=fake)
    result = trigger.analyze(*_dummy_args())
    assert trigger.get_last() is result
