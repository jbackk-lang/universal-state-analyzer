# ============================================
# TIMDR Trigger Module (generyczny, bez wiedzy domenowej)
# ============================================
#
# ROLA: czujnik sygnałowy — NIE model, NIE predyktor. Ten plik nie liczy
# własnej statystyki: dispatcher nad już przetestowanym TIMDRCore.analyze_
# multi() (core.py). Jedyna jego robota: zapytać, KTÓRY typ zdarzenia
# odpalił się, w KTÓRYM kanale (parametrze z `params`) i GDZIE (indeks).
#
# Priorytet: RESONANCE (>=rezonans_min kanałów anomalnych naraz —
# najsilniejszy, najbardziej ugruntowany dowód, ta sama zasada koincydencji
# co w analyze_multi()/rezonans()) > STRUCTURE (twist — załamanie trendu —
# w KTÓRYMKOLWIEK kanale) > DEFEKT (nagły skok w KTÓRYMKOLWIEK kanale) >
# SCALE (pojedyncza anomalia statystyczna w jednym kanale) > NONE.
# Silniejszy/łączny dowód wygrywa niezależnie od chronologii ani od
# kolejności kanałów w słowniku `params` — ta sama zasada priorytetu co w
# TIMDR-Security-Module / TIMDR-Aviation-Diagnostics / deliverable_timdr_
# finanse. W obrębie STRUCTURE/DEFEKT/SCALE, gdy kilka kanałów flaguje
# równolegle, wygrywa najmniejszy indeks czasowy (najwcześniejszy sygnał).

from enum import Enum

from .core import TIMDRCore


class SignalTriggerType(Enum):
    RESONANCE = "resonance"
    STRUCTURE = "structure_twist"
    DEFEKT = "defekt"
    SCALE = "scale_anomaly"
    NONE = "none"


class SignalTriggerResult:
    def __init__(self, triggered=False, trigger_type=SignalTriggerType.NONE,
                 location=None, channel=None, message=""):
        self.triggered = triggered
        self.trigger_type = trigger_type
        self.location = location
        self.channel = channel
        self.message = message

    def as_dict(self):
        return {
            "triggered": self.triggered,
            "type": self.trigger_type.value,
            "location": self.location,
            "channel": self.channel,
            "message": self.message,
        }


class TIMDRTrigger:
    """
    Dispatcher nad TIMDRCore.analyze_multi(). `core` można wstrzyknąć (np.
    w testach) - domyślnie tworzy prawdziwy TIMDRCore(). Progi
    (anomaly_factor, twist_threshold, defekt_factor, rezonans_min) to te
    same punkty startowe do dostrojenia co w reszcie ekosystemu TIMDR, nie
    wartości uniwersalne.
    """

    def __init__(self, anomaly_factor=3.0, twist_threshold=0.4, defekt_factor=0.3,
                 rezonans_min=3, floor_frac=0.05, core=None):
        self.core = core if core is not None else TIMDRCore()
        self.anomaly_factor = anomaly_factor
        self.twist_threshold = twist_threshold
        self.defekt_factor = defekt_factor
        self.rezonans_min = rezonans_min
        self.floor_frac = floor_frac
        self.last_result = SignalTriggerResult()

    def analyze(self, t, params, baselines=None):
        r = self.core.analyze_multi(
            t, params,
            anomaly_factor=self.anomaly_factor,
            defekt_factor=self.defekt_factor,
            rezonans_min=self.rezonans_min,
            twist_threshold=self.twist_threshold,
            floor_frac=self.floor_frac,
            baselines=baselines,
        )

        if len(r["rezonans_idx"]):
            loc = int(r["rezonans_idx"][0])
            count = int(r["rezonans_counts"][loc])
            return self._set_result(
                True, SignalTriggerType.RESONANCE, loc, None,
                f"{count} kanałów anomalnych naraz."
            )

        best = self._earliest(r["twist_idx"])
        if best is not None:
            loc, channel = best
            return self._set_result(
                True, SignalTriggerType.STRUCTURE, loc, channel,
                f"Załamanie trendu (twist) w kanale '{channel}'."
            )

        best = self._earliest(r["defekt_idx"])
        if best is not None:
            loc, channel = best
            return self._set_result(
                True, SignalTriggerType.DEFEKT, loc, channel,
                f"Nagły skok w kanale '{channel}'."
            )

        best = self._earliest(r["anomaly_idx"])
        if best is not None:
            loc, channel = best
            return self._set_result(
                True, SignalTriggerType.SCALE, loc, channel,
                f"Pojedyncza anomalia statystyczna w kanale '{channel}'."
            )

        return self._set_result(
            False, SignalTriggerType.NONE, None, None,
            "Brak wykrytego zdarzenia sygnałowego."
        )

    @staticmethod
    def _earliest(idx_per_channel):
        """idx_per_channel: {nazwa: array-like indeksów}. Zwraca
        (najmniejszy_indeks, nazwa_kanału) po wszystkich kanałach naraz,
        albo None jeśli wszystkie puste. Przy remisie (ten sam najmniejszy
        indeks w kilku kanałach) wygrywa kanał najwcześniej wstawiony do
        słownika `params` - deterministyczne (Python dict zachowuje
        kolejność wstawiania), nie zależy od hashowania."""
        best = None
        for name, idxs in idx_per_channel.items():
            if len(idxs) == 0:
                continue
            candidate = int(min(idxs))
            if best is None or candidate < best[0]:
                best = (candidate, name)
        return best

    def _set_result(self, triggered, trigger_type, location, channel, message):
        self.last_result = SignalTriggerResult(triggered, trigger_type, location, channel, message)
        return self.last_result

    def get_last(self):
        return self.last_result
