"""
timdr_core/core.py — silnik detekcji anomalia/defekt/rezonans/skręt(twist)/
rhythm, w PEŁNI niezależny od domeny.

To jest wydestylowana, wspólna część rdzenia używanego już w tym środowisku
dla pogody (Synoptyk), finansów (timdr_core_finance.py w
deliverable_timdr_finanse/) i sejsmiki - tu bez żadnych założeń o tym, co
to za sygnał: nie ma "temp"/"price"/cokolwiek. Wszystko to po prostu
S(t) - dowolny szereg czasowy z liczbami.

Metody statyczne/instancyjne operują na gołych tablicach (t, s) — Twój
kod dostarcza własne funkcje ekstrakcji komponentów z surowych danych
(patrz examples/accelerator/adapters.py po przykład dla lattice QCD, albo
timdr_core_finance.py po przykład dla świec OHLCV: delta_proxy/spread_proxy).

Zaimplementowane pułapki, na które warto uważać przy każdej nowej domenie
(patrz też skill timdr-signal-framework):
- MAD=0 / rozrzut p90-p10=0 na "cichych" seriach (mostly-zero/prawie stałe)
  -> podłoga (floor_frac), nie dzielenie przez zero i nie "wszystko anomalią".
- gradient (flow/twist) liczony WZGLĘDEM CZASU (t), nie indeksu próbki -
  inaczej luki w danych dają fałszywe alarmy.
- rhythm() na wartości ZE ZNAKIEM, po odjęciu trendu liniowego - rektyfikacja
  (|x|) tworzy sztuczną okresowość z samego wyprostowania sygnału.
"""
from __future__ import annotations

import numpy as np


class TIMDRCore:
    """Rdzeń bez żadnej wiedzy domenowej - patrz docstring modułu."""

    def __init__(self, mad_scale: float = 1.4826) -> None:
        self.mad_scale = mad_scale

    # ------------------------------------------------------------------
    # TRM — wygładzanie (mediana k-NN w czasie, po indeksie próbki)
    # ------------------------------------------------------------------
    @staticmethod
    def trm(t, s, k: int = 5) -> np.ndarray:
        t = np.asarray(t, float)
        s = np.asarray(s, float)
        n = len(s)
        if n == 0:
            return np.array([])
        out = np.empty(n)
        half = k // 2
        for i in range(n):
            j0, j1 = max(0, i - half), min(n, i + half + 1)
            out[i] = np.median(s[j0:j1])
        return out

    # ------------------------------------------------------------------
    # FLOW — lokalny gradient WZGLĘDEM CZASU (LSQ na oknie), nie indeksu
    # ------------------------------------------------------------------
    @staticmethod
    def flow(t, s, window: int = 5) -> np.ndarray:
        t = np.asarray(t, float)
        s = np.asarray(s, float)
        n = len(s)
        if n == 0:
            return np.array([])
        out = np.zeros(n)
        half = window // 2
        for i in range(n):
            j0, j1 = max(0, i - half), min(n, i + half + 1)
            tt, ss = t[j0:j1], s[j0:j1]
            if len(tt) < 2 or tt[-1] == tt[0]:
                continue
            A = np.column_stack([tt, np.ones_like(tt)])
            a, _ = np.linalg.lstsq(A, ss, rcond=None)[0]
            out[i] = a
        return out

    # ------------------------------------------------------------------
    # TWIST — nagła zmiana kierunku FLOW, różniczkowana względem CZASU
    # ------------------------------------------------------------------
    @staticmethod
    def twist(flow_vals, t, threshold: float = 0.4):
        flow_vals = np.asarray(flow_vals, float)
        t = np.asarray(t, float)
        n = len(flow_vals)
        if n < 3:
            return np.array([], dtype=int), np.array([])
        dg = np.gradient(flow_vals, t)  # WZGLĘDEM CZASU, nie indeksu
        std = np.std(dg) if np.std(dg) > 0 else 1e-9
        thr = threshold * std
        idx = np.where(np.abs(dg) > thr)[0]
        return idx, dg

    # ------------------------------------------------------------------
    # ANOMALIA — MAD-owy z-score z podłogą (unika pułapki zero-inflation)
    # ------------------------------------------------------------------
    def anomalies(self, t, s, factor: float = 3.0, floor_frac: float = 0.05):
        s = np.asarray(s, float)
        n = len(s)
        if n == 0:
            return np.array([], dtype=int), np.array([]), 0.0
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * self.mad_scale
        if mad == 0 or not np.isfinite(mad):
            std = np.std(s)
            mad = std if std > 0 else max(abs(med) * floor_frac, 1e-9)
        z = (s - med) / mad
        idx = np.where(np.abs(z) > factor)[0]
        return idx, z, mad * factor

    # ------------------------------------------------------------------
    # DEFEKT — nagły skok między kolejnymi odczytami, próg z p90-p10 + podłoga
    # ------------------------------------------------------------------
    @staticmethod
    def defekt(s, factor: float = 0.3, floor_frac: float = 0.05):
        s = np.asarray(s, float)
        n = len(s)
        if n < 2:
            return np.array([], dtype=int), np.array([])
        diffs = np.diff(s)
        p10, p90 = np.percentile(s, 10), np.percentile(s, 90)
        spread = p90 - p10
        if spread <= 0 or not np.isfinite(spread):
            spread = max(abs(np.median(s)) * floor_frac, 1e-9)
        thr = factor * spread
        idx = np.where(np.abs(diffs) > thr)[0] + 1  # indeks NOWEJ próbki
        return idx, diffs

    # ------------------------------------------------------------------
    # RHYTHM — autokorelacja na wartości ZE ZNAKIEM, po odjęciu trendu
    # ------------------------------------------------------------------
    @staticmethod
    def rhythm(values, max_lag: int = 48, power_thresh: float = 0.4):
        E = np.asarray(values, float)
        n = len(E)
        if n < 2:
            return [], 0.0
        idx = np.arange(n, dtype=float)
        if n > 2:
            slope, intercept = np.polyfit(idx, E, 1)
            E = E - (slope * idx + intercept)
        else:
            E = E - np.mean(E)
        max_lag = min(max_lag, n - 1)
        ac = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                ac[lag] = np.dot(E, E) / n
            else:
                overlap = n - lag
                if overlap <= 0:
                    break
                ac[lag] = np.dot(E[:-lag], E[lag:]) / overlap
        if ac[0] == 0:
            return [], 0.0
        ac /= ac[0]
        peaks = [(i, float(ac[i])) for i in range(1, len(ac) - 1)
                 if ac[i] > ac[i - 1] and ac[i] > ac[i + 1] and ac[i] >= power_thresh]
        if not peaks:
            return [], 0.0
        score = max(p for _, p in peaks)
        return [p for p, _ in peaks], score

    # ------------------------------------------------------------------
    # REZONANS — >=min_count parametrów flaguje anomalie() w tej samej chwili
    # ------------------------------------------------------------------
    @staticmethod
    def rezonans(anomaly_index_lists, n: int, min_count: int = 3):
        counts = np.zeros(n, dtype=int)
        for idxs in anomaly_index_lists:
            counts[np.asarray(idxs, dtype=int)] += 1
        idx = np.where(counts >= min_count)[0]
        return idx, counts

    # ------------------------------------------------------------------
    # WYGODNY PIPELINE — wiele parametrów naraz -> anomalie/defekty/rezonans
    # ------------------------------------------------------------------
    def analyze_multi(
        self,
        t,
        params: dict[str, np.ndarray],
        anomaly_factor: float = 3.0,
        defekt_factor: float = 0.3,
        rezonans_min: int = 3,
        twist_threshold: float = 0.4,
        floor_frac: float = 0.05,
    ) -> dict:
        """params: {nazwa: tablica wartości (ta sama długość co t)}.
        Zwraca anomalie/defekty per parametr, wspólny rezonans, i
        twist/flow policzone dla KAŻDEGO parametru osobno (nie jednego
        wybranego - w przeciwieństwie do finansowej wersji, gdzie flow/twist
        liczy się tylko na sigma; tu, bez wiedzy domenowej, robimy to dla
        wszystkich, a wywołujący ignoruje to, czego nie potrzebuje)."""
        t = np.asarray(t, float)
        n = len(t)
        anomaly_idx_per_param: dict[str, np.ndarray] = {}
        defekt_idx_per_param: dict[str, np.ndarray] = {}
        twist_idx_per_param: dict[str, np.ndarray] = {}

        for name, vals in params.items():
            vals = np.asarray(vals, float)
            an_idx, _, _ = self.anomalies(t, vals, factor=anomaly_factor, floor_frac=floor_frac)
            anomaly_idx_per_param[name] = an_idx

            de_idx, _ = self.defekt(vals, factor=defekt_factor, floor_frac=floor_frac)
            defekt_idx_per_param[name] = de_idx

            smoothed = self.trm(t, vals, k=5)
            flow_vals = self.flow(t, smoothed, window=5)
            tw_idx, _ = self.twist(flow_vals, t, threshold=twist_threshold)
            twist_idx_per_param[name] = tw_idx

        rez_idx, rez_counts = self.rezonans(
            list(anomaly_idx_per_param.values()), n=n, min_count=rezonans_min,
        )

        return dict(
            anomaly_idx=anomaly_idx_per_param,
            defekt_idx=defekt_idx_per_param,
            twist_idx=twist_idx_per_param,
            rezonans_idx=rez_idx,
            rezonans_counts=rez_counts,
        )
