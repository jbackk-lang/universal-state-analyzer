"""
analyze_trajectory.py — pierwszy przykład użycia timdr_core POZA
pogodą/finansami: analiza trajektorii z symulacji lattice/glueball
(glueball_mass.py, lattice_demo.py) tym samym generycznym silnikiem
anomalia/defekt/rezonans.

WAŻNE ZASTRZEŻENIE: glueball_mass.py i lattice_demo.py to jawnie
uproszczone/mock symulacje (patrz komentarze w tych plikach: "su2_mock",
"nieortodoksyjny, ale wystarczy jako mock", "tylko symbolicznie") - losowe
pola bez faktycznej dynamiki Monte Carlo sprzężonej działaniem (poza
trybem `metropolis`). To NIE jest zwalidowane obliczenie lattice QCD.
Trzy "kanały" (korelacja glueballa, plaquette Wilsona, action SU(3)) są tu
generowane NIEZALEŻNIE (różne wywołania RNG) - traktuj `rezonans` w tym
przykładzie jako demonstrację MECHANIZMU wykrywania (czy silnik poprawnie
łapie, gdy kilka kanałów jednocześnie wychodzi poza normę), nie jako
odkrycie fizycznej korelacji między nimi. Podłączenie do prawdziwych,
sprzężonych kanałów z jednej symulacji Monte Carlo to osobna praca.

Użycie:
    python analyze_trajectory.py --T 40 --N 12
    python analyze_trajectory.py --T 40 --N 12 --inject-anomaly-at 20
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from glueball_mass import compute_correlation
from lattice_demo import run_wilson_2d, run_su3
from timdr_core import TIMDRCore


def build_trajectory(T: int, N: int, mode: str = "basic", su3_N: int | None = None) -> dict[str, np.ndarray]:
    """Zbiera 3 niezależne kanały tej samej długości T:
    - glueball_C: korelacja operatora glueballowego C(t)
    - wilson_plaq: średni plaquette U(1) 2D
    - su3_action: mock 'action' SU(3) (mniejsza krata - su3_N - bo
      generate_su3_links jest O(N^2) z SVD per komórka, wolne dla dużego N)
    """
    su3_N = su3_N or max(4, N // 2)
    glueball_C = compute_correlation(T, N, mode)
    wilson_plaq = run_wilson_2d(T, N)
    su3_action = run_su3(T, su3_N)
    return {
        "glueball_C": glueball_C,
        "wilson_plaq": wilson_plaq,
        "su3_action": su3_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza trajektorii symulacji lattice/glueball generycznym silnikiem timdr_core."
    )
    parser.add_argument("--T", type=int, default=40, help="liczba kroków czasowych")
    parser.add_argument("--N", type=int, default=12, help="rozmiar kraty (glueball/wilson)")
    parser.add_argument("--mode", type=str, default="basic", choices=["basic", "noise", "su2_mock"],
                        help="nakładka pola dla glueball_mass")
    parser.add_argument("--seed", type=int, default=None, help="ziarno RNG (powtarzalność)")
    parser.add_argument("--inject-anomaly-at", type=int, default=None,
                        help="wstrzykuje sztuczny skok w glueball_C na tym indeksie t (demonstracja/test wykrywania)")
    parser.add_argument("--rezonans-min", type=int, default=2,
                        help="ile kanałów musi jednocześnie flagować anomalię, żeby zaliczyć rezonans (domyślnie 2 z 3 - patrz zastrzeżenie w docstringu modułu)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Start: T={args.T}, N={args.N}, mode={args.mode}, seed={args.seed}")
    series = build_trajectory(args.T, args.N, mode=args.mode)

    if args.inject_anomaly_at is not None:
        idx = args.inject_anomaly_at
        if 0 <= idx < args.T:
            spike = series["glueball_C"][idx] + 10 * (np.std(series["glueball_C"]) or 1.0)
            series["glueball_C"][idx] = spike
            print(f"Wstrzyknięto sztuczny skok w glueball_C[{idx}] = {spike:.4f}")

    t = np.arange(args.T)
    core = TIMDRCore()
    result = core.analyze_multi(t, series, rezonans_min=args.rezonans_min)

    print("\n=== WYNIK ===")
    for name in series:
        an = result["anomaly_idx"][name]
        de = result["defekt_idx"][name]
        tw = result["twist_idx"][name]
        print(f"{name:12s}: anomalie@{list(an)}  defekty@{list(de)}  skręt@{list(tw)}")

    rez = result["rezonans_idx"]
    if len(rez):
        print(f"\nREZONANS (>= {args.rezonans_min} kanałów naraz) w krokach: {list(rez)}")
    else:
        print(f"\nBrak rezonansu (>= {args.rezonans_min} kanałów naraz nigdzie się nie pokrywa).")


if __name__ == "__main__":
    main()
