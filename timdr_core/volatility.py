"""
timdr_core/volatility.py — wykrywanie skoku ("EV") między kolejnymi
uruchomieniami dla tego samego celu (stacja+dzień, sensor+event_id,
cokolwiek), niezależnie od domeny.

Stan poprzedniego odczytu MUSI być trzymany na dysku, nie tylko w pamięci
procesu — inaczej restart procesu cicho zeruje pamięć i detektor prawie
nigdy nie ma z czym porównać (patrz timdr-signal-framework, sekcja 5).
"""
from __future__ import annotations

import json
import os


def load_last_state(path: str) -> dict:
    """Wczytuje zapisany stan poprzednich odczytów. Brak pliku/uszkodzony
    plik -> pusty stan (fail-safe, nie fail-loud - to wygoda, nie krytyczna
    ścieżka)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {tuple(k.split("|", 1)): v for k, v in raw.items()}
    except Exception:
        return {}


def save_last_state(path: str, state: dict[tuple[str, str], dict]) -> None:
    try:
        raw = {f"{k[0]}|{k[1]}": v for k, v in state.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
    except Exception:
        pass  # wygoda, nie krytyczna ścieżka


def clear_state(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
        return True
    except Exception:
        return False


def detect_jump(prev_row: dict, new_row: dict, thresholds: dict[str, float]) -> dict[str, bool]:
    """thresholds: {nazwa_parametru: próg}. Zwraca {f"{param}_jump": True}
    tylko dla parametrów, które faktycznie przekroczyły próg - brak klucza
    w wyniku = brak skoku (albo brak danych do porównania), nigdy False
    jawnie wpisane, żeby wynik dało się łatwo sprawdzić przez `if flags:`."""
    flags: dict[str, bool] = {}
    for param, thr in thresholds.items():
        a, b = prev_row.get(param), new_row.get(param)
        if a is None or b is None:
            continue
        if abs(a - b) > thr:
            flags[f"{param}_jump"] = True
    return flags
