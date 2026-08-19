"""
timdr_core/bias_correction.py — prosta, w pełni przejrzysta korekta
obciążenia z sparowanych (prognoza, rzeczywistość) obserwacji, niezależna
od domeny.

To NIE jest model ML - żadnego treningu, żadnego zapisanego stanu poza
samymi parami. Grupuje po `lead` (dowolna miara "jak daleko naprzód" -
dni, kroki symulacji, cokolwiek), liczy średni błąd (bias) i średni błąd
bezwzględny (MAE) per grupa, i stosuje korektę tylko tam, gdzie jest
wystarczająco dużo sparowanych obserwacji.
"""
from __future__ import annotations


def compute_lead_bias(
    pairs: list[dict],
    lead_key: str = "lead",
    forecast_key: str = "forecast",
    actual_key: str = "actual",
    min_samples: int = 5,
) -> dict[int, dict]:
    """pairs: lista {lead_key: int, forecast_key: float, actual_key: float}.
    Zwraca {lead: {"bias":..., "mae":..., "n":...}} TYLKO dla lead z
    >= min_samples parami - brak wpisu = brak korekty (za mało danych,
    NIE zakłada się zera)."""
    groups: dict[int, list[float]] = {}
    for p in pairs:
        if lead_key not in p or forecast_key not in p or actual_key not in p:
            continue
        if p[forecast_key] is None or p[actual_key] is None:
            continue
        lead = int(p[lead_key])
        err = float(p[actual_key]) - float(p[forecast_key])
        groups.setdefault(lead, []).append(err)

    result: dict[int, dict] = {}
    for lead, errors in groups.items():
        n = len(errors)
        if n < min_samples:
            continue
        mean_err = sum(errors) / n
        mae = sum(abs(e) for e in errors) / n
        result[lead] = {"bias": round(mean_err, 3), "mae": round(mae, 3), "n": n}
    return result


def apply_bias_correction(value: float, lead: int, bias_table: dict[int, dict]) -> float:
    entry = bias_table.get(int(lead))
    if entry is None:
        return value
    return round(value + entry["bias"], 3)


def badge(lead: int, bias_table: dict[int, dict], solid_n: int = 15) -> str:
    """🔴 za mało próbek (brak korekty) / 🟠 korekta aktywna, mała próbka /
    🟢 korekta aktywna, solidna próbka. Zawsze zwraca znaczek, nawet
    czerwony - brak znaczka czytałoby się jako "wszystko OK", co bywa
    fałszywe, gdy po prostu nie ma jeszcze danych."""
    entry = bias_table.get(int(lead))
    if entry is None:
        return "🔴"
    return "🟢" if entry["n"] >= solid_n else "🟠"
