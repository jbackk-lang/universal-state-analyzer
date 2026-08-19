"""timdr_core - generyczny, niezależny od domeny silnik sygnałowy TIMDR
(anomalia/defekt/rezonans/skręt/rhythm) + korekta obciążenia + wykrywanie
skoku między uruchomieniami ("EV").

Wspólny rdzeń z timdr_core_finance.py (deliverable_timdr_finanse/) -
ta sama matematyka (TRM/FLOW/TWIST/ANOMALIE/DEFEKT/RHYTHM/REZONANS),
oczyszczona z założeń o konkretnej domenie (żadnych "price"/"volume").
Podłącz własne funkcje ekstrakcji komponentów z surowych danych (patrz
examples/accelerator/adapters.py) i korzystaj z tego samego silnika.

Nic tu nie jest zwalidowaną fizyką/finansami/czymkolwiek - to narzędzie do
wykrywania NIETYPOWYCH ODCZYTÓW W SZEREGU CZASOWYM, nie predykcyjny model.
"""

from .core import TIMDRCore
from .volatility import detect_jump, load_last_state, save_last_state, clear_state
from .bias_correction import compute_lead_bias, apply_bias_correction, badge

__all__ = [
    "TIMDRCore",
    "detect_jump",
    "load_last_state",
    "save_last_state",
    "clear_state",
    "compute_lead_bias",
    "apply_bias_correction",
    "badge",
]
