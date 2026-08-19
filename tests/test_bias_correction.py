from timdr_core import compute_lead_bias, apply_bias_correction, badge


def _pairs(lead, n, forecast=10.0, actual=12.0):
    return [{"lead": lead, "forecast": forecast, "actual": actual} for _ in range(n)]


def test_below_min_samples_gives_no_entry():
    pairs = _pairs(lead=1, n=4)  # min_samples domyslnie 5
    table = compute_lead_bias(pairs)
    assert 1 not in table


def test_at_min_samples_computes_bias_and_mae():
    pairs = _pairs(lead=1, n=5, forecast=10.0, actual=12.0)
    table = compute_lead_bias(pairs, min_samples=5)
    assert table[1]["n"] == 5
    assert table[1]["bias"] == 2.0
    assert table[1]["mae"] == 2.0


def test_apply_bias_correction_no_entry_returns_unchanged():
    assert apply_bias_correction(15.0, lead=3, bias_table={}) == 15.0


def test_apply_bias_correction_adds_bias():
    table = {2: {"bias": 1.5, "mae": 1.5, "n": 10}}
    assert apply_bias_correction(15.0, lead=2, bias_table=table) == 16.5


def test_badge_red_when_missing():
    assert badge(5, {}) == "🔴"


def test_badge_orange_small_sample():
    table = {5: {"bias": 0.1, "mae": 0.1, "n": 6}}
    assert badge(5, table, solid_n=15) == "🟠"


def test_badge_green_solid_sample():
    table = {5: {"bias": 0.1, "mae": 0.1, "n": 20}}
    assert badge(5, table, solid_n=15) == "🟢"
