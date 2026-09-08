"""Testy timdr_core/hypothesis_testing.py -- sprawdzają TYLKO, że
sibling-import z TIMDR-Math-Formalism działa i re-eksportowane funkcje są
wywoływalne z tego repo (smoke test integracji), nie duplikują testowania
samej matematyki Manna-Whitneya/kontroli/Bonferroniego -- to już jest
wyczerpująco przetestowane w
TIMDR-Math-Formalism/tests/test_pipeline.py, jedynym miejscu definicji
tej logiki (patrz nagłówek hypothesis_testing.py)."""
import numpy as np
import pytest

from timdr_core.hypothesis_testing import (
    Hypothesis,
    Preregistration,
    mann_whitney_test,
    run_controls,
    rank_biserial_effect_size,
    format_report,
)


def test_sibling_import_resolves_to_real_pipeline_module():
    # Upewnia sie, ze to NIE jest lokalna kopia -- modul pochodzi
    # faktycznie z pakietu timdr_formalism w repo-siostrze
    # TIMDR-Math-Formalism, zaimportowanego przez sys.path, nie
    # przepisanego tutaj.
    assert mann_whitney_test.__module__ == "timdr_formalism.pipeline"


def test_mann_whitney_test_usable_on_timdr_core_style_metric_values():
    # Symuluje typowy przypadek uzycia: metryka zbudowana na wyniku
    # jednego z operatorow timdr_core (tu: sztucznie, wartosci float)
    # porownana test-vs-tlo.
    rng = np.random.default_rng(0)
    candidate_metric_test = rng.normal(loc=5.0, scale=1.0, size=30)
    candidate_metric_background = rng.normal(loc=0.0, scale=1.0, size=30)
    result = mann_whitney_test(candidate_metric_test, candidate_metric_background)
    assert result.pvalue < 0.01
    assert result.effect_size_r > 0.5


def test_hypothesis_and_preregistration_roundtrip():
    h = Hypothesis(
        name="nowa_hipoteza_detekcyjna_test",
        description="testowa hipoteza dla smoke-testu integracji",
        effect_description="roznica median metryki kandydackiej",
    )
    prereg = Preregistration.create(h, {"n": 30})
    assert prereg.verify_unchanged(h, {"n": 30}) is True


def test_run_controls_and_format_report_usable_end_to_end():
    def metric_fn(data):
        return float(np.mean(data))

    def positive_injector(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=10.0, scale=0.5, size=window_size)

    def negative_generator(window_size, seed):
        rng = np.random.default_rng(seed)
        return rng.normal(loc=0.0, scale=0.5, size=window_size)

    controls = run_controls(
        metric_fn=metric_fn,
        positive_injector=positive_injector,
        negative_generator_a=negative_generator,
        negative_generator_b=negative_generator,
        n_windows=15,
        window_size=30,
        seed=42,
    )
    assert controls.passed is True

    h = Hypothesis(
        name="smoke_test", description="d", effect_description="e",
    )
    report = format_report(h, controls, main_result=None)
    assert isinstance(report, str) and len(report) > 0
