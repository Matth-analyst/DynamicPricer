"""Tests de l'analyse de sensibilité : briques légères, sans balayage complet."""

import json

import pytest

from dynamicpricer import sensitivity as S


def test_single_environment_problem_has_one_regime():
    pb = S.single_environment_problem(mu=0.2, h=0.01)
    assert pb.N == 1 and pb.mu == 0.2 and pb.h == 0.01


def test_figure_1_is_non_negative_and_below_published_ceiling():
    """En mono-environnement, le gain DP/S reste sous le plafond publié de 3,81 %."""
    rows = S.figure_1_data(mu_values=[0.25], h_values=(0.01,))
    assert len(rows) == 1
    assert 0.0 <= rows[0]["pg_dp_s"] <= 3.9


def test_menu_share_of_continuous_averages_ratios():
    rows = [
        {"pg_2": 1.0, "pg_3": 1.5, "pg_continuous": 2.0},
        {"pg_2": 3.0, "pg_3": 3.5, "pg_continuous": 4.0},
    ]
    share = S.menu_share_of_continuous(rows)
    assert share[2] == pytest.approx((0.5 + 0.75) / 2)
    assert share[3] == pytest.approx((0.75 + 0.875) / 2)


def test_stored_results_are_consistent_with_readme_claims():
    """Le JSON versionné vérifie les ordres de grandeur annoncés dans le README."""
    res = json.loads(S.RESULTS_PATH.read_text(encoding="utf-8"))
    mu = res["sweeps"]["mu"]
    assert max(r["pg_dp_edp"] for r in mu) < 3.3
    assert max(r["pg_dp_s"] for r in mu) > 13
    assert 0.77 < res["menu_share"]["2"] + 0.0 < 0.78
    assert 0.93 < res["menu_share"]["3"] < 0.94
