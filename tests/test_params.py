"""Tests de la dataclass `Problem`."""

import numpy as np
import pytest

from dynamicpricer.params import Problem


def test_two_env_matches_paper_setup():
    """Le cas de référence du papier : Λ_L = 1-ε, Λ_H = 1+ε, q symétrique."""
    pb = Problem.two_env(eps=0.8, q=0.01, mu=0.11, h=0.01, c=0.0, a=1.0)
    assert pb.N == 2
    assert pb.Lam == pytest.approx((0.2, 1.8))
    assert pb.Q[0][1] == pb.Q[1][0] == 0.01


def test_uniformization_sums_to_one():
    pb = Problem.two_env(eps=0.6)
    norm, gamma = pb.normalized()
    total = norm.mu + sum(norm.Lam) + np.asarray(norm.Q).sum()
    assert total == pytest.approx(1.0)
    assert gamma == pytest.approx(pb.mu + sum(pb.Lam) + np.asarray(pb.Q).sum())


def test_normalization_preserves_cost_rate_ratios():
    """`h` et `b` sont des taux : ils suivent la même mise à l'échelle que `µ`."""
    pb = Problem.two_env(eps=0.3, b=0.05)
    norm, gamma = pb.normalized()
    assert norm.h == pytest.approx(pb.h / gamma)
    assert norm.b == pytest.approx(pb.b / gamma)
    # `c` et `K` sont des coûts par événement : inchangés.
    assert norm.c == pb.c and norm.K == pb.K


def test_linear_demand_and_inverse_are_consistent():
    pb = Problem.two_env(eps=0.6)
    for e in range(pb.N):
        for p in (0.0, 0.25, 0.5, 0.99):
            assert pb.price(pb.demand(p, e), e) == pytest.approx(p)


def test_revenue_is_maximized_at_p_star():
    """`r_e(λ)` est maximisé en `p* = 1/(2a)`, indépendamment de `Λ_e` (§5.1)."""
    pb = Problem.two_env(eps=0.8)
    for e in range(pb.N):
        best = max(pb.price_grid, key=lambda p: pb.revenue(pb.demand(p, e), e))
        assert best == pytest.approx(pb.p_star, abs=pb.price_step)


def test_holding_cost_switches_to_backlog_penalty():
    pb = Problem.two_env(eps=0.3, h=0.01, b=0.5, m_backlog=3)
    assert pb.holding_cost([-2, 0, 4]) == pytest.approx([1.0, 0.0, 0.04])


def test_two_env_is_birth_death_and_satisfies_condition_1():
    """Remarque 1 du papier : N=2 est un cas particulier de naissance-mort."""
    pb = Problem.two_env(eps=0.8)
    assert pb.is_birth_death()
    assert pb.satisfies_condition_1()


def test_condition_1_can_fail_for_more_than_two_environments():
    """Au-delà de N=2, la Condition 1 doit être vérifiée explicitement."""
    bad = Problem(
        Lam=(0.5, 1.0, 1.5),
        Q=((0.0, 0.01, 0.05), (0.01, 0.0, 0.01), (0.001, 0.01, 0.0)),
    )
    assert not bad.is_birth_death()
    assert not bad.satisfies_condition_1()


def test_environments_must_be_ordered():
    with pytest.raises(ValueError, match="ordonnés"):
        Problem(Lam=(1.5, 0.5), Q=((0.0, 0.01), (0.01, 0.0)))


@pytest.mark.parametrize("kwargs", [{"mu": 0.0}, {"h": -1.0}, {"m_stock": 0}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Problem.two_env(eps=0.3, **kwargs)
