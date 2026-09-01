"""Tests des cinq stratégies (Étape 2 de la spec).

Les tests utilisent un pas de prix grossier (`price_step=0.05`) pour rester rapides :
le pas de 0.01 de l'Appendix E est réservé à la validation chiffrée de l'Étape 3,
qui vit dans les notebooks. Ce qui est vérifié ici, ce sont les **invariants** de la
classification de la Table 1, pas les valeurs publiées.
"""

import numpy as np
import pytest

from metrics import check_ordering, profit_gain
from params import Problem
from strategies import (
    MAX_PRICE_COMBINATIONS,
    STRATEGIES,
    solve_all,
    solve_dp,
    solve_edp,
    solve_static,
)

COARSE = dict(price_step=0.05, m_stock=40)


@pytest.fixture(scope="module")
def fluctuating():
    """Un environnement nettement fluctuant, où les cinq stratégies se séparent."""
    return solve_all(Problem.two_env(eps=0.6, **COARSE))


# ---------------------------------------------------------------------------
# Invariants de la classification (Table 1)
# ---------------------------------------------------------------------------


def test_all_five_strategies_are_returned(fluctuating):
    assert set(fluctuating) == set(STRATEGIES)


def test_profit_ordering_follows_policy_nesting(fluctuating):
    """§5.1 : `g_s <= g_sb <= g_edp <= g_dp` et `g_s <= g_sp <= g_edp <= g_dp`.

    Les classes de politiques sont emboîtées, donc l'ordre est structurel : il doit
    tenir pour toute instance, et le violer signalerait une erreur de résolution.
    """
    assert check_ordering({k: sol.g for k, sol in fluctuating.items()})


def test_constrained_strategies_have_a_single_base_stock(fluctuating):
    """S et SB imposent un seuil commun ; SP, EDP et DP le laissent libre."""
    assert fluctuating["s"].is_common_base_stock
    assert fluctuating["sb"].is_common_base_stock
    # À ε = 0.6, les stratégies libres exploitent effectivement cette liberté.
    assert not fluctuating["sp"].is_common_base_stock
    assert not fluctuating["edp"].is_common_base_stock


def test_single_price_strategies_quote_one_price(fluctuating):
    """S et SP n'ont qu'un prix ; SB et EDP en ont un par environnement."""
    for name in ("s", "sp"):
        assert len(set(fluctuating[name].prices)) == 1
    assert fluctuating["sb"].prices[0] < fluctuating["sb"].prices[1]
    assert fluctuating["edp"].prices[0] < fluctuating["edp"].prices[1]


def test_dp_prices_are_state_dependent(fluctuating):
    """DP fixe un prix par état `(x, e)`, pas un scalaire par environnement."""
    pb = Problem.two_env(eps=0.6, **COARSE)
    prices = fluctuating["dp"].prices
    assert prices.shape == (pb.states.size, pb.N)
    assert np.nanmax(prices) > np.nanmin(prices)


def test_base_stock_levels_are_ordered_by_environment(fluctuating):
    """Théorème 3 : `s_L <= s_H` pour SP et EDP (et constaté aussi pour DP, §5.3.2)."""
    for name in ("sp", "edp", "dp"):
        s = fluctuating[name].base_stock
        assert s[0] <= s[1]


def test_free_base_stock_stores_at_least_as_much_as_dp_allows(fluctuating):
    """§5.3.2 : `s^φ_e <= s^dp_e` — la flexibilité de DP autorise plus de stock."""
    for name in ("s", "sb", "sp", "edp"):
        assert all(
            a <= b for a, b in zip(fluctuating[name].base_stock, fluctuating["dp"].base_stock)
        )


# ---------------------------------------------------------------------------
# Cas dégénérés et garde-fous
# ---------------------------------------------------------------------------


def test_without_fluctuation_the_four_static_strategies_coincide():
    """§5.2 : à `ε = 0` le système est mono-environnement — S, SB, SP et EDP se confondent.

    Seule DP s'en détache, puisqu'elle seule fait dépendre le prix du stock.
    """
    sols = solve_all(Problem.two_env(eps=0.0, **COARSE))
    gains = [sols[name].g for name in ("s", "sb", "sp", "edp")]
    assert gains == pytest.approx([gains[0]] * 4, rel=1e-6)
    assert sols["dp"].g > gains[0]

    base = [sols[name].base_stock for name in ("s", "sb", "sp", "edp")]
    assert all(b == base[0] for b in base)


def test_profit_gains_grow_with_demand_variability():
    """Table 3 : `PG^{φ,s}` croît avec `ε` — c'est l'enseignement central du papier."""
    gains = []
    for eps in (0.0, 0.3, 0.6):
        sols = solve_all(Problem.two_env(eps=eps, **COARSE))
        gains.append(profit_gain(sols["edp"].g, sols["s"].g))
    assert gains[0] < gains[1] < gains[2]


def test_price_grid_guard_rejects_intractable_sweeps():
    """Le balayage EDP croît en `(1/price_step)^N` : refus explicite plutôt que blocage."""
    pb = Problem(Lam=(0.5, 1.0, 1.5), Q=((0, 0.01, 0.01), (0.01, 0, 0.01), (0.01, 0.01, 0)))
    assert pb.price_grid.size**pb.N > MAX_PRICE_COMBINATIONS
    with pytest.raises(ValueError, match="price_step"):
        solve_edp(pb)


def test_three_environments_are_supported_with_a_coarser_grid():
    """`N > 2` fonctionne dès que la grille de prix reste praticable."""
    pb = Problem(
        Lam=(0.5, 1.0, 1.5),
        Q=((0, 0.01, 0.0), (0.01, 0, 0.01), (0.0, 0.01, 0)),
        price_step=0.1,
        m_stock=30,
    )
    sols = solve_all(pb)
    assert check_ordering({k: sol.g for k, sol in sols.items()})
    assert len(sols["edp"].prices) == 3
    s = sols["edp"].base_stock
    assert s[0] <= s[1] <= s[2]  # Théorème 3, Condition 1 satisfaite (naissance-mort)


def test_static_strategy_is_the_profit_gain_denominator():
    """S doit être la moins profitable des cinq : c'est ce qui en fait la référence."""
    pb = Problem.two_env(eps=0.6, **COARSE)
    sols = solve_all(pb)
    assert solve_static(pb).g == pytest.approx(sols["s"].g)
    assert all(sols[name].g >= sols["s"].g - 1e-12 for name in STRATEGIES)


def test_dp_solution_matches_the_underlying_mdp():
    """`solve_dp` n'est qu'une enveloppe autour de `solve_mdp` : rien ne doit diverger."""
    from mdp import solve_mdp

    pb = Problem.two_env(eps=0.6, **COARSE)
    assert solve_dp(pb).g == pytest.approx(solve_mdp(pb).g)
