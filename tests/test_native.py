"""Équivalence entre le moteur natif Rust et le moteur Python de référence.

Le natif n'a d'intérêt que s'il donne la même réponse : c'est le moteur Python qui est
validé contre le papier, et le portage n'hérite de cette validation que par
l'équivalence testée ici.

Ces tests sont ignorés si le module natif n'a pas été compilé — `cd rust && cargo build
--release` — pour que la suite reste exécutable sans chaîne Rust.
"""

import numpy as np
import pytest

from dynamicpricer import native
from dynamicpricer.mdp import solve_batch_truncated_python
from dynamicpricer.params import Problem
from dynamicpricer.strategies import demand_from_prices, price_vector_grid, solve_edp

pytestmark = pytest.mark.skipif(not native.AVAILABLE, reason="moteur natif non compilé")


def _both(pb, lam):
    """Résout le même lot par les deux moteurs, à troncature et normalisation égales."""
    npb, gamma = pb.normalized()
    lam_n = np.asarray(lam, dtype=float) / gamma
    rust = native.solve_batch_truncated(npb, lam_n, gamma, 100)
    python = solve_batch_truncated_python(npb, lam_n, gamma, 100)
    return rust, python


@pytest.mark.parametrize("eps", [0.0, 0.3, 0.8])
def test_gains_match_to_machine_precision(eps):
    """Le gain moyen doit coïncider instance par instance, pas seulement à l'optimum."""
    pb = Problem.two_env(eps=eps, m_stock=40, price_step=0.05)
    lam = demand_from_prices(pb, price_vector_grid(pb))
    (g_rust, _), (g_py, _) = _both(pb, lam)
    assert g_rust == pytest.approx(g_py, abs=1e-12)


def test_optimum_agrees_exactly():
    """Ce qui compte en pratique : même argmax, même gain, même politique retenue.

    Les base-stocks peuvent différer sur des instances **non optimales** : plusieurs
    seuils y donnent exactement le même gain (écarts de l'ordre de 1e-17), et les deux
    moteurs ne tranchent pas l'égalité de la même façon. Les deux politiques sont alors
    optimales toutes les deux ; seule la politique retenue à l'optimum est observable
    dans les résultats.
    """
    pb = Problem.two_env(eps=0.6, m_stock=40, price_step=0.05)
    lam = demand_from_prices(pb, price_vector_grid(pb))
    (g_rust, s_rust), (g_py, s_py) = _both(pb, lam)
    i, j = int(np.argmax(g_rust)), int(np.argmax(g_py))
    assert i == j
    assert g_rust[i] == pytest.approx(g_py[j], abs=1e-12)
    assert tuple(s_rust[:, i]) == tuple(s_py[:, j])


def test_tied_policies_are_genuinely_equivalent():
    """Là où les seuils diffèrent, les gains sont rigoureusement égaux.

    C'est ce qui établit que la divergence est une égalité de valeur et non une erreur :
    si l'un des deux moteurs se trompait, l'écart de gain serait visible.
    """
    pb = Problem.two_env(eps=0.6, m_stock=40, price_step=0.05)
    lam = demand_from_prices(pb, price_vector_grid(pb))
    (g_rust, s_rust), (g_py, s_py) = _both(pb, lam)
    differing = (s_rust != s_py).any(axis=0)
    assert np.abs(g_rust[differing] - g_py[differing]).max(initial=0.0) < 1e-12


def test_backlog_and_production_cost_are_honoured():
    """Backlog (`x < 0`) et coût de production unitaire doivent être pris en compte."""
    pb = Problem.two_env(eps=0.6, m_stock=30, m_backlog=6, b=0.02, c=0.05, price_step=0.1)
    lam = demand_from_prices(pb, price_vector_grid(pb))
    (g_rust, _), (g_py, _) = _both(pb, lam)
    assert g_rust == pytest.approx(g_py, abs=1e-12)


def test_three_environments_match():
    """La structure bloc-tridiagonale doit rester correcte pour `N > 2`."""
    pb = Problem(
        Lam=(0.5, 1.0, 1.5),
        Q=((0.0, 0.01, 0.0), (0.01, 0.0, 0.01), (0.0, 0.01, 0.0)),
        m_stock=30, price_step=0.1,
    )
    lam = demand_from_prices(pb, price_vector_grid(pb))
    (g_rust, _), (g_py, _) = _both(pb, lam)
    assert g_rust == pytest.approx(g_py, abs=1e-12)


def test_edp_still_reproduces_the_paper_through_the_native_path():
    """Bout en bout : la stratégie EDP doit toujours donner les valeurs publiées."""
    from dynamicpricer import reference_gayon2009 as ref

    pb = Problem.two_env(eps=0.6, m_stock=80)
    sol = solve_edp(pb)
    assert sol.base_stock == ref.TABLE_4[0.6]["edp"]
    assert tuple(round(float(p), 2) for p in sol.prices) == ref.TABLE_5[0.6]["edp"]


def test_common_level_search_matches_python():
    """Stratégies S et SB : le seuil commun natif doit reproduire l'évaluation Python.

    C'est le poste le plus lourd du solveur, donc celui où une divergence coûterait le
    plus cher à détecter tard.
    """
    from dynamicpricer.mdp import average_profit

    pb = Problem.two_env(eps=0.6, m_stock=60, price_step=0.05)
    lam = demand_from_prices(pb, price_vector_grid(pb)).T
    bound = 14
    levels = np.arange(0, bound + 1)

    g_rust, lvl_rust = native.search_common_level(pb, lam, bound)
    gains = np.stack([average_profit(pb, lam, np.full(pb.N, s)) for s in levels])
    g_py = gains.max(axis=0)

    assert g_rust == pytest.approx(g_py, abs=1e-12)
    # Le niveau retenu doit atteindre ce maximum, y compris en cas d'égalité.
    achieved = gains[lvl_rust, np.arange(lam.shape[0])]
    assert achieved == pytest.approx(g_py, abs=1e-12)


@pytest.mark.parametrize("name", ["s", "sb"])
def test_constrained_strategies_agree_end_to_end(name):
    """Bout en bout : S et SB doivent donner le même gain par les deux chemins."""
    from dynamicpricer import strategies

    pb = Problem.two_env(eps=0.6, m_stock=60, price_step=0.05)
    native_sol = strategies.SOLVERS[name](pb)

    original = native.HAS_COMMON_LEVEL
    strategies.native.HAS_COMMON_LEVEL = False
    try:
        python_sol = strategies.SOLVERS[name](pb)
    finally:
        strategies.native.HAS_COMMON_LEVEL = original

    assert native_sol.g == pytest.approx(python_sol.g, abs=1e-12)
    assert native_sol.base_stock == python_sol.base_stock
    assert native_sol.prices == pytest.approx(python_sol.prices)
