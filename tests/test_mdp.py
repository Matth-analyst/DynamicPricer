"""Tests du solveur MDP (Étape 1 de la spec)."""

import numpy as np
import pytest

from mdp import (
    T0_setup,
    T0_unit,
    T2,
    T_dynamic,
    Tlam,
    env_matrix,
    solve_mdp,
)
from params import Problem

# ---------------------------------------------------------------------------
# Étape 1.1 — opérateurs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [1, 2, 4])
def test_operator_masses_sum_to_one(N):
    """`µ + masse(T2) + masse(Tlam) = 1` dans chaque état, après uniformisation.

    C'est la condition qui fait de `-h + µT0 + T2 + Tlam` un opérateur de Bellman
    en gain moyen propre. On la vérifie en appliquant les opérateurs à `f ≡ 1`.
    """
    Lam = tuple(0.5 + 0.3 * i for i in range(N))
    Q = tuple(tuple(0.0 if i == j else 0.01 for j in range(N)) for i in range(N))
    pb, _ = Problem(Lam=Lam, Q=Q, m_stock=10).normalized()

    ones = np.ones((pb.states.size, pb.N))
    lam = np.asarray(pb.Lam) * 0.4
    # Sur `f ≡ 1`, `Tlam` ajoute le revenu : on le retranche pour ne garder que la masse.
    rev = np.array([pb.revenue(float(lam[e]), e) for e in range(pb.N)])
    mass_lam = Tlam(ones, pb, lam)
    mass_lam[1:] -= rev

    total = pb.mu * T0_unit(ones, pb) + T2(ones, pb) + mass_lam
    assert total == pytest.approx(np.ones_like(total))


def test_env_matrix_reproduces_equation_2():
    """`T2` doit coïncider avec la somme littérale de l'équation (2)."""
    pb = Problem(
        Lam=(0.5, 1.0, 1.5),
        Q=((0.0, 0.02, 0.01), (0.03, 0.0, 0.02), (0.01, 0.04, 0.0)),
        m_stock=6,
    )
    rng = np.random.default_rng(0)
    f = rng.normal(size=(pb.states.size, pb.N))
    Q = np.asarray(pb.Q)
    Lam = np.asarray(pb.Lam)

    expected = np.zeros_like(f)
    for e in range(pb.N):
        fictitious = sum(Lam[i] for i in range(pb.N) if i != e) + sum(
            Q[i, j] for i in range(pb.N) for j in range(pb.N) if i != e and j != i
        )
        expected[:, e] = sum(Q[e, j] * f[:, j] for j in range(pb.N) if j != e)
        expected[:, e] += fictitious * f[:, e]

    assert T2(f, pb, env_matrix(pb)) == pytest.approx(expected)


def test_tlam_loses_demand_at_lower_boundary():
    """En `x = x_min` la demande est perdue : pas de revenu, pas de transition."""
    pb = Problem.two_env(eps=0.3, m_stock=5)
    f = np.arange(pb.states.size * pb.N, dtype=float).reshape(pb.states.size, pb.N)
    out = Tlam(f, pb, np.asarray(pb.Lam) * 0.5)
    assert out[0] == pytest.approx(np.asarray(pb.Lam) * f[0])


def test_dynamic_price_matches_closed_form_of_appendix_e():
    """La forme fermée (11) doit battre toute la grille de prix discrète."""
    pb = Problem.two_env(eps=0.6, m_stock=25)
    rng = np.random.default_rng(1)
    f = np.sort(rng.normal(size=(pb.states.size, pb.N)), axis=0)  # croissante

    best, prices = T_dynamic(f, pb)
    for p in pb.price_grid:
        lam = np.array([pb.demand(float(p), e) for e in range(pb.N)])
        assert np.all(Tlam(f, pb, lam) <= best + 1e-12)

    delta = f[1:] - f[:-1]
    closed_form = np.clip((1 + pb.a * delta) / (2 * pb.a), 0, pb.p_max)
    assert prices[1:] == pytest.approx(closed_form)
    assert np.isnan(prices[0]).all()


def test_dynamic_price_menu_is_restricted_to_the_menu():
    pb = Problem.two_env(eps=0.6, m_stock=15)
    rng = np.random.default_rng(2)
    f = np.sort(rng.normal(size=(pb.states.size, pb.N)), axis=0)
    menu = np.array([0.3, 0.6, 0.9])

    value, prices = T_dynamic(f, pb, price_menu=menu)
    assert set(np.unique(prices[1:])).issubset(set(menu))
    # Un menu fini ne peut pas battre le prix continu.
    assert np.all(value <= T_dynamic(f, pb)[0] + 1e-12)


# ---------------------------------------------------------------------------
# Étape 1.1 — value iteration et propriétés structurelles
# ---------------------------------------------------------------------------


def test_single_environment_case_is_solvable():
    """Cas trivial à un seul environnement (critère de sortie de l'Étape 1.1)."""
    pb = Problem(Lam=(1.0,), Q=((0.0,),), m_stock=40)
    sol = solve_mdp(pb)
    assert sol.converged and sol.g > 0
    assert len(sol.base_stock) == 1


def test_value_function_is_concave():
    """Lemme 1 : les opérateurs préservent la concavité, donc `v` est concave."""
    pb = Problem.two_env(eps=0.6, m_stock=60)
    for lam in (None, np.array([0.2, 0.5])):
        v = solve_mdp(pb, lam=lam).v
        second_diff = np.diff(v, n=2, axis=0)
        assert np.all(second_diff <= 1e-9)


def test_dynamic_prices_are_non_increasing_in_stock():
    """Théorème 2 : `x < y` ⟹ `p(x,e) >= p(y,e)`."""
    sol = solve_mdp(Problem.two_env(eps=0.8, m_stock=60))
    assert np.all(np.diff(sol.prices[1:], axis=0) <= 1e-9)


def test_base_stock_levels_are_ordered_by_environment():
    """Théorème 3 / Remarque 1 : pour `N=2` (naissance-mort), `s_L <= s_H`."""
    for eps in (0.3, 0.6, 0.8):
        pb = Problem.two_env(eps=eps, m_stock=60)
        assert pb.satisfies_condition_1()
        # `λ_e` proportionnel à `Λ_e` : garantit `λ_L <= λ_H` et `λ_L p_L <= λ_H p_H`,
        # les deux hypothèses du Théorème 3, pour tout `ε`.
        s = solve_mdp(pb, lam=0.4 * np.asarray(pb.Lam)).base_stock
        assert s[0] <= s[1]


def test_corollary_1_price_crosses_p_star_at_the_base_stock_level():
    """Corollaire 1 : `p(x,e) >= p*` si `x <= s_e`, `p(x,e) <= p*` sinon."""
    pb = Problem.two_env(eps=0.6, m_stock=60)
    sol = solve_mdp(pb)
    for e in range(pb.N):
        s_e = sol.base_stock[e]
        below = sol.prices[1 : s_e + 1, e]
        above = sol.prices[s_e + 1 :, e]
        assert np.all(below >= pb.p_star - 1e-9)
        assert np.all(above <= pb.p_star + 1e-9)
        # §4.2 : le prix au niveau de base-stock vaut approximativement `p*`.
        assert sol.prices[s_e, e] == pytest.approx(pb.p_star, abs=0.02)


def test_gain_is_insensitive_to_truncation():
    """Appendix E : `m` doit être augmenté jusqu'à insensibilité du gain moyen."""
    gains = [solve_mdp(Problem.two_env(eps=0.8, m_stock=m, tol=1e-9)).g for m in (60, 120)]
    assert gains[0] == pytest.approx(gains[1], rel=1e-4)


# ---------------------------------------------------------------------------
# Étape 1.2 — coût fixe de commande
# ---------------------------------------------------------------------------


def test_batch_operator_does_not_degenerate_to_the_unit_operator():
    """Le test de non-régression `K=0` de la spec est impossible sous cette forme.

    `T0_setup` remonte au sommet de `y ↦ f(y,e) - c·y` en une seule opportunité,
    là où `T0_unit` avance d'une unité : les deux opérateurs diffèrent strictement
    dès que le sommet est à plus d'un pas, même avec `K = 0`.
    """
    pb = Problem.two_env(eps=0.6, c=0.0, K=0.0, m_stock=10)
    f = np.array([[-((x - 6) ** 2) / 10.0, -((x - 4) ** 2) / 10.0] for x in pb.states])
    assert np.all(np.diff(f, n=2, axis=0) <= 1e-12), "f doit être concave"

    unit, batch = T0_unit(f, pb), T0_setup(f, pb)
    assert np.all(batch >= unit - 1e-12)  # le lot domine toujours
    assert np.any(batch > unit + 1e-6)  # et strictement, loin du sommet
    # Les deux coïncident au sommet et au-delà, où la cible optimale est `x` ou `x+1`.
    assert batch[-3:] == pytest.approx(unit[-3:])


def test_batch_model_dominates_the_capacitated_model():
    """Conséquence : le gain du modèle « lot » n'est pas comparable à l'Étape 1.1.

    La quantité commandée par opportunité étant illimitée, la contrainte de
    capacité disparaît et le gain moyen dépasse celui du make-to-stock queue.
    """
    lam = np.array([0.2, 0.5])
    capacitated = solve_mdp(Problem.two_env(eps=0.6, K=0.0, m_stock=60), lam=lam)
    batch = solve_mdp(Problem.two_env(eps=0.6, K=0.5, m_stock=60), lam=lam)
    assert batch.g > capacitated.g


def test_positive_setup_cost_yields_an_sS_policy():
    """Étape 1.2 : la politique optimale est de type `(s_e, S_e)` avec `s_e <= S_e`."""
    pb = Problem.two_env(eps=0.6, K=0.5, m_stock=60)
    sol = solve_mdp(pb, lam=np.array([0.2, 0.5]))
    assert sol.base_stock is None and sol.sS is not None
    for s_e, S_e in sol.sS:
        assert s_e <= S_e
    # Un coût fixe creuse un écart strict entre les deux seuils, contrairement au
    # base-stock pur où l'on commanderait dès `x = S - 1`.
    assert any(S_e - s_e > 1 for s_e, S_e in sol.sS)


def test_higher_setup_cost_lowers_profit_within_the_batch_model():
    """La comparaison qui a un sens : `K` croissant, à modèle constant."""
    lam = np.array([0.2, 0.5])
    cheap = solve_mdp(Problem.two_env(eps=0.6, K=0.5, m_stock=60), lam=lam)
    dear = solve_mdp(Problem.two_env(eps=0.6, K=1.0, m_stock=60), lam=lam)
    assert dear.g < cheap.g
    # Un `K` plus élevé espace les commandes : l'écart `S - s` s'élargit.
    assert sum(S - s for s, S in dear.sS) >= sum(S - s for s, S in cheap.sS)


# ---------------------------------------------------------------------------
# Étape 1.3 — backlog
# ---------------------------------------------------------------------------


def test_large_backlog_penalty_converges_to_lost_sales():
    """Étape 1.3 : `b → ∞` doit redonner exactement le comportement lost-sales.

    Sous DP, une pénurie très coûteuse pousse le prix au prix d'étranglement `1/a`
    en `x = 0`, donc `λ = 0` : le stock ne descend jamais sous zéro et le modèle
    coïncide avec celui de l'Étape 1.1.
    """
    ref = solve_mdp(Problem.two_env(eps=0.6, m_stock=60, tol=1e-9))
    backlog = solve_mdp(
        Problem.two_env(eps=0.6, m_stock=60, m_backlog=5, b=1e6, tol=1e-9)
    )
    assert backlog.g == pytest.approx(ref.g, rel=1e-6)
    assert backlog.base_stock == ref.base_stock


def test_backlog_is_used_when_the_penalty_is_low():
    """Avec un `b` faible, le système accepte encore de vendre en stock négatif.

    Le test porte sur le backlog **peu profond** : près de la borne `x_min`, la
    valeur marginale d'une unité explose et le prix atteint `1/a` quel que soit
    `b`, ce qui est un effet de la troncature et non du coût de pénurie.
    """
    pb = Problem.two_env(eps=0.6, m_stock=60, m_backlog=10, b=0.005)
    sol = solve_mdp(pb)
    shallow = np.isin(pb.states, [-2, -1])
    assert np.all(sol.prices[shallow] < pb.p_max - 1e-6)


def test_backlog_penalty_raises_prices_in_the_shortage_region():
    """Statique comparative : `b` croissant ⟹ le prix monte vers le prix d'étranglement.

    À `b` élevé le prix atteint `1/a`, soit `λ = 0` : le système cesse d'accepter
    les commandes qui creuseraient le backlog.
    """
    prices = []
    for b in (0.005, 0.02, 0.2):
        pb = Problem.two_env(eps=0.6, m_stock=60, m_backlog=10, b=b)
        sol = solve_mdp(pb)
        prices.append(sol.prices[np.flatnonzero(pb.states == 0)[0], 1])
    assert prices[0] < prices[1] < prices[2]
    assert prices[2] == pytest.approx(1.0)


def test_backlog_penalty_lowers_the_average_profit():
    lam = np.array([0.2, 0.5])
    weak = solve_mdp(Problem.two_env(eps=0.6, m_stock=60, m_backlog=8, b=0.02), lam=lam)
    strong = solve_mdp(Problem.two_env(eps=0.6, m_stock=60, m_backlog=8, b=0.5), lam=lam)
    assert strong.g < weak.g
