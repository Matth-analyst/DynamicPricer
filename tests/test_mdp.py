"""Tests du solveur MDP (Étape 1 de la spec)."""

import numpy as np
import pytest

from mdp import (
    T0_unit,
    T_produce,
    T_switch,
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
# Étape 1.2 — coût fixe de mise en route du serveur
# ---------------------------------------------------------------------------


def test_switch_operator_charges_K_only_when_restarting():
    """`S f(x,e,σ) = max{f(x,e,0), f(x,e,1) - K·1{σ=0}}` (Étape 1.2)."""
    pb = Problem.two_env(eps=0.6, K=0.5, m_stock=5)
    rng = np.random.default_rng(4)
    f = rng.normal(size=(pb.states.size, pb.N, 2))

    out = T_switch(f, pb)
    assert out[..., 0] == pytest.approx(np.maximum(f[..., 0], f[..., 1] - pb.K))
    assert out[..., 1] == pytest.approx(np.maximum(f[..., 0], f[..., 1]))
    # Rester à l'arrêt est toujours gratuit, donc être en marche vaut au moins autant.
    assert np.all(out[..., 1] >= out[..., 0] - 1e-12)


def test_production_operator_only_advances_the_running_slice():
    pb = Problem.two_env(eps=0.6, c=0.05, K=0.5, m_stock=5)
    rng = np.random.default_rng(5)
    f = rng.normal(size=(pb.states.size, pb.N, 2))

    out = T_produce(f, pb)
    assert out[..., 0] == pytest.approx(f[..., 0])  # serveur arrêté : rien
    assert out[:-1, :, 1] == pytest.approx(f[1:, :, 1] - pb.c)  # une unité sort
    assert out[-1, :, 1] == pytest.approx(f[-1, :, 1])  # troncature : pas de production


def test_setup_model_with_K_zero_reproduces_step_1_1_exactly():
    """Non-régression de l'Étape 1.2 : `K = 0` redonne **exactement** l'Étape 1.1.

    Avec `K = 0`, `T_switch` devient `max{inner(·,0), inner(·,1)}`, ce qui reconstitue
    littéralement `max{f(x,e), f(x+1,e) - c}` : la réduction est exacte, pas approchée.
    """
    pb = Problem.two_env(eps=0.6, m_stock=60, tol=1e-9)
    ref = solve_mdp(pb)
    augmented = solve_mdp(pb, force_setup_model=True)

    assert augmented.g == pytest.approx(ref.g, rel=1e-12)
    # La bande d'hystérésis dégénère : les deux seuils se confondent avec le base-stock.
    assert augmented.sS == tuple((s, s) for s in ref.base_stock)


def test_positive_setup_cost_yields_a_hysteresis_band():
    """Étape 1.2 : deux seuils par environnement, avec `s_e <= S_e`."""
    sol = solve_mdp(Problem.two_env(eps=0.6, K=0.5, m_stock=120, tol=1e-9))
    assert sol.base_stock is None and sol.sS is not None
    for s_e, S_e in sol.sS:
        assert s_e < S_e  # un coût de mise en route creuse une bande stricte


def test_band_widens_and_profit_falls_as_the_setup_cost_rises():
    """Statique comparative : `K` croissant ⟹ on commute moins souvent, on gagne moins."""
    sols = [
        solve_mdp(Problem.two_env(eps=0.6, K=K, m_stock=160, tol=1e-9))
        for K in (0.05, 0.5, 2.0)
    ]
    widths = [sum(S - s for s, S in sol.sS) for sol in sols]
    assert widths[0] < widths[1] < widths[2]
    assert sols[0].g > sols[1].g > sols[2].g


def test_non_monotone_d_is_reported_rather_than_silently_accepted():
    """L'Étape 1.2 demande de *constater* la structure en bande, pas de la supposer."""
    from mdp import extract_setup_policy

    pb = Problem.two_env(eps=0.6, K=0.5, m_stock=5)
    d = np.tile(np.array([0.9, 0.1, 0.8, 0.05, 0.0, -0.5])[:, None], (1, pb.N))
    with pytest.raises(ValueError, match="décroissante"):
        extract_setup_policy(d, pb)


def test_threshold_outside_the_grid_is_reported():
    """Une troncature trop serrée doit être signalée, pas rendue en valeur sentinelle."""
    with pytest.raises(ValueError, match="m_stock"):
        solve_mdp(Problem.two_env(eps=0.6, K=2.0, m_stock=20, tol=1e-9))


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


# ---------------------------------------------------------------------------
# Étape 2 — résolution en lot et évaluation exacte
# ---------------------------------------------------------------------------


def test_exact_evaluation_agrees_with_value_iteration():
    """Deux méthodes indépendantes doivent donner le même gain moyen.

    `solve_mdp` itère l'opérateur de Bellman ; `average_profit` résout `πQ = 0` sur
    le générateur en temps continu. Leur accord valide simultanément l'uniformisation,
    la normalisation du gain et la construction du générateur.
    """
    from mdp import average_profit

    pb = Problem.two_env(eps=0.6, m_stock=60, tol=1e-10)
    for lam in ([0.25, 0.6], [0.1, 0.9], [0.35, 0.35]):
        sol = solve_mdp(pb, lam=np.array(lam))
        assert average_profit(pb, np.array(lam), sol.base_stock) == pytest.approx(
            sol.g, abs=1e-9
        )


def test_exact_evaluation_is_maximised_at_the_optimal_base_stock():
    """La politique issue de la value iteration doit battre tous les autres seuils."""
    from mdp import average_profit

    pb = Problem.two_env(eps=0.6, m_stock=40, tol=1e-10)
    lam = np.array([0.25, 0.6])
    best = solve_mdp(pb, lam=lam)
    for s_L in range(0, 15):
        for s_H in range(0, 30):
            assert average_profit(pb, lam, (s_L, s_H)) <= best.g + 1e-9


def test_batch_solver_matches_the_single_solve():
    """`solve_batch` doit reproduire `solve_mdp` sur chaque élément du lot."""
    from mdp import solve_batch

    pb = Problem.two_env(eps=0.6, m_stock=40)
    lams = np.array([[0.25, 0.1, 0.35], [0.6, 0.9, 0.35]])  # (N, B)
    g, base_stock = solve_batch(pb, lams)
    for b in range(lams.shape[1]):
        ref = solve_mdp(pb, lam=lams[:, b])
        # Le lot s'arrête quand *tous* ses éléments ont convergé, la résolution simple
        # dès que le sien l'est : les deux ne coupent pas à la même itération. L'écart
        # attendu est donc borné par le critère d'arrêt, pas nul.
        assert g[b] == pytest.approx(ref.g, abs=pb.tol)
        assert tuple(base_stock[:, b]) == ref.base_stock


def test_batch_solver_refuses_the_setup_cost_model():
    from mdp import solve_batch

    pb = Problem.two_env(eps=0.6, K=0.5, m_stock=40)
    with pytest.raises(ValueError, match="K = 0"):
        solve_batch(pb, np.array([[0.2], [0.5]]))


def test_policy_iteration_agrees_with_value_iteration():
    """Les deux solveurs en lot doivent converger vers le même optimum.

    L'itération sur les politiques est la plus exacte des deux — elle résout
    l'équation d'évaluation au lieu de l'approcher — donc l'écart attendu est borné
    par la tolérance de la *value iteration*, pas par zéro.
    """
    from mdp import solve_batch, solve_batch_vi

    pb = Problem.two_env(eps=0.6, m_stock=40)
    lams = np.array([[0.25, 0.1, 0.35], [0.6, 0.9, 0.35]])
    g_pi, s_pi = solve_batch(pb, lams)
    g_vi, s_vi = solve_batch_vi(pb, lams)

    assert g_pi == pytest.approx(g_vi, abs=pb.tol)
    assert np.array_equal(s_pi, s_vi)


def test_policy_iteration_matches_the_exact_evaluator():
    """Le gain rendu doit être celui de la politique rendue, au chiffre près."""
    from mdp import average_profit, solve_batch

    pb = Problem.two_env(eps=0.6, m_stock=40)
    lams = np.array([[0.25, 0.1], [0.6, 0.9]])
    g, s = solve_batch(pb, lams)
    for b in range(lams.shape[1]):
        assert average_profit(pb, lams[:, b], s[:, b]) == pytest.approx(g[b], abs=1e-12)
