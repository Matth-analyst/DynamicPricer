"""Itération sur les politiques pour la stratégie DP (prix dépendant de l'état)."""

from __future__ import annotations

import numpy as np

from ..params import Problem
from .policies import base_stock_levels

# ---------------------------------------------------------------------------
# Itération sur les politiques pour la stratégie DP (prix dépendant de l'état)
# ---------------------------------------------------------------------------


def _state_dependent_chain(pb: Problem, lam: np.ndarray, s: np.ndarray):
    """Chaîne uniformisée d'une politique DP : `(P (n, n), r (n))`, `n = X·N`.

    Contrairement à `_policy_free_dynamics`, le taux de demande `lam` est ici indexé
    par l'état `(x, e)` et non par le seul environnement : c'est ce qui distingue la
    stratégie DP des quatre autres. Rien n'est donc précalculable d'un tour à l'autre,
    mais DP ne demande qu'une résolution par instance, pas un balayage.
    """
    x, N = pb.states, pb.N
    X = x.size
    n = X * N
    Q = np.asarray(pb.Q, dtype=float).copy()
    np.fill_diagonal(Q, 0.0)
    Lam = np.asarray(pb.Lam, dtype=float)

    P = np.zeros((n, n))
    r = np.zeros(n)
    k = np.arange(n).reshape(X, N)

    for e in range(N):
        rows = k[:, e]
        # Production : une unité au taux µ tant que x < s_e.
        a = (x < s[e]).astype(int)
        P[rows, k[np.clip(np.arange(X) + a, 0, X - 1), e]] += pb.mu
        r[rows] -= pb.mu * pb.c * a

        # Environnement : transitions réelles et masse fictive (éq. 2).
        for j in range(N):
            if j != e:
                P[rows, k[:, j]] += Q[e, j]
        P[rows, rows] += Lam.sum() - Lam[e] + Q.sum() - Q[e].sum()

        # Demande, au taux propre à chaque état ; perdue à la borne basse.
        lam_e = lam[1:, e]
        P[rows[1:], k[:-1, e]] += lam_e
        P[rows[1:], rows[1:]] += Lam[e] - lam_e
        P[rows[0], rows[0]] += Lam[e]
        r[rows[1:]] += lam_e * (1.0 - lam_e / Lam[e]) / pb.a

    r -= np.tile(pb.holding_cost(x)[:, None], (1, N)).reshape(n)
    return P, r


def _greedy_demand(v: np.ndarray, pb: Problem, price_menu: np.ndarray | None):
    """Taux de demande glouton vis-à-vis de `v`, et prix associés (Appendix E, éq. 11).

    Le maximiseur de `r_e(λ) - λ·Δv(x-1,e)` est `λ = Λ_e(1 - a·Δv)/2`, soit
    `p = clip((1 + a·Δv)/(2a), 0, 1/a)`. En `x = x_min` la demande est perdue : le
    prix n'a aucun effet et vaut `NaN` par convention.
    """
    Lam = np.asarray(pb.Lam, dtype=float)
    lam = np.zeros_like(v)
    prices = np.full(v.shape, np.nan)
    delta = v[1:] - v[:-1]

    if price_menu is None:
        p = np.clip((1.0 + pb.a * delta) / (2.0 * pb.a), 0.0, pb.p_max)
    else:
        menu = np.asarray(price_menu, dtype=float)[:, None, None]
        cand = Lam * (1.0 - pb.a * menu)
        p = menu.reshape(-1)[np.argmax(cand * menu - cand * delta[None], axis=0)]

    lam[1:] = Lam * (1.0 - pb.a * p)
    prices[1:] = p
    return lam, prices


#: Seuil absolu de stabilisation des taux de demande (cas bien conditionnés).
_LAM_ABS_TOL = 1e-12

#: Seuil relatif de non-amélioration du gain (filet pour les cas mal conditionnés).
GAIN_STALL_TOL = 1e-12

#: Plafond de tours d'itération sur les politiques : elle converge en une poignée,
#: donc dépasser ce seuil signale un problème plutôt qu'une lenteur.
MAX_POLICY_SWEEPS = 200


def solve_policy_iteration(
    pb: Problem, gamma: float, price_menu, max_sweeps: int, lam_fixed=None
):
    """Itération sur les politiques pour une instance unique, problème déjà normalisé.

    `lam_fixed=None` → stratégie DP : la politique est le couple (base-stocks, fonction
    de prix), et les deux se relisent dans la fonction de valeur — les seuils par
    concavité, les prix par la forme fermée. Avec des prix continus, l'itération est
    une méthode de Newton sur l'équation de Bellman, et l'arrêt porte sur la
    stabilisation conjointe des seuils et des taux.

    `lam_fixed` donné → stratégies S/SB/SP/EDP : seuls les seuils sont améliorés.
    """
    X, N = pb.states.size, pb.N
    n = X * N
    Lam = np.asarray(pb.Lam, dtype=float)

    s = np.zeros(N, dtype=int)
    frozen = lam_fixed is not None
    if frozen:
        lam = np.tile(np.asarray(lam_fixed, dtype=float), (X, 1))
        prices = np.tile((1.0 - np.asarray(lam_fixed) / Lam) / pb.a, (X, 1))
    else:
        lam = np.tile(Lam / 2.0, (X, 1))  # p = p*, l'optimum sans coût de stockage
        prices = np.full((X, N), pb.p_star)
    prices[0] = np.nan  # en `x_min` la demande est perdue : le prix n'a aucun effet

    g_prev = -np.inf
    best = None  # (g, v, s, prices) de la meilleure politique rencontrée
    seen = set()  # politiques déjà visitées, pour détecter les cycles
    for sweep in range(1, max_sweeps + 1):
        P, r = _state_dependent_chain(pb, lam, s)
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = np.eye(n) - P
        M[:n, n] = 1.0
        M[n, 0] = 1.0
        rhs = np.zeros(n + 1)
        rhs[:n] = r
        sol = np.linalg.solve(M, rhs)
        v = sol[:n].reshape(X, N)
        g_n = sol[n]

        if best is None or g_n >= best[0]:
            best = (g_n, v, s.copy(), prices.copy())
        stalled = g_n <= g_prev + GAIN_STALL_TOL * max(1.0, abs(g_n))
        g_prev = max(g_prev, g_n)

        s_next = base_stock_levels(v, pb)
        if frozen:
            lam_next, prices_next, settled = lam, prices, True
        else:
            lam_next, prices_next = _greedy_demand(v, pb, price_menu)
            settled = np.all(np.abs(lam_next - lam) <= _LAM_ABS_TOL)

        if np.array_equal(s_next, s) and (settled or stalled):
            g_b, v_b, s_b, p_b = best
            return g_b * gamma, v_b, tuple(int(i) for i in s_b), p_b, sweep

        # Détection de cycle. Deux politiques distinctes peuvent avoir exactement la
        # même valeur — c'est fréquent avec un menu de prix fini — et l'amélioration
        # bascule alors indéfiniment de l'une à l'autre. On le constate par une
        # politique déjà visitée, ce qui est exact et ne demande aucun seuil.
        #
        # Le critère de non-amélioration ne peut pas jouer ce rôle seul : quand le
        # système est mal conditionné (un coût de pénurie élevé porte `v` à ~1e6),
        # le bruit numérique sur `g` dépasse les progrès des premiers tours et ferait
        # conclure à tort à la convergence. D'où l'exigence conjointe que `s` soit
        # stable ci-dessus.
        key = (s_next.tobytes(), lam_next.tobytes())
        if key in seen:
            g_b, v_b, s_b, p_b = best
            return g_b * gamma, v_b, tuple(int(i) for i in s_b), p_b, sweep
        seen.add(key)

        s, lam, prices = s_next, lam_next, prices_next

    raise RuntimeError(
        f"L'itération sur les politiques n'a pas convergé en {max_sweeps} tours."
    )
