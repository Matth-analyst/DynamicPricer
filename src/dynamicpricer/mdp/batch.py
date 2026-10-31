"""Résolution en lot pour `B` vecteurs de prix fixés (`K = 0`) et évaluation exacte de politique."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .. import native
from ..params import Problem
from .dynamic_pricing import GAIN_STALL_TOL
from .operators import T2, T0_unit, Tlam, env_matrix
from .policies import base_stock_levels

# ---------------------------------------------------------------------------
# Résolution en lot et évaluation exacte de politique
# ---------------------------------------------------------------------------


#: Budget mémoire des générateurs empilés de `average_profit` (~300 Mo).
_MEMORY_BUDGET_BYTES = 300_000_000


def solve_batch_vi(pb: Problem, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration pour `B` vecteurs de prix simultanément (`K = 0`, prix fixés).

    Les opérateurs étant vectorisés sur les axes de queue, une résolution en lot
    coûte à peine plus qu'une résolution simple : c'est ce qui rend le balayage
    `N`-dimensionnel de la stratégie EDP praticable (100^N programmes dynamiques
    dans l'Appendix E du papier).

    Parameters
    ----------
    lam : de forme `(N, B)`, les taux de demande imposés, en unités réelles.

    Returns
    -------
    (g, base_stock) : gains moyens `(B,)` en unités réelles, et seuils `(N, B)`.
    """
    npb, gamma = pb.normalized()
    if npb.K > 0:
        raise ValueError("`solve_batch` ne couvre que le cas K = 0.")
    A = env_matrix(npb)
    h_vec = npb.holding_cost(npb.states)[:, None, None]
    lam_n = np.asarray(lam, dtype=float) / gamma

    Lam = np.asarray(npb.Lam, dtype=float)[:, None]
    if np.any(lam_n < -1e-12) or np.any(lam_n > Lam + 1e-12):
        raise ValueError("Chaque λ_e doit appartenir à [0, Λ_e].")
    lam_n = np.clip(lam_n, 0.0, Lam)

    v = np.zeros((npb.states.size, npb.N, lam_n.shape[1]))
    g_n = np.zeros(lam_n.shape[1])

    for _ in range(npb.max_iter):
        w = -h_vec + npb.mu * T0_unit(v, npb) + T2(v, npb, A) + Tlam(v, npb, lam_n)
        diff = w - v
        lo, hi = diff.min(axis=(0, 1)), diff.max(axis=(0, 1))
        g_n = 0.5 * (lo + hi)
        v = w - w[0, 0]
        if np.max(hi - lo) < npb.tol:
            break
    else:
        raise RuntimeError(
            f"{int(np.sum(hi - lo >= npb.tol))} élément(s) du lot n'ont pas convergé en "
            f"{npb.max_iter} itérations. Cause habituelle : un prix dégénéré dans la "
            "grille (cf. `Problem.search_price_grid`)."
        )

    return g_n * gamma, base_stock_levels(v, npb)


def average_profit(pb: Problem, lam: np.ndarray, base_stock) -> np.ndarray:
    """Gain moyen **exact** d'une politique base-stock donnée, par la loi stationnaire.

    Contrairement à `solve_mdp`, cette fonction n'optimise rien : elle évalue la
    politique « produire ssi `x < s_e` » aux taux `lam` imposés. C'est ce dont ont
    besoin les stratégies S et SB, dont le base-stock est **contraint** identique
    dans tous les environnements et ne peut donc pas sortir d'une value iteration.

    On résout `π Q = 0` sur le générateur en temps continu (pas d'uniformisation
    nécessaire ici), puis `g = Σ π(x,e)·[revenu - coût de stockage - coût de production]`.
    Le résultat est exact à la précision machine, sans critère de convergence.

    Parameters
    ----------
    lam : `(N,)` ou `(B, N)` — taux de demande imposés, en unités réelles.
    base_stock : `(N,)` — un seuil par environnement, commun à tout le lot.

    Returns
    -------
    Gain moyen : scalaire si `lam` est `(N,)`, sinon `(B,)`.
    """
    s = np.asarray(base_stock, dtype=int)
    lam = np.asarray(lam, dtype=float)
    single = lam.ndim == 1
    lam = np.atleast_2d(lam)
    B, N = lam.shape

    # Les générateurs empilés sont en O(B·n²) : on découpe le lot pour borner la
    # mémoire, ce qui est indispensable au balayage N-dimensionnel de la stratégie SB.
    n_states = (max(int(s.max()), pb.x_min) + 1 - pb.x_min) * N
    max_batch = max(1, _MEMORY_BUDGET_BYTES // max(1, 24 * n_states * n_states))
    if B > max_batch:
        return np.concatenate(
            [average_profit(pb, lam[i : i + max_batch], s) for i in range(0, B, max_batch)]
        )

    # L'espace d'état utile s'arrête au plus grand seuil : au-delà, les états sont
    # transitoires et de mesure stationnaire nulle.
    x = np.arange(pb.x_min, max(int(s.max()), pb.x_min) + 1)
    X, n = x.size, x.size * N
    Q = np.asarray(pb.Q, dtype=float).copy()
    np.fill_diagonal(Q, 0.0)

    gen = np.zeros((B, n, n))
    k = np.arange(n).reshape(X, N)  # index aplati de l'état (x, e)

    produces = x[:, None] < s[None, :]  # (X, N)
    for e in range(N):
        # Production : (x,e) -> (x+1,e) au taux µ tant que x < s_e.
        up = produces[:-1, e]
        gen[:, k[:-1, e][up], k[1:, e][up]] += pb.mu
        # Demande servie : (x,e) -> (x-1,e) au taux λ_e dès que x > x_min.
        gen[:, k[1:, e], k[:-1, e]] += lam[:, e][:, None]
        # Environnement : (x,e) -> (x,j).
        for j in range(N):
            if j != e:
                gen[:, k[:, e], k[:, j]] += Q[e, j]
    diag = np.arange(n)
    gen[:, diag, diag] = -gen.sum(axis=2)

    # π Q = 0 avec Σπ = 1 : on remplace une équation par la normalisation.
    M = np.transpose(gen, (0, 2, 1)).copy()
    M[:, -1, :] = 1.0
    rhs = np.zeros((B, n, 1))
    rhs[:, -1, 0] = 1.0
    pi = np.linalg.solve(M, rhs)[..., 0].reshape(B, X, N)

    price = (1.0 - lam / np.asarray(pb.Lam)) / pb.a  # (B, N)
    revenue = lam * price  # (B, N), perçu dès que la demande est servie
    served = (x > pb.x_min)[None, :, None]
    reward = revenue[:, None, :] * served
    reward = reward - pb.holding_cost(x)[None, :, None] - pb.mu * pb.c * produces[None]

    g = (pi * reward).sum(axis=(1, 2))
    return float(g[0]) if single else g


# ---------------------------------------------------------------------------
# Itération sur les politiques à prix fixés (remplace la value iteration)
# ---------------------------------------------------------------------------


def _policy_free_dynamics(pb: Problem, lam: np.ndarray):
    """Part de la chaîne qui ne dépend **pas** de la politique : environnement + demande.

    Elle est constante d'un tour d'itération sur les politiques à l'autre ; seul le
    terme de production change. La calculer une fois évite de reconstruire tout le
    générateur à chaque tour.

    Renvoie `(P (B, n, n), r (B, n))` avec `n = X·N` l'état aplati `(x, e)`.
    """
    x, N = pb.states, pb.N
    X, B = x.size, lam.shape[1]
    n = X * N
    Q = np.asarray(pb.Q, dtype=float).copy()
    np.fill_diagonal(Q, 0.0)
    Lam = np.asarray(pb.Lam, dtype=float)

    P = np.zeros((B, n, n))
    r = np.zeros((B, n))
    k = np.arange(n).reshape(X, N)

    for e in range(N):
        rows = k[:, e]
        for j in range(N):
            if j != e:
                P[:, rows, k[:, j]] += Q[e, j]
        # Masse fictive de l'environnement e (éq. 2).
        P[:, rows, rows] += Lam.sum() - Lam[e] + Q.sum() - Q[e].sum()

        # Demande : servie dès que `x > x_min`, perdue à la borne basse.
        lam_e = lam[e][:, None]
        P[:, rows[1:], k[:-1, e]] += lam_e
        P[:, rows[1:], rows[1:]] += Lam[e] - lam_e
        P[:, rows[0], rows[0]] += Lam[e]
        r[:, rows[1:]] += lam_e * (1.0 - lam_e / Lam[e]) / pb.a

    r -= np.tile(pb.holding_cost(x)[:, None], (1, N)).reshape(1, n)
    return P, r


def _policy_evaluation(pb: Problem, s: np.ndarray, P_free: np.ndarray, r_free: np.ndarray):
    """Résout `(I - P)v + g·1 = r` avec `v[0] = 0`, pour un lot de politiques base-stock.

    `s` est `(N, B)` ; la politique n'intervient que par le terme de production, ajouté
    ici à la part libre précalculée. Exact à la précision machine : contrairement à la
    value iteration, aucun critère de convergence n'intervient.

    Renvoie `(g (B,), v (X, N, B))`.
    """
    X, N = pb.states.size, pb.N
    B, n, _ = P_free.shape
    k = np.arange(n).reshape(X, N)
    rows_x = np.arange(X)

    M = np.zeros((B, n + 1, n + 1))
    M[:, :n, :n] = -P_free
    r = r_free.copy()

    produces = pb.states[:, None, None] < s[None]  # (X, N, B)
    idx_b = np.arange(B)[:, None]
    for e in range(N):
        a = produces[:, e].T.astype(int)  # (B, X)
        cols = k[np.clip(rows_x[None] + a, 0, X - 1), e]
        rows = np.broadcast_to(k[:, e], (B, X))
        # Chaque couple (b, ligne) n'apparaît qu'une fois : l'indexation avancée
        # accumule sans collision.
        M[idx_b, rows, cols] -= pb.mu
        r[:, k[:, e]] -= pb.mu * pb.c * a

    diag = np.arange(n)
    M[:, diag, diag] += 1.0  # I - P
    M[:, :n, n] = 1.0  # coefficient de g
    M[:, n, 0] = 1.0  # normalisation v[0] = 0

    rhs = np.zeros((B, n + 1, 1))
    rhs[:, :n, 0] = r
    sol = np.linalg.solve(M, rhs)[..., 0]
    v = sol[:, :n].reshape(B, X, N)
    return sol[:, n], np.moveaxis(v, 0, -1)


#: Taille de la sonde qui calibre la troncature d'un balayage.
_PROBE_SIZE = 32

#: Marge ajoutée au plus grand seuil observé par la sonde.
_PROBE_MARGIN = 4


def _probe_truncation(pb: Problem, lam_n: np.ndarray, gamma: float, max_sweeps: int) -> int:
    """Calibre la troncature du balayage sur un petit échantillon, à pleine taille.

    La troncature pilote le coût — cubique en la taille de l'état — mais la choisir
    trop courte oblige à tout refaire. Plutôt que de partir petit et doubler à
    l'aveugle, on résout d'abord quelques dizaines de vecteurs de prix sans
    troncature agressive et on en déduit le niveau utile.

    L'échantillon inclut la **première** colonne de la grille, celle des prix les plus
    bas : c'est la demande la plus forte, donc le plus grand base-stock. La boucle
    appelante conserve malgré tout un filet de sécurité par doublement.
    """
    B = lam_n.shape[1]
    idx = np.unique(np.r_[0, np.linspace(0, B - 1, min(_PROBE_SIZE, B)).astype(int)])
    _, s = _solve_batch_truncated(pb, lam_n[:, idx], gamma, max_sweeps)
    return int(min(s.max() + _PROBE_MARGIN, pb.m_stock))


def _solve_batch_truncated(pb: Problem, lam_n: np.ndarray, gamma: float, max_sweeps: int):
    """Itération sur les politiques à troncature fixée, sur le problème normalisé.

    Délègue au moteur natif quand il est disponible (`dynamicpricer/native.py`). Celui-ci
    exploite la structure **bloc-tridiagonale** de `I - P` — la production couple le bloc
    `x` à `x+1`, la demande à `x-1`, l'environnement reste dans le bloc — pour un coût en
    `O(X·N³)` au lieu de `O((X·N)³)`, et répartit les instances sur les cœurs. Sinon,
    `solve_batch_truncated_python` sert de repli.
    """
    if native.AVAILABLE:
        return native.solve_batch_truncated(pb, lam_n, gamma, max_sweeps)
    return solve_batch_truncated_python(pb, lam_n, gamma, max_sweeps)


def solve_batch_truncated_python(
    pb: Problem, lam_n: np.ndarray, gamma: float, max_sweeps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Moteur Python de référence : itération sur les politiques à troncature fixée.

    `pb` est le problème **normalisé**, `lam_n` les taux normalisés `(N, B)`. Renvoie
    `(gains (B,) en unités réelles, base_stock (N, B))`. C'est la version contre laquelle
    le moteur natif est testé (`tests/test_native.py`) : elle ne passe jamais par le natif.

    L'arrêt est **par élément**, et pas seulement sur la stabilité de la politique.
    Deux base-stocks voisins peuvent en effet donner exactement le même gain — c'est
    fréquent quand le coût de stockage est faible — et le départage bascule alors
    indéfiniment entre eux. Un seul élément dans ce cas bloquerait tout le lot, qui
    n'aboutit que lorsque tous ont convergé.

    Le second critère exploite la monotonie du gain propre à l'itération sur les
    politiques : un tour qui n'améliore plus signale la convergence, et l'élément est
    figé sur la meilleure politique rencontrée.
    """
    P_free, r_free = _policy_free_dynamics(pb, lam_n)
    B = lam_n.shape[1]
    s = np.zeros((pb.N, B), dtype=int)
    s_best = s.copy()
    g_best = np.full(B, -np.inf)
    g_prev = np.full(B, -np.inf)
    done = np.zeros(B, dtype=bool)

    for _ in range(max_sweeps):
        g_n, v = _policy_evaluation(pb, s, P_free, r_free)
        better = g_n >= g_best
        s_best[:, better] = s[:, better]
        g_best[better] = g_n[better]

        s_next = base_stock_levels(v, pb)
        stable = (s_next == s).all(axis=0)
        stalled = g_n <= g_prev + GAIN_STALL_TOL * np.maximum(1.0, np.abs(g_n))
        done |= stable | stalled
        if done.all():
            return g_best * gamma, s_best
        g_prev = g_n
        s = np.where(done[None], s, s_next)

    raise RuntimeError(
        f"L'itération sur les politiques n'a pas convergé en {max_sweeps} tours "
        f"({int((~done).sum())} élément(s) encore actifs)."
    )


def solve_batch(
    pb: Problem, lam: np.ndarray, max_sweeps: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Itération sur les politiques pour `B` vecteurs de prix simultanément.

    À prix fixé, la politique optimale est entièrement décrite par `N` entiers — les
    base-stocks (Théorème 1). L'espace des politiques est donc minuscule et
    l'itération sur les politiques converge en une poignée de tours, là où la value
    iteration en demande plusieurs milliers : avec `q = 0.01`, les transitions
    d'environnement sont deux ordres de grandeur plus rares que les arrivées de
    demande, le mélange de la chaîne est très lent et le module de contraction en
    span vaut à peu près `1 - O(q)`.

    Chaque tour évalue la politique **exactement** (`_policy_evaluation`) puis
    l'améliore, l'amélioration se réduisant à relire les seuils dans la nouvelle
    fonction de valeur (`base_stock_levels`) — par concavité, la règle « produire ssi
    `Δv(x,e) > c` » *est* une politique base-stock.

    La troncature de l'espace d'état est ajustée automatiquement, en partant petit et
    en doublant tant qu'un seuil touche le bord. Ce n'est pas une approximation : sous
    une politique base-stock de niveau `s_e`, les états `x > s_e` sont transitoires et
    n'influencent ni le gain moyen ni le seuil suivant. Le coût d'une résolution étant
    cubique en la taille de l'état, c'est le poste d'économie principal.

    Parameters
    ----------
    lam : `(N, B)`, taux de demande imposés, en unités réelles.

    Returns
    -------
    (g, base_stock) : gains moyens `(B,)` en unités réelles, seuils `(N, B)`.
    """
    npb, gamma = pb.normalized()
    if npb.K > 0:
        raise ValueError("`solve_batch` ne couvre que le cas K = 0.")

    lam_n = np.asarray(lam, dtype=float) / gamma
    Lam = np.asarray(npb.Lam, dtype=float)[:, None]
    if np.any(lam_n < -1e-12) or np.any(lam_n > Lam + 1e-12):
        raise ValueError("Chaque λ_e doit appartenir à [0, Λ_e].")
    lam_n = np.clip(lam_n, 0.0, Lam)

    m = _probe_truncation(npb, lam_n, gamma, max_sweeps)
    while True:
        trial = replace(npb, m_stock=m)
        n = (trial.states.size * trial.N) + 1
        budget = max(1, _MEMORY_BUDGET_BYTES // max(1, 16 * n * n))
        B = lam_n.shape[1]

        if B > budget:
            parts = [
                _solve_batch_truncated(trial, lam_n[:, i : i + budget], gamma, max_sweeps)
                for i in range(0, B, budget)
            ]
            g = np.concatenate([p[0] for p in parts])
            s = np.concatenate([p[1] for p in parts], axis=1)
        else:
            g, s = _solve_batch_truncated(trial, lam_n, gamma, max_sweeps)

        # Un seuil au contact du bord signalerait une troncature qui mord.
        if s.max() < m - 1 or m >= npb.m_stock:
            return g, s
        m = min(2 * m, npb.m_stock)
