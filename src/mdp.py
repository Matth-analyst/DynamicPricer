"""Solveur MDP générique : opérateurs de base et value iteration (Étape 1 de la spec).

Notation des opérateurs (cf. note de notation §Étape 1.1 de la spec) :
`T0_unit` = décision de production à l'unité (`T1` dans le papier 2009, éq. 1),
`T2` = transitions d'environnement réelles + fictives (éq. 2),
`Tlam` = arrivée de demande à taux `λ` fixé,
`T_dynamic` = `max_λ Tlam`, l'opérateur de tarification dynamique (§3.3).

L'équation d'optimalité en gain moyen s'écrit, après uniformisation
(`µ + Σ_e Λ_e + Σ_e Σ_{j≠e} q_ej = 1`) :

    v(x,e) + g = -h(x) + µ·T0 v(x,e) + T2 v(x,e) + Tlam v(x,e)

Avec un coût de mise en route (`K > 0`, Étape 1.2), l'état s'augmente de
l'indicateur `σ ∈ {0, 1}` (serveur arrêté / en marche), l'opérateur `T0` se scinde
en production `T_produce` et commutation `T_switch`, et l'équation devient :

    v(x,e,σ) + g = T_switch( -h(x) + µ·T_produce v + T2 v + Tlam v )(x,e,σ)

Avec `K = 0` cette forme se réduit **exactement** à la précédente.

Les masses des trois opérateurs somment exactement à 1, ce qui en fait un
opérateur de Bellman en gain moyen propre (cf. `test_operator_masses_sum_to_one`).

Toutes les fonctions de ce module travaillent sur le problème **normalisé**
(`Problem.normalized()`). `solve_mdp` fait la normalisation et la conversion
inverse du gain, et s'utilise donc avec un `Problem` en unités réelles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from params import Problem

# ---------------------------------------------------------------------------
# Précalculs dépendant uniquement du problème
# ---------------------------------------------------------------------------


def env_matrix(pb: Problem) -> np.ndarray:
    """Matrice `A` telle que `T2 f = f @ A`, de forme `(N, N)`.

    `A[j,e] = q_ej` pour `j != e`, et `A[e,e] = Σ_{i≠e} Λ_i + Σ_{i≠e} Σ_{j≠i} q_ij`,
    c'est-à-dire la masse des transitions fictives de l'environnement `e` (éq. 2).
    """
    N = pb.N
    Q = np.asarray(pb.Q, dtype=float).copy()
    np.fill_diagonal(Q, 0.0)
    Lam = np.asarray(pb.Lam, dtype=float)

    A = np.zeros((N, N))
    for e in range(N):
        for j in range(N):
            if j != e:
                A[j, e] = Q[e, j]
        # Transitions fictives : demandes des autres environnements + transitions
        # sortant des autres environnements.
        A[e, e] = Lam.sum() - Lam[e] + Q.sum() - Q[e].sum()
    return A


# ---------------------------------------------------------------------------
# Opérateurs
# ---------------------------------------------------------------------------


def T0_unit(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Réapprovisionnement à l'unité, sans coût fixe (éq. 1 du papier, Étape 1.1).

        `T0 f(x,e) = max{f(x,e), f(x+1,e) - c}`

    La borne haute `x_max` est absorbante : on n'y produit pas (troncature).
    """
    produced = np.full_like(f, -np.inf)
    produced[:-1] = f[1:] - pb.c
    return np.maximum(f, produced)


def T_switch(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Opérateur de commutation marche/arrêt du serveur (Étape 1.2).

        `S f(x,e,σ) = max{ f(x,e,0), f(x,e,1) - K·1{σ=0} }`

    `f` est de forme `(x, e, σ)` avec `σ = 0` serveur à l'arrêt, `σ = 1` en marche.
    Arrêter est gratuit ; démarrer coûte `K`, et seulement si le serveur était arrêté.

    Quand `K = 0`, `S f` ne dépend plus de `σ` et vaut `max{f(·,0), f(·,1)}` : le
    modèle se réduit alors exactement à celui de l'Étape 1.1 (cf. `_bellman_setup`).
    """
    off, on = f[..., 0], f[..., 1]
    return np.stack([np.maximum(off, on - pb.K), np.maximum(off, on)], axis=-1)


def T_produce(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Opérateur de production unitaire, exercé au taux `µ` (Étape 1.2).

        `P f(x,e,1) = f(x+1,e,1) - c`   une unité sort du serveur en marche
        `P f(x,e,0) = f(x,e,0)`         serveur arrêté : rien ne se passe

    La capacité reste celle du papier — une unité par événement `µ` — contrairement
    à un réapprovisionnement par lots, qui la rendrait illimitée.
    """
    out = f.copy()
    out[:-1, :, 1] = f[1:, :, 1] - pb.c
    return out


def _by_env(vec: np.ndarray, ndim: int) -> np.ndarray:
    """Met un vecteur indexé par environnement en forme pour diffuser sur l'axe 1.

    Les opérateurs acceptent indifféremment une fonction de valeur `(x, e)`
    (Étapes 1.1/1.3) ou `(x, e, σ)` (Étape 1.2) : seul l'axe 1 est l'environnement.
    """
    return vec.reshape((1, -1) + (1,) * (ndim - 2))


def T2(f: np.ndarray, pb: Problem, A: np.ndarray | None = None) -> np.ndarray:
    """Transitions d'environnement, réelles et fictives, générique en `N` (éq. 2).

    `T2 f(x,e) = Σ_{j≠e} q_ej f(x,j) + (Σ_{i≠e} Λ_i + Σ_{i≠e} Σ_{j≠i} q_ij) f(x,e)`

    L'environnement n'interagit pas avec l'état du serveur : l'opérateur agit à
    l'identique sur chaque tranche `σ`.
    """
    A = env_matrix(pb) if A is None else A
    return np.moveaxis(np.tensordot(f, A, axes=([1], [0])), -1, 1)


def Tlam(f: np.ndarray, pb: Problem, lam) -> np.ndarray:
    """Arrivée de demande au taux `λ_e` fixé, un taux par environnement (§3.1).

        `Tlam f(x,e) = r_e(λ_e) + λ_e f(x-1,e) + (Λ_e - λ_e) f(x,e)`  si `x > x_min`
        `Tlam f(x,e) = Λ_e f(x,e)`                                     si `x = x_min`

    À la borne basse la demande est perdue : c'est le cas `x = 0` du papier
    (lost sales) et, quand `m_backlog > 0`, l'artefact de troncature du backlog.
    """
    lam = np.asarray(lam, dtype=float)
    Lam = _by_env(np.asarray(pb.Lam, dtype=float), f.ndim)
    rev = _by_env(np.array([pb.revenue(float(lam[e]), e) for e in range(pb.N)]), f.ndim)
    lam = _by_env(lam, f.ndim)

    out = Lam * f
    # Δf(x-1,e) = f(x,e) - f(x-1,e) : bénéfice de l'unité vendue.
    out[1:] += rev - lam * (f[1:] - f[:-1])
    return out


def T_dynamic(
    f: np.ndarray, pb: Problem, price_menu: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Opérateur de tarification dynamique : `T f(x,e) = max_λ Tlam f(x,e)` (§3.3).

    En réécrivant `Tlam f(x,e) = Λ_e f(x,e) + max_λ [r_e(λ) - λ·Δf(x-1,e)]`, la
    demande linéaire donne la forme fermée de l'Appendix E (éq. 11) :

        `p(x,e) = clip((1 + a·Δf(x-1,e)) / (2a), 0, 1/a)`

    `price_menu` restreint le choix à un ensemble fini de prix (Étape 4, Table 2) ;
    par défaut le prix est continu.

    Returns
    -------
    (valeur, prix) : le prix vaut `NaN` en `x = x_min`, où la demande est perdue
    et où le prix n'a donc aucun effet (le papier ne rapporte les prix DP que
    pour `x >= 1`).
    """
    Lam = _by_env(np.asarray(pb.Lam, dtype=float), f.ndim)
    out = Lam * f
    prices = np.full(f.shape, np.nan)

    delta = f[1:] - f[:-1]  # Δf(x-1,e) pour x = x_min+1 .. x_max

    if price_menu is None:
        p = np.clip((1.0 + pb.a * delta) / (2.0 * pb.a), 0.0, pb.p_max)
        lam = Lam * (1.0 - pb.a * p)
        out[1:] += lam * p - lam * delta
        prices[1:] = p
    else:
        menu = np.asarray(price_menu, dtype=float).reshape((-1,) + (1,) * f.ndim)
        # (n_prix, x, e[, σ]) : on évalue le menu complet puis on prend l'argmax.
        lam = Lam[None] * (1.0 - pb.a * menu)
        gains = lam * menu - lam * delta[None]
        best = np.argmax(gains, axis=0)
        out[1:] += np.take_along_axis(gains, best[None], axis=0)[0]
        prices[1:] = menu.reshape(-1)[best]

    return out, prices


# ---------------------------------------------------------------------------
# Value iteration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MDPSolution:
    """Résultat de `solve_mdp`."""

    g: float
    """Gain moyen de long terme, **en unités de temps réelles** (déjà remis à l'échelle)."""
    v: np.ndarray
    """Fonction de valeur relative, de forme `(x, e)`, normalisée à `v[0,0] = 0`."""
    base_stock: tuple[int, ...] | None
    """Seuils `s_e` : on produit ssi `x < s_e`. `None` si `K > 0` (voir `sS`)."""
    sS: tuple[tuple[int, int], ...] | None
    """Politique `(s_e, S_e)` quand `K > 0` ; `None` sinon.

    Avec un coût fixe, `v` n'est plus concave mais seulement K-concave : la notion
    de base-stock simple ne s'applique plus, d'où ce champ distinct.
    """
    prices: np.ndarray | None
    """Prix `p(x,e)` pour la stratégie DP ; `None` si les taux `λ` étaient imposés."""
    n_iter: int
    converged: bool

    def __iter__(self):
        """Permet le dépaquetage `g, v, s = solve_mdp(...)` prescrit par la spec."""
        return iter((self.g, self.v, self.base_stock))


def _demand_term(v, pb, lam, price_menu):
    if lam is None:
        return T_dynamic(v, pb, price_menu)
    return Tlam(v, pb, lam), None


def _bellman_unit(v, pb, A, lam, price_menu, h_vec):
    """Opérateur de Bellman de l'Étape 1.1 : `-h + µ·T0 + T2 + Tλ`, sur `(x, e)`."""
    demand, prices = _demand_term(v, pb, lam, price_menu)
    return -h_vec[:, None] + pb.mu * T0_unit(v, pb) + T2(v, pb, A) + demand, prices, None


def _bellman_setup(v, pb, A, lam, price_menu, h_vec):
    """Opérateur de Bellman de l'Étape 1.2 : `S(-h + µ·P + T2 + Tλ)`, sur `(x, e, σ)`.

    Renvoie aussi `d(x,e) = inner(x,e,1) - inner(x,e,0)`, le gain net de tourner
    **hors** coût de mise en route, dont se déduisent les deux seuils : on produit
    si `σ=1` et `d > 0`, on redémarre si `σ=0` et `d > K`.

    Avec `K = 0`, `T_switch` devient `max{inner(·,0), inner(·,1)}` : les deux tranches
    `σ` se confondent, `inner(·,1) = -h + µ(v(x+1)-c) + …` et `inner(·,0) = -h + µ·v(x) + …`,
    donc leur maximum est littéralement `-h + µ·T0_unit v + T2 v + Tλ v`. La réduction
    à l'Étape 1.1 est donc exacte, pas approchée.
    """
    demand, prices = _demand_term(v, pb, lam, price_menu)
    inner = -h_vec[:, None, None] + pb.mu * T_produce(v, pb) + T2(v, pb, A) + demand
    return T_switch(inner, pb), prices, inner[..., 1] - inner[..., 0]


def extract_base_stock(v: np.ndarray, pb: Problem) -> tuple[int, ...]:
    """Seuils de production `s_e` : `s_e = min{x : Δv(x,e) <= c}` (Définition 1).

    Par concavité de `v` (Lemme 1), l'ensemble `{x : Δv(x,e) > c}` est un segment
    initial, donc ce minimum définit bien un base-stock : on produit ssi `x < s_e`.
    Les égalités sont tranchées en faveur du non-réapprovisionnement, comme le
    `max{f(x,e), f(x+1,e) - c}` du papier.
    """
    delta = v[1:] - v[:-1]
    produce = delta > pb.c
    s = []
    for e in range(pb.N):
        idx = np.flatnonzero(~produce[:, e])
        s.append(int(pb.states[idx[0]]) if idx.size else int(pb.x_max))
    return tuple(s)


def solve_mdp(
    pb: Problem,
    lam=None,
    price_menu: np.ndarray | None = None,
    force_setup_model: bool = False,
) -> MDPSolution:
    """Value iteration relative en gain moyen (Appendix E : 5 décimales).

    Parameters
    ----------
    pb :
        Le problème, **en unités réelles** — la normalisation est faite ici.
    lam :
        Vecteur des taux de demande imposés, un par environnement (stratégies
        S, SB, SP, EDP). `None` → stratégie DP : le taux est optimisé dans chaque
        état. Les taux sont exprimés dans les unités de `pb` et normalisés ici.
    price_menu :
        Menu de prix fini pour la stratégie DP (Étape 4). `None` → prix continu.
    force_setup_model :
        Force l'espace d'état augmenté `(x, e, σ)` même quand `K = 0`. Sert au test
        de non-régression de l'Étape 1.2 : les deux modèles doivent alors coïncider
        exactement. Sans intérêt en usage normal — le chemin `(x, e)` est plus rapide.

    Returns
    -------
    MDPSolution, dépaquetable en `(g, v, base_stock)`.
    """
    npb, gamma = pb.normalized()
    A = env_matrix(npb)
    h_vec = npb.holding_cost(npb.states)
    lam_n = None if lam is None else np.asarray(lam, dtype=float) / gamma

    if lam_n is not None:
        Lam = np.asarray(npb.Lam, dtype=float)
        if np.any(lam_n < -1e-12) or np.any(lam_n > Lam + 1e-12):
            raise ValueError("Chaque λ_e doit appartenir à [0, Λ_e].")
        lam_n = np.clip(lam_n, 0.0, Lam)

    setup_model = npb.K > 0 or force_setup_model
    shape = (npb.states.size, npb.N, 2) if setup_model else (npb.states.size, npb.N)
    bellman = _bellman_setup if setup_model else _bellman_unit

    v = np.zeros(shape)
    g_n, prices, d, converged, it = 0.0, None, None, False, 0

    for it in range(1, npb.max_iter + 1):
        w, prices, d = bellman(v, npb, A, lam_n, price_menu, h_vec)
        diff = w - v
        lo, hi = diff.min(), diff.max()
        # Estimation du gain par le milieu de l'intervalle : l'erreur est
        # majorée par la demi-amplitude (Puterman, value iteration relative).
        g_n = 0.5 * (lo + hi)
        v = w - w.flat[0]
        if hi - lo < npb.tol:
            converged = True
            break

    return MDPSolution(
        g=g_n * gamma,
        v=v,
        base_stock=None if setup_model else extract_base_stock(v, npb),
        sS=extract_setup_policy(d, npb) if setup_model else None,
        prices=prices,
        n_iter=it,
        converged=converged,
    )


def extract_setup_policy(d: np.ndarray, pb: Problem) -> tuple[tuple[int, int], ...]:
    """Extrait la bande d'hystérésis `(s_e, S_e)` du modèle à mise en route (Étape 1.2).

    `d(x,e) = inner(x,e,1) - inner(x,e,0)` est le gain net de faire tourner le
    serveur, **hors** coût de mise en route. La décision optimale s'en déduit :

        serveur en marche (`σ=1`) : on continue ssi `d(x,e) > 0`
        serveur arrêté   (`σ=0`) : on redémarre ssi `d(x,e) > K`

    D'où `S_e = min{x : d(x,e) <= 0}` (seuil d'arrêt) et `s_e = min{x : d(x,e) <= K}`
    (seuil de redémarrage), avec `s_e <= S_e` puisque `K >= 0`.

    `d` doit être décroissante en `x` pour que ces deux seuils définissent bien une
    politique de type bande ; l'Étape 1.2 demande de le **constater** plutôt que de
    le supposer, d'où le `ValueError` en cas de violation.
    """
    tol = 1e-9
    # La dernière ligne `x_max` est écartée : la troncature y interdit de produire,
    # ce qui y fait remonter `d` artificiellement. Les deux seuils doivent de toute
    # façon être strictement intérieurs à la grille.
    interior = d[:-1]
    if np.any(np.diff(interior, axis=0) > tol):
        raise ValueError(
            "d(x,e) n'est pas décroissante en x : la politique optimale n'est pas "
            "une bande d'hystérésis à deux seuils."
        )

    def first_below(col, threshold, name, e):
        idx = np.flatnonzero(col <= threshold + tol)
        if idx.size == 0:
            raise ValueError(
                f"Seuil {name} hors grille en environnement {e} : augmenter `m_stock` "
                "jusqu'à ce que la troncature ne morde plus (cf. Appendix E)."
            )
        return int(pb.states[idx[0]])

    return tuple(
        (
            first_below(interior[:, e], pb.K, "s_e", e),
            first_below(interior[:, e], 0.0, "S_e", e),
        )
        for e in range(pb.N)
    )
