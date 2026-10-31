"""Opérateurs de Bellman du modèle (uniformisation, demande, production, mise en route).

Notation des opérateurs :
`T0_unit` = décision de production à l'unité (`T1` dans le papier 2009, éq. 1),
`T2` = transitions d'environnement réelles + fictives (éq. 2),
`Tlam` = arrivée de demande à taux `λ` fixé,
`T_dynamic` = `max_λ Tlam`, l'opérateur de tarification dynamique (§3.3).

L'équation d'optimalité en gain moyen s'écrit, après uniformisation
(`µ + Σ_e Λ_e + Σ_e Σ_{j≠e} q_ej = 1`) :

    v(x,e) + g = -h(x) + µ·T0 v(x,e) + T2 v(x,e) + Tlam v(x,e)

Avec un coût de mise en route (`K > 0`), l'état s'augmente de
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

import numpy as np

from ..params import Problem

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
    """Réapprovisionnement à l'unité, sans coût fixe (éq. 1 du papier).

        `T0 f(x,e) = max{f(x,e), f(x+1,e) - c}`

    La borne haute `x_max` est absorbante : on n'y produit pas (troncature).
    """
    produced = np.full_like(f, -np.inf)
    produced[:-1] = f[1:] - pb.c
    return np.maximum(f, produced)


def T_switch(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Opérateur de commutation marche/arrêt du serveur (modèle à mise en route, `K > 0`).

        `S f(x,e,σ) = max{ f(x,e,0), f(x,e,1) - K·1{σ=0} }`

    `f` est de forme `(x, e, σ)` avec `σ = 0` serveur à l'arrêt, `σ = 1` en marche.
    Arrêter est gratuit ; démarrer coûte `K`, et seulement si le serveur était arrêté.

    Quand `K = 0`, `S f` ne dépend plus de `σ` et vaut `max{f(·,0), f(·,1)}` : le
    modèle se réduit alors exactement au modèle de base, sans `σ` (cf. `_bellman_setup`).
    """
    off, on = f[..., 0], f[..., 1]
    return np.stack([np.maximum(off, on - pb.K), np.maximum(off, on)], axis=-1)


def T_produce(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Opérateur de production unitaire, exercé au taux `µ` (modèle à mise en route).

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
    (modèle de base, avec ou sans backlog) ou `(x, e, σ)` (modèle à mise en route) : seul
    l'axe 1 est l'environnement.
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

    `lam` doit être diffusable sur `f.shape[1:]` : un vecteur `(N,)`, ou une matrice
    `(N, B)` pour évaluer `B` vecteurs de prix en une seule passe (balayage de prix).
    """
    Lam = _by_env(np.asarray(pb.Lam, dtype=float), f.ndim)
    lam = np.asarray(lam, dtype=float)[None]
    rev = lam * (1.0 - lam / Lam) / pb.a  # r_e(λ) = λ·p_e(λ), forme linéaire

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

    `price_menu` restreint le choix à un ensemble fini de prix (Table 2 du papier) ;
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
