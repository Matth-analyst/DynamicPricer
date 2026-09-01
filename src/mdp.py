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

from dataclasses import dataclass, replace

import numpy as np

import native
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

    `lam` doit être diffusable sur `f.shape[1:]` : un vecteur `(N,)`, ou une matrice
    `(N, B)` pour évaluer `B` vecteurs de prix en une seule passe (Étape 2).
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


def base_stock_levels(v: np.ndarray, pb: Problem) -> np.ndarray:
    """Seuils de production `s_e = min{x : Δv(x,e) <= c}`, vectorisé (Définition 1).

    Renvoie un tableau de forme `v.shape[1:]` — donc `(N,)` pour une résolution
    simple, `(N, B)` pour une résolution en lot.

    Par concavité de `v` (Lemme 1), `{x : Δv(x,e) > c}` est un segment initial : ce
    minimum définit bien un base-stock, on produit ssi `x < s_e`. Les égalités sont
    tranchées en faveur du non-réapprovisionnement, comme le `max{f(x,e), f(x+1,e)-c}`
    du papier.
    """
    stop = (v[1:] - v[:-1]) <= pb.c
    found = stop.any(axis=0)
    first = pb.states[np.argmax(stop, axis=0)]
    return np.where(found, first, pb.x_max)


def extract_base_stock(v: np.ndarray, pb: Problem) -> tuple[int, ...]:
    """Version scalaire de `base_stock_levels`, pour une fonction de valeur `(x, e)`."""
    return tuple(int(s) for s in base_stock_levels(v, pb))


def solve_mdp(
    pb: Problem,
    lam=None,
    price_menu: np.ndarray | None = None,
    force_setup_model: bool = False,
    method: str = "auto",
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
    method :
        `"auto"` (défaut) : itération sur les politiques si `K = 0`, value iteration
        sinon — le modèle à mise en route n'a pas de variante en politiques.
        `"pi"` / `"vi"` forcent l'une ou l'autre, ce qui sert à la validation croisée.
        L'itération sur les politiques est exacte ; la value iteration s'arrête à `tol`.

    Returns
    -------
    MDPSolution, dépaquetable en `(g, v, base_stock)`.
    """
    npb, gamma = pb.normalized()
    setup_model = npb.K > 0 or force_setup_model
    if method not in ("auto", "pi", "vi"):
        raise ValueError("`method` doit valoir 'auto', 'pi' ou 'vi'.")
    if setup_model and method == "pi":
        raise ValueError(
            "L'itération sur les politiques ne couvre pas le modèle à coût de mise en "
            "route (l'état augmenté `σ` n'y est pas traité)."
        )
    if not setup_model and method in ("auto", "pi"):
        lam_n = None if lam is None else np.asarray(lam, dtype=float) / gamma
        if lam_n is not None:
            Lam = np.asarray(npb.Lam, dtype=float)
            if np.any(lam_n < -1e-12) or np.any(lam_n > Lam + 1e-12):
                raise ValueError("Chaque λ_e doit appartenir à [0, Λ_e].")
            lam_n = np.clip(lam_n, 0.0, Lam)
        g, v, base_stock, prices, sweeps = _solve_policy_iteration(
            npb, gamma, price_menu, _MAX_POLICY_SWEEPS, lam_n
        )
        return MDPSolution(
            g=g, v=v, base_stock=base_stock, sS=None,
            prices=None if lam is not None else prices,
            n_iter=sweeps, converged=True,
        )

    A = env_matrix(npb)
    h_vec = npb.holding_cost(npb.states)
    lam_n = None if lam is None else np.asarray(lam, dtype=float) / gamma

    if lam_n is not None:
        Lam = np.asarray(npb.Lam, dtype=float)
        if np.any(lam_n < -1e-12) or np.any(lam_n > Lam + 1e-12):
            raise ValueError("Chaque λ_e doit appartenir à [0, Λ_e].")
        lam_n = np.clip(lam_n, 0.0, Lam)

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


# ---------------------------------------------------------------------------
# Résolution en lot et évaluation exacte de politique (Étape 2)
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
        raise ValueError("`solve_batch` ne couvre que le cas K = 0 (Étapes 1.1/1.3).")
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
# Itération sur les politiques (Étape 2 — remplace la value iteration à prix fixé)
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

    Délègue au moteur natif quand il est disponible (`src/native.py`). Celui-ci exploite
    la structure **bloc-tridiagonale** de `I - P` — la production couple le bloc `x` à
    `x+1`, la demande à `x-1`, l'environnement reste dans le bloc — pour un coût en
    `O(X·N³)` au lieu de `O((X·N)³)`, et répartit les instances sur les cœurs. Le
    chemin Python ci-dessous reste la référence et le repli.

    L'arrêt est **par élément**, et pas seulement sur la stabilité de la politique.
    Deux base-stocks voisins peuvent en effet donner exactement le même gain — c'est
    fréquent quand le coût de stockage est faible — et le départage bascule alors
    indéfiniment entre eux. Un seul élément dans ce cas bloquerait tout le lot, qui
    n'aboutit que lorsque tous ont convergé.

    Le second critère exploite la monotonie du gain propre à l'itération sur les
    politiques : un tour qui n'améliore plus signale la convergence, et l'élément est
    figé sur la meilleure politique rencontrée.
    """
    if native.AVAILABLE:
        return native.solve_batch_truncated(pb, lam_n, gamma, max_sweeps)

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
        stalled = g_n <= g_prev + _GAIN_STALL_TOL * np.maximum(1.0, np.abs(g_n))
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
        raise ValueError("`solve_batch` ne couvre que le cas K = 0 (Étapes 1.1/1.3).")

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
_GAIN_STALL_TOL = 1e-12

#: Plafond de tours d'itération sur les politiques : elle converge en une poignée,
#: donc dépasser ce seuil signale un problème plutôt qu'une lenteur.
_MAX_POLICY_SWEEPS = 200


def _solve_policy_iteration(
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
        stalled = g_n <= g_prev + _GAIN_STALL_TOL * max(1.0, abs(g_n))
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
