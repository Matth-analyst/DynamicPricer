"""Solveur MDP générique : opérateurs de base et value iteration (Étape 1 de la spec).

Notation des opérateurs (cf. note de notation §Étape 1.1 de la spec) :
`T0` = décision de réapprovisionnement (`T1` dans le papier 2009, éq. 1),
`T2` = transitions d'environnement réelles + fictives (éq. 2),
`Tlam` = arrivée de demande à taux `λ` fixé,
`T_dynamic` = `max_λ Tlam`, l'opérateur de tarification dynamique (§3.3).

L'équation d'optimalité en gain moyen s'écrit, après uniformisation
(`µ + Σ_e Λ_e + Σ_e Σ_{j≠e} q_ej = 1`) :

    v(x,e) + g = -h(x) + µ·T0 v(x,e) + T2 v(x,e) + Tlam v(x,e)

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


def T0_setup(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Réapprovisionnement par lots avec coût fixe (Étape 1.2, hors périmètre du papier).

        `T0 f(x,e) = max{f(x,e), max_{y>x}[f(y,e) - c(y-x) - K]}`

    Interprétation : des opportunités de réapprovisionnement arrivent au taux `µ`,
    et l'on peut alors commander une quantité **quelconque** `y - x` au coût
    `c(y-x) + K`. Ce n'est ni le modèle du papier, ni le coût de mise en route d'un
    serveur unique produisant à l'unité (qui exigerait d'augmenter l'espace d'état
    d'un indicateur marche/arrêt).

    ATTENTION — cet opérateur ne dégénère **pas** en `T0_unit` quand `K = 0`.
    Le maximiseur de `max_{y>x}[f(y,e) - c(y-x)]` est le sommet de la concave
    `y ↦ f(y,e) - c·y`, pas `y = x+1` : la quantité commandée par opportunité est
    illimitée. Le modèle « lot » relâche donc la contrainte de capacité qui fonde
    tout le make-to-stock queue, et son gain moyen **domine** celui de l'Étape 1.1
    au lieu de le retrouver. Le test de non-régression `K=0` prescrit par la spec
    (Étape 1.2) est de ce fait mathématiquement impossible sous cette forme
    d'opérateur ; voir la section « Coût fixe » du README.
    """
    x = pb.states[:, None].astype(float)
    # G(y) = f(y,e) - c·y ; on cherche max_{y > x} G(y), soit un cummax suffixe strict.
    G = f - pb.c * x
    suffix_max = np.full_like(G, -np.inf)
    suffix_max[:-1] = np.maximum.accumulate(G[::-1], axis=0)[::-1][1:]
    return np.maximum(f, suffix_max + pb.c * x - pb.K)


def T0(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Décision de réapprovisionnement : aiguille vers `T0_unit` ou `T0_setup`.

    `K = 0` → `T0_unit`, le make-to-stock queue du papier (production à l'unité).
    `K > 0` → `T0_setup`, le modèle « lot ». Ce sont **deux modèles distincts**,
    dont les gains moyens ne sont pas comparables entre eux (cf. `T0_setup`).
    """
    return T0_unit(f, pb) if pb.K == 0.0 else T0_setup(f, pb)


def T2(f: np.ndarray, pb: Problem, A: np.ndarray | None = None) -> np.ndarray:
    """Transitions d'environnement, réelles et fictives, générique en `N` (éq. 2).

    `T2 f(x,e) = Σ_{j≠e} q_ej f(x,j) + (Σ_{i≠e} Λ_i + Σ_{i≠e} Σ_{j≠i} q_ij) f(x,e)`
    """
    return f @ (env_matrix(pb) if A is None else A)


def Tlam(f: np.ndarray, pb: Problem, lam) -> np.ndarray:
    """Arrivée de demande au taux `λ_e` fixé, un taux par environnement (§3.1).

        `Tlam f(x,e) = r_e(λ_e) + λ_e f(x-1,e) + (Λ_e - λ_e) f(x,e)`  si `x > x_min`
        `Tlam f(x,e) = Λ_e f(x,e)`                                     si `x = x_min`

    À la borne basse la demande est perdue : c'est le cas `x = 0` du papier
    (lost sales) et, quand `m_backlog > 0`, l'artefact de troncature du backlog.
    """
    Lam = np.asarray(pb.Lam, dtype=float)
    lam = np.asarray(lam, dtype=float)
    rev = np.array([pb.revenue(float(lam[e]), e) for e in range(pb.N)])

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
    Lam = np.asarray(pb.Lam, dtype=float)
    out = Lam * f
    prices = np.full(f.shape, np.nan)

    delta = f[1:] - f[:-1]  # Δf(x-1,e) pour x = x_min+1 .. x_max

    if price_menu is None:
        p = np.clip((1.0 + pb.a * delta) / (2.0 * pb.a), 0.0, pb.p_max)
        lam = Lam * (1.0 - pb.a * p)
        out[1:] += lam * p - lam * delta
        prices[1:] = p
    else:
        menu = np.asarray(price_menu, dtype=float)
        # (n_prix, x, e) : on évalue le menu complet puis on prend l'argmax.
        lam = Lam[None, None, :] * (1.0 - pb.a * menu[:, None, None])
        gains = lam * menu[:, None, None] - lam * delta[None, :, :]
        best = np.argmax(gains, axis=0)
        out[1:] += np.take_along_axis(gains, best[None], axis=0)[0]
        prices[1:] = menu[best]

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


def _bellman(v, pb, A, lam, price_menu, h_vec):
    """Une application de l'opérateur de Bellman complet."""
    if lam is None:
        demand_term, prices = T_dynamic(v, pb, price_menu)
    else:
        demand_term, prices = Tlam(v, pb, lam), None
    return -h_vec[:, None] + pb.mu * T0(v, pb) + T2(v, pb, A) + demand_term, prices


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

    v = np.zeros((npb.states.size, npb.N))
    g_n, prices, converged, it = 0.0, None, False, 0

    for it in range(1, npb.max_iter + 1):
        w, prices = _bellman(v, npb, A, lam_n, price_menu, h_vec)
        diff = w - v
        lo, hi = diff.min(), diff.max()
        # Estimation du gain par le milieu de l'intervalle : l'erreur est
        # majorée par la demi-amplitude (Puterman, value iteration relative).
        g_n = 0.5 * (lo + hi)
        v = w - w[0, 0]
        if hi - lo < npb.tol:
            converged = True
            break

    return MDPSolution(
        g=g_n * gamma,
        v=v,
        base_stock=None if npb.K > 0 else extract_base_stock(v, npb),
        sS=extract_sS_policy(v, npb) if npb.K > 0 else None,
        prices=prices,
        n_iter=it,
        converged=converged,
    )


def extract_sS_policy(v: np.ndarray, pb: Problem) -> tuple[tuple[int, int], ...]:
    """Extrait la politique `(s_e, S_e)` quand `K > 0` (Étape 1.2).

    Renvoie, pour chaque environnement, `(s_e, S_e)` où l'on commande ssi
    `x <= s_e`, en remontant jusqu'à `S_e`. Lève `ValueError` si la cible de
    remontée n'est pas constante sur la région de commande — c'est-à-dire si la
    politique optimale n'est **pas** de type `(s, S)`, ce que l'Étape 1.2 demande
    de vérifier numériquement plutôt que de supposer.
    """
    x = pb.states.astype(float)
    out = []
    for e in range(pb.N):
        G = v[:, e] - pb.c * x
        targets, orders = [], []
        for i in range(len(x)):
            if i == len(x) - 1:
                orders.append(False)
                continue
            j = int(np.argmax(G[i + 1:])) + i + 1
            gain = G[j] + pb.c * x[i] - pb.K
            order = gain > v[i, e]
            orders.append(order)
            if order:
                targets.append(int(pb.states[j]))
        idx = np.flatnonzero(orders)
        if idx.size == 0:
            out.append((int(pb.x_min) - 1, int(pb.x_min) - 1))  # on ne commande jamais
            continue
        if len(set(targets)) != 1:
            raise ValueError(
                f"La cible de remontée n'est pas constante en environnement {e} "
                f"(valeurs {sorted(set(targets))}) : la politique optimale n'est pas (s, S)."
            )
        s_e = int(pb.states[idx[-1]])
        S_e = targets[0]
        out.append((s_e, S_e))
    return tuple(out)
