"""Résolution d'une instance en gain moyen : `solve_mdp` (value iteration relative, ou itération
sur les politiques quand `K = 0`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..params import Problem
from .dynamic_pricing import MAX_POLICY_SWEEPS, solve_policy_iteration
from .operators import T2, T0_unit, T_dynamic, T_produce, T_switch, Tlam, env_matrix
from .policies import extract_base_stock, extract_setup_policy

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
        """Permet le dépaquetage `g, v, s = solve_mdp(...)`."""
        return iter((self.g, self.v, self.base_stock))


def _demand_term(v, pb, lam, price_menu):
    if lam is None:
        return T_dynamic(v, pb, price_menu)
    return Tlam(v, pb, lam), None


def _bellman_unit(v, pb, A, lam, price_menu, h_vec):
    """Opérateur de Bellman du modèle de base : `-h + µ·T0 + T2 + Tλ`, sur `(x, e)`."""
    demand, prices = _demand_term(v, pb, lam, price_menu)
    return -h_vec[:, None] + pb.mu * T0_unit(v, pb) + T2(v, pb, A) + demand, prices, None


def _bellman_setup(v, pb, A, lam, price_menu, h_vec):
    """Opérateur de Bellman du modèle à mise en route : `S(-h + µ·P + T2 + Tλ)`, sur `(x, e, σ)`.

    Renvoie aussi `d(x,e) = inner(x,e,1) - inner(x,e,0)`, le gain net de tourner
    **hors** coût de mise en route, dont se déduisent les deux seuils : on produit
    si `σ=1` et `d > 0`, on redémarre si `σ=0` et `d > K`.

    Avec `K = 0`, `T_switch` devient `max{inner(·,0), inner(·,1)}` : les deux tranches
    `σ` se confondent, `inner(·,1) = -h + µ(v(x+1)-c) + …` et `inner(·,0) = -h + µ·v(x) + …`,
    donc leur maximum est littéralement `-h + µ·T0_unit v + T2 v + Tλ v`. La réduction
    au modèle de base est donc exacte, pas approchée.
    """
    demand, prices = _demand_term(v, pb, lam, price_menu)
    inner = -h_vec[:, None, None] + pb.mu * T_produce(v, pb) + T2(v, pb, A) + demand
    return T_switch(inner, pb), prices, inner[..., 1] - inner[..., 0]


def solve_mdp(
    pb: Problem,
    lam=None,
    price_menu: np.ndarray | None = None,
    force_setup_model: bool = False,
    method: str = "auto",
) -> MDPSolution:
    """Résout le problème en gain moyen (Appendix E : 5 décimales).

    Itération sur les politiques quand `K = 0`, value iteration relative sinon.

    Parameters
    ----------
    pb :
        Le problème, **en unités réelles** — la normalisation est faite ici.
    lam :
        Vecteur des taux de demande imposés, un par environnement (stratégies
        S, SB, SP, EDP). `None` → stratégie DP : le taux est optimisé dans chaque
        état. Les taux sont exprimés dans les unités de `pb` et normalisés ici.
    price_menu :
        Menu de prix fini pour la stratégie DP. `None` → prix continu.
    force_setup_model :
        Force l'espace d'état augmenté `(x, e, σ)` même quand `K = 0`. Sert au test
        de non-régression du modèle à mise en route : les deux modèles doivent alors coïncider
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
        g, v, base_stock, prices, sweeps = solve_policy_iteration(
            npb, gamma, price_menu, MAX_POLICY_SWEEPS, lam_n
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
