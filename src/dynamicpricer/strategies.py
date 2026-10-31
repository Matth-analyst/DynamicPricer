"""Les cinq stratégies de tarification/réapprovisionnement.

Classification du papier (Table 1), sur deux axes indépendants :

                       | prix unique | un prix par `e` | un prix par `(x,e)`
    base-stock unique  |      S      |       SB        |    non considérée
    base-stock par `e` |     SP      |      EDP        |         DP

Les deux axes appellent deux machineries différentes :

* base-stock **libre** (SP, EDP, DP) — l'itération sur les politiques l'optimise elle-même ;
  il suffit de balayer les prix, ce que `mdp.solve_batch` fait en une passe.
* base-stock **contraint** identique (S, SB) — c'est une classe de politiques
  restreinte, qu'aucune résolution du MDP ne produit. On énumère donc les seuils et
  on évalue chaque politique exactement par sa loi stationnaire (`mdp.average_profit`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import native
from .mdp import average_profit, solve_batch, solve_mdp
from .params import Problem

#: Les cinq stratégies, dans l'ordre croissant de flexibilité.
STRATEGIES = ("s", "sb", "sp", "edp", "dp")

#: Garde-fou : au-delà, le balayage `N`-dimensionnel des prix devient impraticable.
MAX_PRICE_COMBINATIONS = 200_000

#: Marge ajoutée à la borne de recherche du base-stock contraint, en unités de stock.
_BOUND_MARGIN = 5


@dataclass(frozen=True)
class Solution:
    """Résultat d'une stratégie."""

    strategy: str
    g: float
    """Gain moyen de long terme, en unités de temps réelles."""
    base_stock: tuple[int, ...]
    """Seuils `s_e` par environnement (tous égaux pour S et SB)."""
    prices: np.ndarray
    """Prix optimaux : `(N,)` pour S, SB, SP, EDP ; `(x, e)` pour DP."""

    @property
    def is_common_base_stock(self) -> bool:
        return len(set(self.base_stock)) == 1


# ---------------------------------------------------------------------------
# Grilles de prix
# ---------------------------------------------------------------------------


def single_price_grid(pb: Problem) -> np.ndarray:
    """Prix unique répliqué sur les environnements : `(N, P)`."""
    return np.tile(pb.search_price_grid, (pb.N, 1))


def price_vector_grid(pb: Problem) -> np.ndarray:
    """Produit cartésien des prix par environnement : `(N, P^N)`."""
    grid = pb.search_price_grid
    size = grid.size**pb.N
    if size > MAX_PRICE_COMBINATIONS:
        raise ValueError(
            f"{size} combinaisons de prix pour N={pb.N} au pas {pb.price_step} : "
            "au-delà de MAX_PRICE_COMBINATIONS. Augmenter `price_step` (le papier "
            "utilise 0.01, mais seulement pour N=2)."
        )
    mesh = np.meshgrid(*([grid] * pb.N), indexing="ij")
    return np.stack([m.ravel() for m in mesh])


def demand_from_prices(pb: Problem, prices: np.ndarray) -> np.ndarray:
    """`λ_e(p_e)` pour une grille de prix `(N, B)`, renvoyée en `(N, B)`."""
    Lam = np.asarray(pb.Lam, dtype=float)[:, None]
    return Lam * np.clip(1.0 - pb.a * prices, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Base-stock libre : SP, EDP, DP
# ---------------------------------------------------------------------------


def price_sweep(pb: Problem, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Itération sur les politiques sur toute la grille de prix : `(g (B,), base_stock (N, B))`.

    Coûteux (c'est l'un des postes dominants du solveur), et partagé entre deux stratégies :
    SP/EDP y lisent directement leur optimum, S/SB n'en tirent que la borne de
    recherche du seuil commun. `solve_all` ne l'exécute donc qu'une fois par grille.
    """
    return solve_batch(pb, demand_from_prices(pb, prices))


def _solve_free_base_stock(
    pb: Problem, prices: np.ndarray, name: str, sweep=None
) -> Solution:
    """Balaie une grille de prix, base-stock optimisé par itération sur les politiques."""
    g, base_stock = price_sweep(pb, prices) if sweep is None else sweep
    best = int(np.argmax(g))
    return Solution(
        strategy=name,
        g=float(g[best]),
        base_stock=tuple(int(s) for s in base_stock[:, best]),
        prices=prices[:, best].copy(),
    )


def solve_static_pricing(pb: Problem, sweep=None) -> Solution:
    """**SP** : prix unique, base-stock libre par environnement."""
    return _solve_free_base_stock(pb, single_price_grid(pb), "sp", sweep)


def solve_edp(pb: Problem, sweep=None) -> Solution:
    """**EDP** : un prix par environnement, base-stock libre par environnement."""
    return _solve_free_base_stock(pb, price_vector_grid(pb), "edp", sweep)


def solve_dp(pb: Problem) -> Solution:
    """**DP** : un prix par état `(x, e)`, base-stock libre par environnement."""
    sol = solve_mdp(pb)
    return Solution(
        strategy="dp",
        g=sol.g,
        base_stock=tuple(sol.base_stock),
        prices=sol.prices,
    )


# ---------------------------------------------------------------------------
# Base-stock contraint identique : S, SB
# ---------------------------------------------------------------------------


def _base_stock_search_bound(pb: Problem, prices: np.ndarray, sweep=None) -> int:
    """Borne haute de recherche du seuil commun.

    Le seuil commun optimal ne peut raisonnablement pas dépasser le plus grand
    base-stock **non contraint** rencontré sur la grille de prix : le contraindre à
    être identique partout ne peut pas justifier de stocker plus que ne le ferait la
    politique libre la plus généreuse. Une marge est ajoutée, et l'argmax retenu est
    vérifié intérieur (cf. `_search_common_base_stock`) — si la borne mordait, le
    résultat serait faux silencieusement.
    """
    _, base_stock = price_sweep(pb, prices) if sweep is None else sweep
    return int(min(base_stock.max() + _BOUND_MARGIN, pb.x_max))


def _search_common_base_stock(
    pb: Problem, prices: np.ndarray, name: str, sweep=None
) -> Solution:
    """Maximise sur (grille de prix) × (seuil commun `s`), par évaluation exacte."""
    lam = demand_from_prices(pb, prices).T  # (B, N) attendu par `average_profit`
    bound = _base_stock_search_bound(pb, prices, sweep)
    levels = np.arange(max(pb.x_min, 0), bound + 1)

    # C'est le poste le plus lourd du solveur complet : un niveau candidat × chaque
    # vecteur de prix, soit ~2,6 millions d'évaluations à `N=4` au pas 0.05. Le moteur
    # natif le prend en charge quand il est disponible.
    if native.HAS_COMMON_LEVEL:
        best_g, best_level = native.search_common_level(pb, lam, int(bound))
        i_p = int(np.argmax(best_g))
        best = float(best_g[i_p])
        chosen = int(best_level[i_p])
    else:
        gains = np.stack([average_profit(pb, lam, np.full(pb.N, s)) for s in levels])
        flat = int(np.argmax(gains))
        i_s, i_p = np.unravel_index(flat, gains.shape)
        best = float(gains[i_s, i_p])
        chosen = int(levels[i_s])

    if chosen == bound:
        raise ValueError(
            f"Le seuil commun optimal atteint la borne de recherche ({bound}) pour la "
            f"stratégie {name.upper()} : augmenter `_BOUND_MARGIN` ou `m_stock`."
        )
    return Solution(
        strategy=name,
        g=best,
        base_stock=(chosen,) * pb.N,
        prices=prices[:, i_p].copy(),
    )


def solve_static(pb: Problem, sweep=None) -> Solution:
    """**S** : prix unique **et** base-stock unique.

    C'est la stratégie de référence au dénominateur de tous les profit gains (§5.1).
    """
    return _search_common_base_stock(pb, single_price_grid(pb), "s", sweep)


def solve_static_basestock(pb: Problem, sweep=None) -> Solution:
    """**SB** : un prix par environnement, base-stock unique."""
    return _search_common_base_stock(pb, price_vector_grid(pb), "sb", sweep)


SOLVERS = {
    "s": solve_static,
    "sb": solve_static_basestock,
    "sp": solve_static_pricing,
    "edp": solve_edp,
    "dp": solve_dp,
}


def solve_all(pb: Problem) -> dict[str, Solution]:
    """Résout les cinq stratégies pour une instance donnée.

    Les deux balayages — grille de prix unique, puis produit
    cartésien — sont mutualisés entre les stratégies qui partagent la même grille,
    ce qui divise par deux le poste de calcul dominant.
    """
    single, vector = single_price_grid(pb), price_vector_grid(pb)
    sweep_single, sweep_vector = price_sweep(pb, single), price_sweep(pb, vector)
    return {
        "s": _search_common_base_stock(pb, single, "s", sweep_single),
        "sb": _search_common_base_stock(pb, vector, "sb", sweep_vector),
        "sp": _solve_free_base_stock(pb, single, "sp", sweep_single),
        "edp": _solve_free_base_stock(pb, vector, "edp", sweep_vector),
        "dp": solve_dp(pb),
    }
