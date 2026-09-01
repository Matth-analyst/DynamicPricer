"""Les cinq stratégies de tarification/réapprovisionnement (Étape 2 de la spec).

Classification du papier (Table 1), sur deux axes indépendants :

                       | prix unique | un prix par `e` | un prix par `(x,e)`
    base-stock unique  |      S      |       SB        |    non considérée
    base-stock par `e` |     SP      |      EDP        |         DP

Les deux axes appellent deux machineries différentes :

* base-stock **libre** (SP, EDP, DP) — la value iteration l'optimise elle-même ;
  il suffit de balayer les prix, ce que `mdp.solve_batch` fait en une passe.
* base-stock **contraint** identique (S, SB) — c'est une classe de politiques
  restreinte, qu'aucune value iteration ne produit. On énumère donc les seuils et
  on évalue chaque politique exactement par sa loi stationnaire (`mdp.average_profit`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mdp import average_profit, solve_batch, solve_mdp
from params import Problem

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


def _single_price_grid(pb: Problem) -> np.ndarray:
    """Prix unique répliqué sur les environnements : `(N, P)`."""
    return np.tile(pb.search_price_grid, (pb.N, 1))


def _price_vector_grid(pb: Problem) -> np.ndarray:
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


def _demand_from_prices(pb: Problem, prices: np.ndarray) -> np.ndarray:
    """`λ_e(p_e)` pour une grille de prix `(N, B)`, renvoyée en `(N, B)`."""
    Lam = np.asarray(pb.Lam, dtype=float)[:, None]
    return Lam * np.clip(1.0 - pb.a * prices, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Base-stock libre : SP, EDP, DP
# ---------------------------------------------------------------------------


def _sweep(pb: Problem, prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration sur toute la grille de prix : `(g (B,), base_stock (N, B))`.

    Coûteux (c'est le poste dominant de l'Étape 2), et partagé entre deux stratégies :
    SP/EDP y lisent directement leur optimum, S/SB n'en tirent que la borne de
    recherche du seuil commun. `solve_all` ne l'exécute donc qu'une fois par grille.
    """
    return solve_batch(pb, _demand_from_prices(pb, prices))


def _solve_free_base_stock(
    pb: Problem, prices: np.ndarray, name: str, sweep=None
) -> Solution:
    """Balaie une grille de prix, base-stock optimisé par la value iteration."""
    g, base_stock = _sweep(pb, prices) if sweep is None else sweep
    best = int(np.argmax(g))
    return Solution(
        strategy=name,
        g=float(g[best]),
        base_stock=tuple(int(s) for s in base_stock[:, best]),
        prices=prices[:, best].copy(),
    )


def solve_static_pricing(pb: Problem, sweep=None) -> Solution:
    """**SP** : prix unique, base-stock libre par environnement."""
    return _solve_free_base_stock(pb, _single_price_grid(pb), "sp", sweep)


def solve_edp(pb: Problem, sweep=None) -> Solution:
    """**EDP** : un prix par environnement, base-stock libre par environnement."""
    return _solve_free_base_stock(pb, _price_vector_grid(pb), "edp", sweep)


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
    _, base_stock = _sweep(pb, prices) if sweep is None else sweep
    return int(min(base_stock.max() + _BOUND_MARGIN, pb.x_max))


def _search_common_base_stock(
    pb: Problem, prices: np.ndarray, name: str, sweep=None
) -> Solution:
    """Maximise sur (grille de prix) × (seuil commun `s`), par évaluation exacte."""
    lam = _demand_from_prices(pb, prices).T  # (B, N) attendu par `average_profit`
    bound = _base_stock_search_bound(pb, prices, sweep)
    levels = np.arange(max(pb.x_min, 0), bound + 1)

    gains = np.stack([average_profit(pb, lam, np.full(pb.N, s)) for s in levels])
    flat = int(np.argmax(gains))
    i_s, i_p = np.unravel_index(flat, gains.shape)

    if levels[i_s] == bound:
        raise ValueError(
            f"Le seuil commun optimal atteint la borne de recherche ({bound}) pour la "
            f"stratégie {name.upper()} : augmenter `_BOUND_MARGIN` ou `m_stock`."
        )
    return Solution(
        strategy=name,
        g=float(gains[i_s, i_p]),
        base_stock=(int(levels[i_s]),) * pb.N,
        prices=prices[:, i_p].copy(),
    )


def solve_static(pb: Problem, sweep=None) -> Solution:
    """**S** : prix unique **et** base-stock unique.

    C'est la stratégie de référence au dénominateur de tous les profit gains (§5.1).
    """
    return _search_common_base_stock(pb, _single_price_grid(pb), "s", sweep)


def solve_static_basestock(pb: Problem, sweep=None) -> Solution:
    """**SB** : un prix par environnement, base-stock unique."""
    return _search_common_base_stock(pb, _price_vector_grid(pb), "sb", sweep)


SOLVERS = {
    "s": solve_static,
    "sb": solve_static_basestock,
    "sp": solve_static_pricing,
    "edp": solve_edp,
    "dp": solve_dp,
}


def solve_all(pb: Problem) -> dict[str, Solution]:
    """Résout les cinq stratégies pour une instance donnée.

    Les deux balayages de value iteration — grille de prix unique, puis produit
    cartésien — sont mutualisés entre les stratégies qui partagent la même grille,
    ce qui divise par deux le poste de calcul dominant.
    """
    single, vector = _single_price_grid(pb), _price_vector_grid(pb)
    sweep_single, sweep_vector = _sweep(pb, single), _sweep(pb, vector)
    return {
        "s": _search_common_base_stock(pb, single, "s", sweep_single),
        "sb": _search_common_base_stock(pb, vector, "sb", sweep_vector),
        "sp": _solve_free_base_stock(pb, single, "sp", sweep_single),
        "edp": _solve_free_base_stock(pb, vector, "edp", sweep_vector),
        "dp": solve_dp(pb),
    }
