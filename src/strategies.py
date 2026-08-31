"""Les cinq stratégies de tarification/réapprovisionnement (Étape 2 de la spec).

Classification du papier (Table 1), sur deux axes indépendants :

                       | prix unique | un prix par `e` | un prix par `(x,e)`
    base-stock unique  |      S      |       SB        |    non considérée
    base-stock par `e` |     SP      |      EDP        |         DP

Statut : squelette (Étape 0). Implémentation à l'Étape 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from params import Problem

#: Les cinq stratégies, dans l'ordre croissant de flexibilité.
STRATEGIES = ("s", "sb", "sp", "edp", "dp")


@dataclass(frozen=True)
class Solution:
    """Résultat d'une stratégie : gain moyen, seuils de production et prix optimaux."""

    strategy: str
    g: float
    """Gain moyen de long terme (average profit), en unités de temps réelles."""
    base_stock: tuple[int, ...]
    """Seuils `s_e` par environnement (tous égaux pour S et SB)."""
    prices: np.ndarray
    """Prix optimaux : scalaire (S, SP), un par environnement (SB, EDP), ou `(x, e)` (DP)."""
    v: np.ndarray
    """Fonction de valeur relative."""


def solve_static(pb: Problem) -> Solution:
    """**S** : prix unique et base-stock unique. Référence au dénominateur des profit gains."""
    raise NotImplementedError("Étape 2")


def solve_static_basestock(pb: Problem) -> Solution:
    """**SB** : un prix par environnement, base-stock unique."""
    raise NotImplementedError("Étape 2")


def solve_static_pricing(pb: Problem) -> Solution:
    """**SP** : prix unique, base-stock libre par environnement."""
    raise NotImplementedError("Étape 2")


def solve_edp(pb: Problem) -> Solution:
    """**EDP** : un prix par environnement, base-stock libre par environnement."""
    raise NotImplementedError("Étape 2")


def solve_dp(pb: Problem) -> Solution:
    """**DP** : un prix par état `(x, e)`, base-stock libre par environnement."""
    raise NotImplementedError("Étape 2")


SOLVERS = {
    "s": solve_static,
    "sb": solve_static_basestock,
    "sp": solve_static_pricing,
    "edp": solve_edp,
    "dp": solve_dp,
}


def solve_all(pb: Problem) -> dict[str, Solution]:
    """Résout les cinq stratégies pour une instance donnée."""
    return {name: SOLVERS[name](pb) for name in STRATEGIES}
