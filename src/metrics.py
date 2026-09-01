"""Métriques de comparaison entre stratégies (§5.1 du papier).

Statut : `profit_gain` est implémenté (formule triviale et utilisée partout) ;
le reste attend l'Étape 3.
"""

from __future__ import annotations

import numpy as np


def profit_gain(g_phi: float, g_ref: float) -> float:
    """Profit gain relatif `PG^{φ,φ'} = (g^φ - g^{φ'}) / g^{φ'}`.

    Invariant par la normalisation d'uniformisation, puisque c'est un ratio.
    """
    if g_ref == 0:
        raise ZeroDivisionError("Le gain de référence est nul.")
    return (g_phi - g_ref) / g_ref


def check_ordering(gains: dict[str, float], tol: float = 1e-9) -> bool:
    """Vérifie `g_s <= g_sb <= g_edp <= g_dp` et `g_s <= g_sp <= g_edp <= g_dp` (§5.1).

    Test de cohérence interne : ces inégalités découlent de l'emboîtement des
    espaces de politiques et doivent tenir pour toute instance.
    """
    chains = (("s", "sb", "edp", "dp"), ("s", "sp", "edp", "dp"))
    return all(
        gains[a] <= gains[b] + tol
        for chain in chains
        for a, b in zip(chain, chain[1:])
    )


def recurrent_states(base_stock, x_min: int = 0) -> np.ndarray:
    """Niveaux de stock récurrents : `x_min <= x <= max_e s_e` (§5.1).

    Le papier restreint toutes ses statistiques à cet ensemble. C'est essentiel pour
    les prix DP : au-delà du plus grand base-stock, les états ne sont jamais visités
    et les prix qu'on y lirait sont dénués de sens — c'est ce qui explique qu'un
    minimum pris sur toute la grille tronquée ne coïnciderait pas avec la Table 5.
    """
    return np.arange(x_min, max(base_stock) + 1)


def dp_price_range(prices: np.ndarray, base_stock, x_min: int = 0):
    """`(p_e, p̄_e)` par environnement sur les états récurrents (légende Table 5).

    Le prix en `x = x_min` vaut `NaN` (demande perdue, prix sans effet) et est donc
    exclu ; par le Théorème 2 le maximum est atteint en `x = x_min + 1`.
    """
    rec = recurrent_states(base_stock, x_min)[1:] - x_min
    window = prices[rec]
    return list(zip(window.min(axis=0), window.max(axis=0)))
