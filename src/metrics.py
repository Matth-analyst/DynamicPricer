"""Métriques de comparaison entre stratégies (§5.1 du papier).

Statut : `profit_gain` est implémenté (formule triviale et utilisée partout) ;
le reste attend l'Étape 3.
"""

from __future__ import annotations


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


def recurrent_states(base_stock: tuple[int, ...]) -> range:
    """Ensemble des niveaux de stock récurrents : `0 <= x <= max_e s_e` (§5.1)."""
    raise NotImplementedError("Étape 3")
