"""Solveur MDP générique : opérateurs de base et value iteration (Étape 1 de la spec).

Notation des opérateurs (cf. note de notation §Étape 1.1 de la spec) :
`T0` = décision de réapprovisionnement (`T1` dans le papier 2009, éq. 1),
`T2` = transitions d'environnement réelles + fictives (éq. 2),
`Tlam` = arrivée de demande à taux `λ` fixé.

Statut : squelette (Étape 0). Implémentation à l'Étape 1.
"""

from __future__ import annotations

import numpy as np

from params import Problem


def T0(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Décision de réapprovisionnement : `T0 f(x,e) = max{f(x,e), f(x+1,e) - c}`.

    Avec `K > 0` (Étape 1.2), devient
    `T0 f(x,e) = max{f(x,e), max_{y>x}[f(y,e) - c(y-x) - K]}`.
    """
    raise NotImplementedError("Étape 1.1")


def T2(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Transitions d'environnement, réelles et fictives, généralisé à `N` états.

    `T2 f(x,e) = Σ_{j≠e} q_ej f(x,j) + (Σ_{i≠e} Λ_i + Σ_{i≠e} Σ_{j≠i} q_ij) f(x,e)`
    """
    raise NotImplementedError("Étape 1.1")


def Tlam(f: np.ndarray, pb: Problem, lam: np.ndarray) -> np.ndarray:
    """Arrivée de demande au taux `λ_e` fixé, un taux par environnement.

    `Tlam f(x,e) = r_e(λ) + λ f(x-1,e) + (Λ_e - λ) f(x,e)` si `x > x_min`,
    `Λ_e f(x,e)` sinon (perte de vente à la borne basse).
    """
    raise NotImplementedError("Étape 1.1")


def T_dynamic(f: np.ndarray, pb: Problem) -> np.ndarray:
    """Opérateur de tarification dynamique : `T f(x,e) = max_λ Tlam f(x,e)`.

    Pour la demande linéaire, le maximiseur admet la forme fermée de l'Appendix E (éq. 11) :
    `p(x,e) = clip((1 + a·Δv(x-1,e)) / (2a), 0, 1/a)`.
    """
    raise NotImplementedError("Étape 2")


def solve_mdp(pb: Problem, lam: np.ndarray | None = None):
    """Value iteration average-cost.

    `lam=None` → stratégie DP (le taux de demande est optimisé dans chaque état).
    Sinon `lam` est le vecteur des taux fixés par environnement (stratégies S/SB/SP/EDP).

    Returns
    -------
    (g, v, base_stock_levels) : gain moyen, fonction de valeur relative, seuils `s_e`.
    """
    raise NotImplementedError("Étape 1.1")
