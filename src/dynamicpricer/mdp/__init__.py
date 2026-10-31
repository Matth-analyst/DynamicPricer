"""Solveur MDP en gain moyen : opérateurs, value iteration, résolution en lot et DP.

* `operators` : opérateurs de Bellman (demande, production, mise en route).
* `policies` : extraction des seuils base-stock depuis une fonction de valeur.
* `value_iteration` : `solve_mdp`, point d'entrée pour une instance unique.
* `batch` : `solve_batch` (grille de prix fixés) et `average_profit` (évaluation exacte).
* `dynamic_pricing` : itération sur les politiques pour la stratégie DP.
"""

from .batch import average_profit, solve_batch, solve_batch_truncated_python, solve_batch_vi
from .operators import T2, T0_unit, T_dynamic, T_produce, T_switch, Tlam, env_matrix
from .policies import base_stock_levels, extract_base_stock, extract_setup_policy
from .value_iteration import MDPSolution, solve_mdp

__all__ = [
    "MDPSolution",
    "T0_unit",
    "T2",
    "T_dynamic",
    "T_produce",
    "T_switch",
    "Tlam",
    "average_profit",
    "base_stock_levels",
    "env_matrix",
    "extract_base_stock",
    "extract_setup_policy",
    "solve_batch",
    "solve_batch_truncated_python",
    "solve_batch_vi",
    "solve_mdp",
]
