"""Validation du solveur.

Deux volets, de statut épistémique très différent :

* **Validation externe.** Le cœur du modèle (`N=2`, `K=0`, lost sales) est
  confronté aux Tables 3 à 5 et à la Figure 2 du papier. C'est la seule partie du
  projet dont la correction est attestée par une source publiée.
* **Cohérence interne.** Les extensions (`N>2`, `K>0`, backlog) n'ont aucune
  référence à reproduire. Elles sont vérifiées par des cas limites et des propriétés
  structurelles, ce qui est plus faible et doit être présenté comme tel.

`python -m dynamicpricer.validation` calcule tout et écrit `report/validation_results.json`, que le
notebook `01_validation.ipynb` relit sans tout recalculer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import reference_gayon2009 as ref
from .mdp import solve_mdp
from .metrics import check_ordering, dp_price_range, profit_gain
from .params import Problem
from .strategies import (
    price_sweep,
    price_vector_grid,
    single_price_grid,
    solve_all,
    solve_dp,
    solve_edp,
    solve_static_pricing,
)

RESULTS_PATH = Path(__file__).resolve().parents[2] / "report" / "validation_results.json"

#: Tolérance sur les profit gains, en points de pourcentage.
#: Le papier ne publie qu'une décimale, donc son propre arrondi vaut déjà ±0.05 point.
PG_TOLERANCE_PP = 0.3

#: Tolérance sur les prix, en unités de prix (la grille du papier a un pas de 0.01).
PRICE_TOLERANCE = 0.015


def reference_problem(eps: float, **kw) -> Problem:
    """Instance de référence du papier (§5.3) pour une variabilité `ε` donnée."""
    p = dict(ref.BASE_PARAMS)
    p.update(kw)
    return Problem.two_env(eps=eps, m_stock=p.pop("m_stock", 80), **p)


# ---------------------------------------------------------------------------
# Validation externe contre le papier
# ---------------------------------------------------------------------------


def solve_reference_instances(eps_values=ref.EPSILONS) -> dict:
    """Résout les cinq stratégies pour chaque `ε` de la Table 3."""
    return {eps: solve_all(reference_problem(eps)) for eps in eps_values}


def compare_table_3(solved: dict) -> list[dict]:
    """Profit gains `PG^{φ,s}` calculés vs publiés (Table 3), en points de %."""
    rows = []
    for eps, sols in solved.items():
        g_ref = sols["s"].g
        for name in ("sb", "sp", "edp", "dp"):
            got = 100 * profit_gain(sols[name].g, g_ref)
            want = 100 * ref.TABLE_3[eps][name]
            rows.append(
                dict(eps=eps, quantity=f"PG_{name},s", computed=round(got, 2),
                     published=round(want, 2), gap_pp=round(got - want, 2))
            )
    return rows


def compare_table_4(solved: dict) -> list[dict]:
    """Base-stocks optimaux calculés vs publiés (Table 4)."""
    rows = []
    for eps, sols in solved.items():
        for name in ("s", "sb", "sp", "edp", "dp"):
            got = sols[name].base_stock
            want = ref.TABLE_4[eps][name]
            got_cmp = got[0] if isinstance(want, int) else tuple(got)
            rows.append(
                dict(eps=eps, quantity=f"s_{name}", computed=got_cmp,
                     published=want, exact=got_cmp == want)
            )
    return rows


def compare_table_5(solved: dict, eps_values=ref.EPSILONS) -> list[dict]:
    """Prix optimaux calculés vs publiés (Table 5).

    Pour DP, on rapporte l'intervalle `(p_e, p̄_e)` sur les **états récurrents**, comme
    la légende de la Table 5 l'indique.
    """
    rows = []
    for eps, sols in solved.items():
        for name in ("s", "sb", "sp", "edp"):
            want = ref.TABLE_5[eps][name]
            got = sols[name].prices
            got_cmp = round(float(got[0]), 3) if isinstance(want, float) else tuple(
                round(float(p), 3) for p in got
            )
            gap = max(abs(np.atleast_1d(got_cmp) - np.atleast_1d(want)))
            rows.append(dict(eps=eps, quantity=f"p_{name}", computed=got_cmp,
                             published=want, gap=round(float(gap), 3)))

        dp = sols["dp"]
        ranges = dp_price_range(dp.prices, dp.base_stock)
        for e, label in enumerate(("L", "H")):
            want = ref.TABLE_5[eps]["dp"][label]
            got_cmp = tuple(round(float(p), 3) for p in ranges[e])
            gap = max(abs(np.array(got_cmp) - np.array(want)))
            rows.append(dict(eps=eps, quantity=f"p_dp_{label}", computed=got_cmp,
                             published=want, gap=round(float(gap), 3)))
    return rows


def figure_2_data(mu_values=None, eps: float = 0.8) -> list[dict]:
    """Profit gains en fonction du taux de service `µ`, à `ε = 0.8` (Figure 2).

    Ne calcule que SP, EDP et DP : la figure ne fait intervenir ni S ni SB, dont la
    recherche de seuil commun est le poste le plus coûteux. Le balayage de prix
    `N`-dimensionnel est partagé entre SP et EDP.
    """
    if mu_values is None:
        # Grille de l'axe de la Figure 2 du papier : 0.01, 0.06, ..., 0.96.
        mu_values = np.round(np.arange(0.01, 0.97, 0.05), 3)
    rows = []
    for mu in mu_values:
        pb = reference_problem(eps, mu=float(mu))
        single, vector = single_price_grid(pb), price_vector_grid(pb)
        g = {
            "sp": solve_static_pricing(pb, price_sweep(pb, single)).g,
            "edp": solve_edp(pb, price_sweep(pb, vector)).g,
            "dp": solve_dp(pb).g,
        }
        rows.append(
            dict(mu=float(mu),
                 PG_dp_edp=100 * profit_gain(g["dp"], g["edp"]),
                 PG_dp_sp=100 * profit_gain(g["dp"], g["sp"]),
                 PG_edp_sp=100 * profit_gain(g["edp"], g["sp"]))
        )
    return rows


def structural_properties(solved: dict) -> list[dict]:
    """Propriétés qualitatives attendues (§4 et §5.3.2), sur chaque instance."""
    rows = []
    for eps, sols in solved.items():
        pb = reference_problem(eps)
        dp = sols["dp"]
        v = solve_mdp(pb).v
        rows.append(dict(
            eps=eps,
            # §5.1 : emboîtement des classes de politiques.
            profit_ordering=check_ordering({k: s.g for k, s in sols.items()}),
            # Lemme 1 : les opérateurs préservent la concavité.
            value_function_concave=bool(np.all(np.diff(v, n=2, axis=0) <= 1e-9)),
            # Théorème 2 : prix DP non croissants en stock.
            dp_prices_non_increasing=bool(
                np.all(np.diff(dp.prices[1:], axis=0) <= 1e-9)
            ),
            # Théorème 3 : base-stocks ordonnés par environnement.
            base_stock_ordered=all(
                sols[n].base_stock[0] <= sols[n].base_stock[1]
                for n in ("sp", "edp", "dp")
            ),
            # §5.3.2 : la flexibilité de DP autorise davantage de stock. Restreint
            # à SP et EDP — voir `note` ci-dessous.
            dp_stores_most=all(
                a <= b
                for n in ("sp", "edp")
                for a, b in zip(sols[n].base_stock, dp.base_stock)
            ),
            note=_common_base_stock_note(sols),
        ))
    return rows


def _common_base_stock_note(sols) -> str:
    """Compare le seuil commun de S/SB au base-stock DP de l'environnement bas.

    Le papier énonce `s^φ_e <= s^dp_e` pour `φ = s, sb, sp, edp` (§5.3.2), mais sa
    **propre Table 4** contredit cette affirmation à `ε = 0.8` : elle y publie
    `s^sb = 4` et `s^dp_L = 3`. Ce n'est pas une erreur du solveur, qui reproduit ces
    deux valeurs exactement — c'est une inexactitude du texte du papier.

    L'explication est simple : S et SB n'ont qu'un seuil, compromis entre les
    environnements, qui n'a pas de raison de rester sous le seuil que DP retient dans
    l'environnement le plus faible. La propriété n'est bien posée que pour les
    stratégies dont le base-stock dépend de l'environnement.
    """
    violations = [
        f"s^{n}={sols[n].base_stock[0]} > s^dp_L={sols['dp'].base_stock[0]}"
        for n in ("s", "sb")
        if sols[n].base_stock[0] > sols["dp"].base_stock[0]
    ]
    return "; ".join(violations) if violations else "s^s, s^sb <= s^dp_L"


# ---------------------------------------------------------------------------
# Cohérence interne des extensions
# ---------------------------------------------------------------------------


def extension_checks() -> list[dict]:
    """Cas limites des extensions, sans référence publiée à reproduire.

    Chaque entrée porte le nom du cas, la grandeur comparée et l'écart obtenu.
    """
    rows = []

    # `N > 2` : trois environnements dont deux identiques doivent reproduire N = 2.
    pb2 = Problem.two_env(eps=0.6, m_stock=60, price_step=0.05)
    two = solve_mdp(pb2)
    L, H = pb2.Lam
    pb3 = Problem(
        Lam=(L, H, H),  # l'environnement H est dédoublé
        Q=((0.0, 0.005, 0.005), (0.01, 0.0, 0.0), (0.01, 0.0, 0.0)),
        m_stock=60, price_step=0.05,
    )
    three = solve_mdp(pb3)
    rows.append(dict(
        case="N>2 : H dédoublé reproduit N=2",
        computed=round(three.g, 9), reference=round(two.g, 9),
        gap=round(abs(three.g - two.g), 9),
        detail=f"s(N=3)={three.base_stock} vs s(N=2)={two.base_stock}",
    ))

    # `K > 0` : la non-régression `K = 0` doit être exacte.
    pb = Problem.two_env(eps=0.6, m_stock=60)
    plain = solve_mdp(pb, method="vi")
    augmented = solve_mdp(pb, force_setup_model=True)
    rows.append(dict(
        case="K=0 : le modèle à mise en route redonne le modèle de base",
        computed=round(augmented.g, 12), reference=round(plain.g, 12),
        gap=round(abs(augmented.g - plain.g), 12),
        detail=f"bande={augmented.sS} vs base-stock={plain.base_stock}",
    ))

    # `b -> ∞` : le backlog doit converger vers le comportement lost-sales.
    lost = solve_mdp(Problem.two_env(eps=0.6, m_stock=60))
    backlogged = solve_mdp(Problem.two_env(eps=0.6, m_stock=60, m_backlog=5, b=1e6))
    rows.append(dict(
        case="b→∞ : le backlog converge vers lost sales",
        computed=round(backlogged.g, 9), reference=round(lost.g, 9),
        gap=round(abs(backlogged.g - lost.g), 9),
        detail=f"s={backlogged.base_stock} vs {lost.base_stock}",
    ))
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(with_figure_2: bool = True) -> dict:
    """Calcule l'ensemble de la validation et renvoie un dictionnaire sérialisable."""
    solved = solve_reference_instances()
    out = {
        "table_3": compare_table_3(solved),
        "table_4": compare_table_4(solved),
        "table_5": compare_table_5(solved),
        "structural": structural_properties(solved),
        "extensions": extension_checks(),
    }
    if with_figure_2:
        out["figure_2"] = figure_2_data()
    return out


def main() -> None:
    results = run_all()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    n_exact = sum(r["exact"] for r in results["table_4"])
    worst_pg = max(abs(r["gap_pp"]) for r in results["table_3"])
    print(f"Table 4 : {n_exact}/{len(results['table_4'])} base-stocks exacts")
    print(f"Table 3 : écart maximum {worst_pg:.2f} point de %")
    print(f"Écrit dans {RESULTS_PATH}")


if __name__ == "__main__":
    main()
