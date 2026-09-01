"""Analyse de sensibilité (Étape 4 de la spec).

Quatre volets :

* **Figure 1** — `PG^{dp,s}` en fonction de `µ` et `h`, système mono-environnement.
  Reproduit une figure publiée, et sert de garde-fou : le papier établit que le gain
  maximal atteignable y est de 3.81 % (§5.2.1).
* **Balayages** de `µ`, `h`, `ε` et `q` sur le système à deux environnements, les cinq
  stratégies à chaque point.
* **Table 2** — impact du nombre de prix disponibles pour DP (2, 3, continu).
* Les **enseignements métier** qui s'en déduisent sont rédigés dans
  `notebooks/02_sensitivity.ipynb`.

`python -m sensitivity` calcule tout et écrit `report/sensitivity_results.json`.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import reference_gayon2009 as ref
from metrics import profit_gain
from mdp import solve_mdp
from params import Problem
from strategies import solve_all, solve_dp, solve_static
from validation import _problem

RESULTS_PATH = Path(__file__).resolve().parents[1] / "report" / "sensitivity_results.json"

#: Valeur de `ε` autour de laquelle les balayages de `µ`, `h` et `q` sont centrés.
#: Le papier retient 0.8 (§5.3.1) : c'est là que les stratégies se séparent le plus.
BASE_EPS = 0.8


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Figure 1 — système mono-environnement
# ---------------------------------------------------------------------------


def single_environment_problem(mu: float, h: float, m_stock: int = 120) -> Problem:
    """Système à un seul environnement (`Λ = 1`, `a = 1`, `c = 0`), cf. §5.2."""
    return Problem(Lam=(1.0,), Q=((0.0,),), mu=mu, h=h, c=0.0, a=1.0, m_stock=m_stock)


def figure_1_data(mu_values=None, h_values=(0.001, 0.01, 0.1)) -> list[dict]:
    """`PG^{dp,s}` en fonction de `µ`, pour plusieurs coûts de stockage (Figure 1).

    En mono-environnement, S, SB, SP et EDP se confondent : seule DP se distingue,
    et le profit gain mesure donc exactement l'apport de la dépendance au stock.
    """
    if mu_values is None:
        mu_values = np.round(np.arange(0.05, 1.001, 0.05), 3)
    rows = []
    for h in h_values:
        for mu in mu_values:
            pb = single_environment_problem(float(mu), float(h))
            rows.append(dict(
                h=float(h), mu=float(mu),
                pg_dp_s=100 * profit_gain(solve_dp(pb).g, solve_static(pb).g),
            ))
        _log(f"  Figure 1 : h={h} terminé")
    return rows


# ---------------------------------------------------------------------------
# Balayages à deux environnements
# ---------------------------------------------------------------------------


def _row(pb: Problem, param: str, value: float) -> dict:
    """Résout les cinq stratégies et résume l'instance en une ligne."""
    sols = solve_all(pb)
    g = {k: v.g for k, v in sols.items()}
    row = dict(param=param, value=float(value), g_s=g["s"])
    row.update({f"pg_{k}_s": 100 * profit_gain(g[k], g["s"]) for k in ("sb", "sp", "edp", "dp")})
    # L'écart DP/EDP est la grandeur décisive : il mesure ce que coûte de renoncer
    # à la tarification dynamique complète au profit d'une simple grille par saison.
    row["pg_dp_edp"] = 100 * profit_gain(g["dp"], g["edp"])
    row["base_stock_dp"] = list(sols["dp"].base_stock)
    row["base_stock_edp"] = list(sols["edp"].base_stock)
    row["prices_edp"] = [round(float(p), 3) for p in sols["edp"].prices]
    return row


def sweep(param: str, values, eps: float = BASE_EPS) -> list[dict]:
    """Balaie un paramètre, les quatre autres restant à leur valeur de référence.

    `param` vaut `"mu"`, `"h"`, `"eps"` ou `"q"`.
    """
    rows = []
    for v in values:
        pb = _problem(float(v)) if param == "eps" else _problem(eps, **{param: float(v)})
        rows.append(_row(pb, param, v))
        _log(f"  balayage {param}={v} terminé")
    return rows


# ---------------------------------------------------------------------------
# Table 2 — impact du menu de prix
# ---------------------------------------------------------------------------


def menu_study(mu_values=(0.1, 0.3, 0.5, 0.7, 0.9), h: float = 0.01) -> list[dict]:
    """`PG^{dp,s}` selon que DP dispose de 2 prix, 3 prix ou d'un continuum (Table 2).

    Le cas à trois prix reprend **exactement** la contrainte de l'Appendix E :
    `p3 = (p1 + p2) / 2`, imposée par le papier pour éviter un balayage à 100³. La
    valeur obtenue est donc une **borne inférieure** de l'optimum à trois prix, et
    c'est cette borne — non l'optimum — qui figure dans la Table 2 publiée.
    """
    rows = []
    for mu in mu_values:
        pb = single_environment_problem(float(mu), h)
        g_s = solve_static(pb).g
        g_cont = solve_dp(pb).g

        best2 = best3 = -np.inf
        for p1, p2 in combinations(pb.search_price_grid, 2):
            best2 = max(best2, solve_mdp(pb, price_menu=np.array([p1, p2])).g)
            best3 = max(best3, solve_mdp(
                pb, price_menu=np.array([p1, p2, 0.5 * (p1 + p2)])
            ).g)

        rows.append(dict(
            mu=float(mu),
            pg_2=100 * profit_gain(best2, g_s),
            pg_3=100 * profit_gain(best3, g_s),
            pg_continuous=100 * profit_gain(g_cont, g_s),
        ))
        _log(f"  menu de prix : µ={mu} terminé")
    return rows


def menu_share_of_continuous(rows: list[dict]) -> dict:
    """Part du bénéfice du continuum captée par un menu de 2 ou 3 prix (§5.2.2).

    Le papier annonce 78.5 % pour deux prix et 92.5 % pour trois, en moyenne sur
    `µ ∈ {0.05, ..., 1.00}`.
    """
    return {
        n: float(np.mean([r[f"pg_{n}"] / r["pg_continuous"] for r in rows]))
        for n in (2, 3)
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all() -> dict:
    _log("Figure 1 (mono-environnement)")
    fig1 = figure_1_data()

    _log("Balayages à deux environnements")
    sweeps = {
        "mu": sweep("mu", [0.01, 0.06, 0.11, 0.21, 0.31, 0.41, 0.51, 0.66, 0.81, 0.96]),
        "h": sweep("h", [0.001, 0.005, 0.01, 0.05, 0.1]),
        "eps": sweep("eps", [0.0, 0.2, 0.4, 0.6, 0.8]),
        "q": sweep("q", [0.001, 0.005, 0.01, 0.05, 0.2]),
    }

    _log("Menu de prix (Table 2)")
    menu = menu_study()

    return {
        "figure_1": fig1,
        "sweeps": sweeps,
        "menu": menu,
        "menu_share": menu_share_of_continuous(menu),
        "published_menu_share": ref.MENU_SHARE_OF_CONTINUOUS,
    }


def main() -> None:
    results = run_all()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    peak = max(results["figure_1"], key=lambda r: r["pg_dp_s"])
    _log(f"Figure 1 : gain maximal {peak['pg_dp_s']:.2f} % "
         f"(µ={peak['mu']}, h={peak['h']}) — papier §5.2.1 : 3.81 %")
    _log(f"Part du continuum captée : {results['menu_share']} "
         f"— papier §5.2.2 : {ref.MENU_SHARE_OF_CONTINUOUS}")
    _log(f"Écrit dans {RESULTS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
