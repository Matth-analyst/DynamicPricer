"""Validation contre le papier — Étape 3, « obligatoire, non négociable ».

Ces tests portent la grille de prix complète de l'Appendix E (pas 0.01) sur les quatre
instances de référence : ils prennent quelques minutes et sont donc marqués `slow`,
donc exclus de la suite par défaut. Pour les exécuter :

    pytest -m slow

Ce qui est vérifié ici a une portée différente du reste de la suite : les autres tests
contrôlent des invariants internes, ceux-ci confrontent le solveur à des valeurs
**publiées**. Seul le cœur du modèle (`N=2`, `K=0`, lost sales) peut l'être ; les
extensions n'ont pas de référence externe et sont traitées en fin de fichier.
"""

import pytest

import reference_gayon2009 as ref
import validation as V

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def solved():
    return V.solve_reference_instances()


# ---------------------------------------------------------------------------
# Étape 3.1 — validation externe
# ---------------------------------------------------------------------------


def test_table_4_base_stock_levels_are_reproduced_exactly(solved):
    """Table 4 : les base-stocks sont des entiers, donc l'exactitude est atteignable."""
    rows = V.compare_table_4(solved)
    wrong = [r for r in rows if not r["exact"]]
    assert not wrong, f"base-stocks non reproduits : {wrong}"


def test_table_3_profit_gains_match_within_tolerance(solved):
    """Table 3 : écart maximum toléré `PG_TOLERANCE_PP` point de pourcentage."""
    rows = V.compare_table_3(solved)
    off = [r for r in rows if abs(r["gap_pp"]) > V.PG_TOLERANCE_PP]
    assert not off, f"profit gains hors tolérance : {off}"


def test_table_5_static_prices_match_within_tolerance(solved):
    """Table 5, stratégies S/SB/SP/EDP : prix à la tolérance près."""
    rows = [r for r in V.compare_table_5(solved) if not r["quantity"].startswith("p_dp")]
    off = [r for r in rows if r["gap"] > V.PRICE_TOLERANCE]
    assert not off, f"prix hors tolérance : {off}"


def test_dp_price_ranges_match_except_two_documented_outliers(solved):
    """Table 5, stratégie DP : deux valeurs publiées restent en écart.

    `ε=0.6, p^dp_L = 0.33` (calculé 0.31) et `ε=0.8, p̄^dp_H = 0.99` (calculé 0.88).
    La seconde rompt une progression que le solveur rend régulière
    (`0.85 → 0.87 → 0.88 → 0.88`) alors que les autres valeurs DP à `ε = 0.8`
    coïncident au centime près, et elle est insensible à la troncature comme à la
    tolérance : le soupçon porte sur le papier. Ce test **fige l'écart** pour qu'une
    régression future du solveur ne passe pas inaperçue derrière lui.
    """
    rows = [r for r in V.compare_table_5(solved) if r["quantity"].startswith("p_dp")]
    known = {(0.6, "p_dp_L"), (0.8, "p_dp_H")}
    for r in rows:
        key = (r["eps"], r["quantity"])
        if key in known:
            assert r["gap"] > V.PRICE_TOLERANCE, (
                f"{key} concorde désormais — mettre à jour la liste des écarts connus"
            )
        else:
            assert r["gap"] <= V.PRICE_TOLERANCE, f"nouvel écart DP : {r}"


def test_structural_properties_hold_on_every_instance(solved):
    """Lemme 1, Théorèmes 2 et 3, et §5.3.2 — vérifiés sur les quatre instances."""
    for row in V.structural_properties(solved):
        failed = [k for k, v in row.items() if k != "eps" and v is not True]
        assert not failed, f"propriétés en défaut à ε={row['eps']} : {failed}"


def test_figure_2_reproduces_the_published_shape():
    """Figure 2 : `PG^{dp,edp}` reste marginal alors que `PG^{dp,sp}` culmine.

    C'est la conclusion de fond du papier (§5.3.4) — EDP capte l'essentiel du bénéfice
    de DP. On vérifie la forme et l'ordre de grandeur, pas un tracé point par point.
    """
    rows = V.figure_2_data(mu_values=[0.11, 0.26, 0.51, 0.81])
    assert all(r["PG_dp_sp"] >= r["PG_edp_sp"] - 1e-9 for r in rows)
    # §5.3.1 : `max_µ PG^{dp,edp}` vaut 3.24 % à ε = 0.8 sur tout le balayage.
    assert max(r["PG_dp_edp"] for r in rows) < 100 * ref.MAX_PG_DP_EDP[0.8]["value"] + 0.5
    # Non-monotonicité en µ : le gain culmine à capacité intermédiaire.
    gains = [r["PG_dp_sp"] for r in rows]
    assert gains[0] < max(gains) and gains[-1] < max(gains)


# ---------------------------------------------------------------------------
# Étape 3.2 — cohérence interne des extensions (aucune référence publiée)
# ---------------------------------------------------------------------------


def test_extensions_pass_their_limit_cases():
    """`N>2`, `K>0` et backlog : cas limites, pas reproduction de résultats publiés.

    C'est une validation **plus faible** que celle de l'Étape 3.1, et le rapport doit
    la présenter comme telle : elle atteste la cohérence interne du solveur, pas la
    correction du modèle vis-à-vis d'une source externe.
    """
    tolerances = {
        "N>2 : H dédoublé reproduit N=2": 1e-6,
        "K=0 : le modèle à mise en route redonne l'Étape 1.1": 1e-9,
        "b→∞ : le backlog converge vers lost sales": 1e-6,
    }
    for row in V.extension_checks():
        assert row["gap"] <= tolerances[row["case"]], row
