"""Valeurs publiées de Gayon, Talay-Değirmenci, Karaesmen & Örmeci (2009).

Transcrites depuis la version HAL du papier (hal-00363686), Tables 2 à 5 et §5.2.1.
Servent de référence à la validation externe (tolérances : `validation.PG_TOLERANCE_PP`
et `validation.PRICE_TOLERANCE`).

Ces valeurs ne couvrent **que** le cœur du modèle : `N=2`, `K=0`, lost sales.
Les extensions (`N>2`, `K>0`, backlog) n'ont pas de référence publiée et sont
validées par cohérence interne.
"""

# ---------------------------------------------------------------------------
# Paramètres communs à toutes les tables (§5.2.1 et §5.3)
# ---------------------------------------------------------------------------
#: Sans perte de généralité le papier fixe `a = 1`, `Λ̄ = 1`, `c = 0` (§5.2).
BASE_PARAMS = dict(Lam_bar=1.0, a=1.0, c=0.0, h=0.01, q=0.01, mu=0.11)

#: Les tables 3 à 5 utilisent `µ = 0.11` ; la Figure 2 balaie `µ` à `ε = 0.8`.
EPSILONS = (0.0, 0.3, 0.6, 0.8)


# ---------------------------------------------------------------------------
# Table 2 — Impact du menu de prix, système mono-environnement, h = 0.01
# ---------------------------------------------------------------------------
# `PG^{dp,s}` selon le nombre de prix disponibles.
# ATTENTION : la colonne « 3 prix » est une BORNE INFÉRIEURE — le papier impose
# `p3 = (p1+p2)/2` pour éviter un balayage à 100³ (Appendix E). Toute reproduction
# doit appliquer la même contrainte.
TABLE_2 = {
    # mu: (2 prix, 3 prix [borne inf.], continu)
    0.1: (0.015, 0.019, 0.020),
    0.3: (0.027, 0.032, 0.036),
    0.5: (0.014, 0.017, 0.018),
    0.7: (0.007, 0.009, 0.009),
    0.9: (0.004, 0.004, 0.005),
}

#: §5.2.2 : moyenne sur `µ ∈ {0.05, 0.10, …, 1.00}` du bénéfice relatif au continu.
MENU_SHARE_OF_CONTINUOUS = {2: 0.785, 3: 0.925}


# ---------------------------------------------------------------------------
# Table 3 — Profit gains par rapport à la stratégie S, µ = 0.11
# ---------------------------------------------------------------------------
TABLE_3 = {
    # eps: {"sb": PG^{sb,s}, "sp": PG^{sp,s}, "edp": PG^{edp,s}, "dp": PG^{dp,s}}
    0.0: {"sb": 0.000, "sp": 0.000, "edp": 0.000, "dp": 0.022},
    0.3: {"sb": 0.015, "sp": 0.000, "edp": 0.015, "dp": 0.038},
    0.6: {"sb": 0.073, "sp": 0.005, "edp": 0.074, "dp": 0.100},
    0.8: {"sb": 0.120, "sp": 0.024, "edp": 0.136, "dp": 0.152},
}


# ---------------------------------------------------------------------------
# Table 4 — Niveaux de base-stock optimaux, µ = 0.11
# ---------------------------------------------------------------------------
# S et SB n'ont qu'un seul base-stock ; SP, EDP et DP en ont un par environnement.
TABLE_4 = {
    # eps: {strategie: base-stock (scalaire) ou (s_L, s_H)}
    0.0: {"s": 8, "sb": 8, "sp": (8, 8), "edp": (8, 8), "dp": (17, 17)},
    0.3: {"s": 7, "sb": 8, "sp": (6, 11), "edp": (7, 9), "dp": (12, 20)},
    0.6: {"s": 5, "sb": 6, "sp": (4, 14), "edp": (5, 10), "dp": (7, 22)},
    0.8: {"s": 3, "sb": 4, "sp": (2, 13), "edp": (3, 10), "dp": (3, 23)},
}


# ---------------------------------------------------------------------------
# Table 5 — Prix optimaux, µ = 0.11
# ---------------------------------------------------------------------------
# Pour DP, on rapporte l'intervalle observé sur les états récurrents :
# `p_e = min_x p^dp(x,e)` et `p̄_e = max_x p^dp(x,e)` (avec `p̄_e = p^dp(1,e)`).
TABLE_5 = {
    # eps: {strategie: prix}
    0.0: {"s": 0.79, "sb": (0.79, 0.79), "sp": 0.79, "edp": (0.79, 0.79),
          "dp": {"L": (0.50, 0.85), "H": (0.50, 0.85)}},
    0.3: {"s": 0.78, "sb": (0.74, 0.82), "sp": 0.78, "edp": (0.74, 0.82),
          "dp": {"L": (0.42, 0.82), "H": (0.51, 0.87)}},
    0.6: {"s": 0.74, "sb": (0.65, 0.83), "sp": 0.75, "edp": (0.65, 0.84),
          "dp": {"L": (0.33, 0.75), "H": (0.51, 0.88)}},
    0.8: {"s": 0.75, "sb": (0.55, 0.84), "sp": 0.78, "edp": (0.57, 0.84),
          # p̄^dp_H = 0.99 tel que publié en 2009 ; la version de conférence de 2006 donne 0.88.
          "dp": {"L": (0.19, 0.65), "H": (0.51, 0.99)}},
}


# ---------------------------------------------------------------------------
# Résultats scalaires cités dans le texte
# ---------------------------------------------------------------------------
#: §5.2.1 — maximum global du profit gain en mono-environnement, sur (h, µ).
SINGLE_ENV_MAX_PG = {"value": 0.0381, "mu": 0.255, "h": 0.0123}

#: §5.3.1 — `max_µ PG^{dp,edp}` par valeur de `ε`, et le `µ` qui le réalise.
MAX_PG_DP_EDP = {
    0.3: {"value": 0.0338, "mu": 0.24},
    0.6: {"value": 0.0289, "mu": 0.44},
    0.8: {"value": 0.0324, "mu": 0.50},
}

#: §5.3.3 — écart de prix `Δp^φ` dans la région récurrente à `ε = 0.8`.
PRICE_SPREAD_EPS_08 = {"dp": 0.80, "sb": 0.29, "edp": 0.27, "s": 0.0, "sp": 0.0}
