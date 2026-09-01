"""Paramètres du modèle (§3 de la spec).

Contient `Problem`, la dataclass qui regroupe tous les paramètres d'une instance,
ainsi que les utilitaires de demande/revenu et la normalisation par uniformisation.

Référence : Gayon, Talay-Değirmenci, Karaesmen, Örmeci (2009), §3 et Appendix E.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Problem:
    """Une instance du problème make-to-stock à demande modulée par un environnement.

    Convention d'indexation : les environnements sont numérotés `0..N-1` et sont
    supposés ordonnés par demande potentielle croissante (`Lam[0] <= ... <= Lam[N-1]`),
    ce qui correspond à l'ordre des labels du papier (environnement 1 = le pire).
    Pour `N=2`, l'indice 0 est Low et l'indice 1 est High.
    """

    # --- Environnement ---
    Lam: tuple[float, ...] = (0.7, 1.3)
    """Demandes potentielles `Λ_e`, une par environnement."""
    Q: tuple[tuple[float, ...], ...] = ((0.0, 0.01), (0.01, 0.0))
    """Matrice des taux de transition `q_ej` (diagonale ignorée, forcée à 0)."""

    # --- Production ---
    mu: float = 0.11
    """Taux de service `µ`."""
    c: float = 0.0
    """Coût de production unitaire."""
    K: float = 0.0
    """Coût fixe de commande, prélevé dès que la quantité produite est > 0 (Étape 1.2)."""

    # --- Inventaire ---
    h: float = 0.01
    """Coût de stockage unitaire par unité de temps (coût linéaire `h(x) = h·x`)."""
    b: float = 0.0
    """Coût de pénurie unitaire par unité de temps, appliqué au backlog (Étape 1.3)."""
    m_stock: int = 60
    """Troncature haute de l'espace d'état : niveau de stock maximum."""
    m_backlog: int = 0
    """Troncature basse : backlog maximum autorisé. `0` = lost sales (cas du papier)."""

    # --- Demande ---
    a: float = 1.0
    """Sensibilité de la demande au prix : `λ_e(p) = Λ_e·(1 - a·p)`."""
    price_step: float = 0.01
    """Pas de discrétisation de la grille de prix (Appendix E : 0.01 sur [0, 1/a])."""

    # --- Numérique ---
    tol: float = 1e-5
    """Critère d'arrêt de la value iteration (5 décimales, cf. Appendix E)."""
    max_iter: int = 200_000

    def __post_init__(self) -> None:
        N = len(self.Lam)
        if N < 1:
            raise ValueError("Il faut au moins un environnement.")
        Q = np.asarray(self.Q, dtype=float)
        if Q.shape != (N, N):
            raise ValueError(f"Q doit être {N}x{N}, reçu {Q.shape}.")
        if np.any(Q < 0):
            raise ValueError("Les taux de transition doivent être positifs ou nuls.")
        if any(L <= 0 for L in self.Lam):
            raise ValueError("Les demandes potentielles doivent être strictement positives.")
        for name in ("mu", "a"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` doit être strictement positif.")
        for name in ("c", "K", "h", "b"):
            if getattr(self, name) < 0:
                raise ValueError(f"`{name}` doit être positif ou nul.")
        if self.m_stock < 1 or self.m_backlog < 0:
            raise ValueError("Troncature invalide.")
        if not 0 < self.price_step <= 1 / self.a:
            raise ValueError("`price_step` doit être dans ]0, 1/a].")
        if N > 1 and list(self.Lam) != sorted(self.Lam):
            raise ValueError(
                "Les environnements doivent être ordonnés par demande potentielle croissante "
                "(convention du papier, requise par la Condition 1)."
            )

    # ------------------------------------------------------------------
    # Constructeurs
    # ------------------------------------------------------------------
    @classmethod
    def two_env(cls, eps: float, q: float = 0.01, Lam_bar: float = 1.0, **kw) -> "Problem":
        """Cas à deux environnements du papier : `Λ_L = Λ̄ - ε`, `Λ_H = Λ̄ + ε`, `q_LH = q_HL = q`.

        C'est le constructeur utilisé pour toute la validation de l'Étape 3.1.
        """
        return cls(
            Lam=(Lam_bar - eps, Lam_bar + eps),
            Q=((0.0, q), (q, 0.0)),
            **kw,
        )

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        """Nombre d'environnements."""
        return len(self.Lam)

    @property
    def p_max(self) -> float:
        """Prix au-delà duquel la demande s'annule : `1/a`."""
        return 1.0 / self.a

    @property
    def x_min(self) -> int:
        return -self.m_backlog

    @property
    def x_max(self) -> int:
        return self.m_stock

    @property
    def states(self) -> np.ndarray:
        """Les niveaux de stock `x` de la grille tronquée, de `-m_backlog` à `m_stock`."""
        return np.arange(self.x_min, self.x_max + 1)

    @property
    def price_grid(self) -> np.ndarray:
        """Grille de prix `P` discrétisée au pas `price_step` sur `[0, 1/a]`."""
        n = int(round(self.p_max / self.price_step))
        return np.linspace(0.0, self.p_max, n + 1)

    @property
    def search_price_grid(self) -> np.ndarray:
        """Grille de prix privée de ses deux extrémités, utilisée par les balayages.

        Le taux de revenu `r_e(λ) = λ·p_e(λ)` s'annule aux deux bouts — en `p = 0`
        parce que le prix est nul, en `p = 1/a` parce que la demande l'est — donc
        aucune des deux extrémités n'est jamais optimale et les exclure ne change
        aucun argmax.

        C'est en revanche indispensable numériquement : en `p = 1/a` la demande
        disparaît, la chaîne dérive vers son base-stock et n'en repart jamais, et la
        value iteration relative n'y converge plus. Un seul prix dégénéré suffirait à
        bloquer tout un balayage en lot, qui n'aboutit que lorsque *tous* ses éléments
        ont convergé.
        """
        return self.price_grid[1:-1]

    def is_birth_death(self) -> bool:
        """`True` si l'environnement est un processus de naissance-mort.

        Dans ce cas la Condition 1 du papier est automatiquement satisfaite et la
        monotonicité des base-stock (Théorème 3) est acquise sans vérification.
        """
        Q = np.asarray(self.Q, dtype=float)
        for e in range(self.N):
            for j in range(self.N):
                if abs(e - j) > 1 and Q[e, j] > 0:
                    return False
        return True

    def satisfies_condition_1(self) -> bool:
        """Vérifie la Condition 1 du papier (§4.3) sur la matrice de transition.

        a. `q_ej >= q_{e+1,j}` pour tout `j <= e-1`
        b. `q_{e+1,j} >= q_ej` pour tout `j >= e+2`
        """
        Q = np.asarray(self.Q, dtype=float)
        for e in range(self.N - 1):
            for j in range(e):
                if Q[e, j] < Q[e + 1, j]:
                    return False
            for j in range(e + 2, self.N):
                if Q[e + 1, j] < Q[e, j]:
                    return False
        return True

    # ------------------------------------------------------------------
    # Demande, prix, revenu (fonction de demande linéaire, §5.1)
    # ------------------------------------------------------------------
    def demand(self, p: float, e: int) -> float:
        """`λ_e(p) = Λ_e·(1 - a·p)`, tronquée à `[0, Λ_e]`."""
        return float(self.Lam[e] * np.clip(1.0 - self.a * p, 0.0, 1.0))

    def price(self, lam: float, e: int) -> float:
        """Fonction inverse `p_e(λ)`."""
        return float((1.0 - lam / self.Lam[e]) / self.a)

    def revenue(self, lam: float, e: int) -> float:
        """Taux de revenu `r_e(λ) = λ·p_e(λ)`."""
        return float(lam * self.price(lam, e))

    @property
    def p_star(self) -> float:
        """Prix maximisant le taux de revenu : `p* = 1/(2a)`, indépendant de `Λ_e` (§5.1)."""
        return 1.0 / (2.0 * self.a)

    def holding_cost(self, x):
        """Coût instantané de détention : `h·x` si `x >= 0`, `b·|x|` si `x < 0` (Étape 1.3)."""
        x = np.asarray(x, dtype=float)
        return np.where(x >= 0, self.h * x, self.b * np.abs(x))

    # ------------------------------------------------------------------
    # Uniformisation / normalisation (§3.1)
    # ------------------------------------------------------------------
    @property
    def uniformization_constant(self) -> float:
        """`Γ = µ + Σ_e Λ_e + Σ_e Σ_{j≠e} q_ej`, le facteur qui ramène le total à 1."""
        Q = np.asarray(self.Q, dtype=float).copy()
        np.fill_diagonal(Q, 0.0)
        return float(self.mu + sum(self.Lam) + Q.sum())

    def normalized(self) -> tuple["Problem", float]:
        """Renvoie `(problème normalisé, Γ)` tel que `µ + ΣΛ_e + ΣΣq_ej = 1`.

        Tous les **taux** (`µ`, `Λ_e`, `q_ej`) et les **coûts par unité de temps**
        (`h`, `b`) sont divisés par `Γ`. Les coûts par événement (`c`, `K`) et les
        prix sont inchangés : le revenu `r_e(λ) = λ·p_e(λ)` est un taux, il est
        donc déjà mis à l'échelle via `λ`.

        Le gain moyen du problème normalisé se reconvertit en gain réel par
        `g_réel = Γ · g_normalisé`. Les *profit gains* étant des ratios, ils sont
        invariants par cette transformation.
        """
        G = self.uniformization_constant
        Q = np.asarray(self.Q, dtype=float).copy() / G
        np.fill_diagonal(Q, 0.0)
        return (
            replace(
                self,
                Lam=tuple(L / G for L in self.Lam),
                Q=tuple(tuple(row) for row in Q),
                mu=self.mu / G,
                h=self.h / G,
                b=self.b / G,
            ),
            G,
        )
