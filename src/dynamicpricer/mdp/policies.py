"""Extraction des politiques (seuils base-stock, bande d'hystérésis) depuis une valeur."""

from __future__ import annotations

import numpy as np

from ..params import Problem


def base_stock_levels(v: np.ndarray, pb: Problem) -> np.ndarray:
    """Seuils de production `s_e = min{x : Δv(x,e) <= c}`, vectorisé (Définition 1).

    Renvoie un tableau de forme `v.shape[1:]` — donc `(N,)` pour une résolution
    simple, `(N, B)` pour une résolution en lot.

    Par concavité de `v` (Lemme 1), `{x : Δv(x,e) > c}` est un segment initial : ce
    minimum définit bien un base-stock, on produit ssi `x < s_e`. Les égalités sont
    tranchées en faveur du non-réapprovisionnement, comme le `max{f(x,e), f(x+1,e)-c}`
    du papier.
    """
    stop = (v[1:] - v[:-1]) <= pb.c
    found = stop.any(axis=0)
    first = pb.states[np.argmax(stop, axis=0)]
    return np.where(found, first, pb.x_max)


def extract_base_stock(v: np.ndarray, pb: Problem) -> tuple[int, ...]:
    """Version scalaire de `base_stock_levels`, pour une fonction de valeur `(x, e)`."""
    return tuple(int(s) for s in base_stock_levels(v, pb))


def extract_setup_policy(d: np.ndarray, pb: Problem) -> tuple[tuple[int, int], ...]:
    """Extrait la bande d'hystérésis `(s_e, S_e)` du modèle à mise en route.

    `d(x,e) = inner(x,e,1) - inner(x,e,0)` est le gain net de faire tourner le
    serveur, **hors** coût de mise en route. La décision optimale s'en déduit :

        serveur en marche (`σ=1`) : on continue ssi `d(x,e) > 0`
        serveur arrêté   (`σ=0`) : on redémarre ssi `d(x,e) > K`

    D'où `S_e = min{x : d(x,e) <= 0}` (seuil d'arrêt) et `s_e = min{x : d(x,e) <= K}`
    (seuil de redémarrage), avec `s_e <= S_e` puisque `K >= 0`.

    `d` doit être décroissante en `x` pour que ces deux seuils définissent bien une
    politique de type bande ; on le **constate** plutôt que de le supposer, d'où le
    `ValueError` en cas de violation.
    """
    tol = 1e-9
    # La dernière ligne `x_max` est écartée : la troncature y interdit de produire,
    # ce qui y fait remonter `d` artificiellement. Les deux seuils doivent de toute
    # façon être strictement intérieurs à la grille.
    interior = d[:-1]
    if np.any(np.diff(interior, axis=0) > tol):
        raise ValueError(
            "d(x,e) n'est pas décroissante en x : la politique optimale n'est pas "
            "une bande d'hystérésis à deux seuils."
        )

    def first_below(col, threshold, name, e):
        idx = np.flatnonzero(col <= threshold + tol)
        if idx.size == 0:
            raise ValueError(
                f"Seuil {name} hors grille en environnement {e} : augmenter `m_stock` "
                "jusqu'à ce que la troncature ne morde plus (cf. Appendix E)."
            )
        return int(pb.states[idx[0]])

    return tuple(
        (
            first_below(interior[:, e], pb.K, "s_e", e),
            first_below(interior[:, e], 0.0, "S_e", e),
        )
        for e in range(pb.N)
    )
