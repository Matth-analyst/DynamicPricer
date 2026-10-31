"""Chargement optionnel du moteur natif Rust (`rust/src/lib.rs`).

Le moteur natif porte les deux noyaux coûteux du solveur : le balayage de prix à prix
fixés (stratégies SP et EDP) et la recherche du seuil commun (stratégies S et SB). Tout
le reste — stratégie DP, modèle à coût de mise en route — reste en Python.

L'absence du module n'est jamais une erreur : `solve_batch` retombe sur
l'implémentation Python, qui reste la référence contre laquelle le natif est testé
(`tests/test_native.py`).

Compilation :

    cd rust && cargo build --release

Le module est cherché d'abord dans le dossier désigné par la variable d'environnement
`DYNAMICPRICER_NATIVE_DIR` (s'il est défini), puis dans `rust/target/` à la racine du dépôt.
Le dossier peut être un `target/` de cargo (sous-dossiers `release/` et `debug/`) ou
contenir directement la bibliothèque compilée. Si la variable est définie mais qu'aucun
module utilisable n'y figure, un `RuntimeWarning` est émis avant le repli : le résultat
reste correct, seule la vitesse change.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2] / "rust" / "target"
_STEM = "dynamicpricer_rust"

#: Variable d'environnement désignant un dossier contenant le moteur compilé.
ENV_VAR = "DYNAMICPRICER_NATIVE_DIR"


def _refresh_windows_copy(base: Path) -> None:
    """Dépose une copie `.pyd` du `.dll` produit par `cargo build`.

    CPython ne charge que les suffixes de `importlib.machinery.EXTENSION_SUFFIXES`,
    dont `.dll` ne fait pas partie — cette copie évite d'exiger `maturin`.

    Le rafraîchissement a lieu **avant** que le `.pyd` ne soit proposé au chargement :
    dans l'autre ordre, une copie périmée serait chargée en priorité et masquerait
    silencieusement une recompilation. La copie peut échouer si un processus a déjà le
    module chargé (Windows verrouille le fichier) ; on garde alors l'existant, faute de
    mieux, plutôt que de perdre le moteur.
    """
    dll = base / f"{_STEM}.dll"
    if not dll.exists():
        return
    pyd = base / f"{_STEM}.pyd"
    if pyd.exists() and pyd.stat().st_mtime >= dll.stat().st_mtime:
        return
    try:
        shutil.copy2(dll, pyd)
    except OSError:
        pass


def _candidates(root: Path):
    """Artefacts possibles sous `root`, release d'abord — un build debug est 10 à 50 fois plus lent.

    `root` peut être un `target/` de cargo ou le dossier contenant directement la bibliothèque.
    """
    for base in (root / "release", root / "debug", root):
        _refresh_windows_copy(base)
        for suffix in (".pyd", ".so", ".dylib"):
            yield base / f"{_STEM}{suffix}"
            yield base / f"lib{_STEM}{suffix}"


def _load_from(root: Path):
    valid = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for path in _candidates(root):
        if not path.exists() or not path.name.endswith(valid):
            continue
        try:
            spec = importlib.util.spec_from_file_location(_STEM, path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except (OSError, ImportError):
            continue
    return None


def _load():
    custom = os.environ.get(ENV_VAR)
    if custom:
        module = _load_from(Path(custom).expanduser())
        if module is not None:
            return module
        warnings.warn(
            f"{ENV_VAR}={custom!r} ne contient aucun module natif utilisable ; "
            "repli sur rust/target/ puis sur le moteur Python.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _load_from(_ROOT)


ENGINE = _load()

#: `True` si le moteur natif est disponible et utilisable.
AVAILABLE = ENGINE is not None and hasattr(ENGINE, "solve_batch_py")

#: `True` si la recherche de seuil commun (stratégies S et SB) est disponible.
HAS_COMMON_LEVEL = ENGINE is not None and hasattr(ENGINE, "search_common_level_py")


def solve_batch_truncated(pb, lam_n, gamma: float, max_sweeps: int):
    """Balayage à troncature fixée, délégué au moteur natif.

    `pb` est le problème **déjà normalisé** et `lam_n` les taux normalisés `(N, B)`.
    Le moteur recalcule sa propre constante d'uniformisation, qui vaut alors 1 : la
    normalisation y est un passage à vide, et la même fonction sert donc aussi bien
    avec des paramètres bruts qu'avec des paramètres déjà normalisés.
    """
    if not AVAILABLE:
        raise RuntimeError("moteur natif indisponible")

    lam_n = np.ascontiguousarray(lam_n, dtype=float)
    g, s = ENGINE.solve_batch_py(
        lam_n.T.reshape(-1).tolist(),
        [float(x) for x in pb.Lam],
        [float(x) for row in pb.Q for x in row],
        float(pb.mu),
        float(pb.c),
        float(pb.h),
        float(pb.b),
        int(pb.m_stock),
        int(pb.m_backlog),
        float(pb.a),
        int(max_sweeps),
    )
    gains = np.asarray(g, dtype=float) * gamma
    base_stock = np.asarray(s, dtype=int).reshape(-1, pb.N).T
    return gains, base_stock


def search_common_level(pb, lam, max_level: int):
    """Meilleur seuil commun par vecteur de prix, délégué au moteur natif.

    `pb` est le problème en **unités réelles** et `lam` la grille `(B, N)` des taux.
    Renvoie `(gains (B,), niveaux (B,))`. C'est le noyau des stratégies S et SB, dont
    le base-stock est contraint identique dans tous les environnements.
    """
    if not HAS_COMMON_LEVEL:
        raise RuntimeError("moteur natif indisponible")

    lam = np.ascontiguousarray(lam, dtype=float)
    g, levels = ENGINE.search_common_level_py(
        lam.reshape(-1).tolist(),
        [float(x) for x in pb.Lam],
        [float(x) for row in pb.Q for x in row],
        float(pb.mu),
        float(pb.c),
        float(pb.h),
        float(pb.b),
        int(pb.m_backlog),
        int(max_level),
        float(pb.a),
    )
    return np.asarray(g, dtype=float), np.asarray(levels, dtype=int)
