"""Chargement optionnel du moteur natif Rust (`rust/src/lib.rs`).

Le moteur natif ne couvre qu'un point du solveur : le balayage de prix à prix fixés,
qui est le poste dominant dès que `N` dépasse 2. Tout le reste — stratégie DP, modèle
à coût de mise en route, évaluation stationnaire — reste en Python.

L'absence du module n'est jamais une erreur : `solve_batch` retombe sur
l'implémentation Python, qui reste la référence contre laquelle le natif est testé
(`tests/test_native.py`).

Compilation :

    cd rust && cargo build --release
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1] / "rust" / "target"
_STEM = "dynamicpricer_rust"


def _candidates():
    """Artefacts possibles, release d'abord — un build debug est 10 à 50 fois plus lent."""
    for profile in ("release", "debug"):
        base = _ROOT / profile
        for suffix in (".pyd", ".so", ".dylib"):
            yield base / f"{_STEM}{suffix}"
            yield base / f"lib{_STEM}{suffix}"
        # `cargo build` produit un `.dll` sous Windows, que CPython refuse de charger :
        # seuls les suffixes de `importlib.machinery.EXTENSION_SUFFIXES` sont reconnus.
        # On en dépose une copie `.pyd` à côté plutôt que d'exiger `maturin`.
        dll = base / f"{_STEM}.dll"
        if dll.exists():
            pyd = base / f"{_STEM}.pyd"
            if not pyd.exists() or pyd.stat().st_mtime < dll.stat().st_mtime:
                try:
                    shutil.copy2(dll, pyd)
                except OSError:
                    continue
            yield pyd


def _load():
    valid = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    for path in _candidates():
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


ENGINE = _load()

#: `True` si le moteur natif est disponible et utilisable.
AVAILABLE = ENGINE is not None and hasattr(ENGINE, "solve_batch_py")


def solve_batch_truncated(pb, lam_n, gamma: float, max_sweeps: int):
    """Balayage à troncature fixée, délégué au moteur natif.

    `pb` est le problème **déjà normalisé** et `lam_n` les taux normalisés `(N, B)`.
    Le moteur recalcule sa propre constante d'uniformisation, qui vaut alors 1 : la
    normalisation y est un passage à vide, et la même fonction sert donc aussi bien
    avec des paramètres bruts qu'avec des paramètres déjà normalisés.
    """
    if not AVAILABLE:
        raise RuntimeError("moteur natif indisponible")
    import numpy as np

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
