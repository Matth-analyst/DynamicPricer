"""Chargement du moteur natif : variable d'environnement `DYNAMICPRICER_NATIVE_DIR`.

Chaque cas tourne dans un sous-processus, car le module natif est chargé une fois pour
toutes à l'import de `dynamicpricer.native`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import dynamicpricer
from dynamicpricer import native

SRC = Path(dynamicpricer.__file__).resolve().parents[1]
ENV_VAR = native.ENV_VAR


def _run(code: str, native_dir: Path | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    env.pop(ENV_VAR, None)
    if native_dir is not None:
        env[ENV_VAR] = str(native_dir)
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def test_unusable_directory_warns_then_falls_back(tmp_path):
    out = _run("from dynamicpricer import native; print(native.AVAILABLE)", tmp_path)
    assert "RuntimeWarning" in out.stderr and ENV_VAR in out.stderr
    assert out.stdout.strip() in {"True", "False"}  # repli : jamais une erreur


def test_unset_variable_emits_no_warning():
    out = _run("from dynamicpricer import native; print(native.AVAILABLE)")
    assert "RuntimeWarning" not in out.stderr


@pytest.mark.skipif(not native.AVAILABLE, reason="moteur natif non compilé")
@pytest.mark.parametrize("layout", ["flat", "cargo"])
def test_custom_directory_takes_priority(tmp_path, layout):
    """Un artefact copié hors du dépôt est chargé en priorité, à plat ou en `release/`."""
    artifact = Path(native.ENGINE.__file__)
    target = tmp_path / ("release" if layout == "cargo" else "")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, target / artifact.name)

    out = _run(
        "from dynamicpricer import native; print(native.AVAILABLE, native.ENGINE.__file__)",
        tmp_path,
    )
    available, loaded = out.stdout.split(maxsplit=1)
    assert available == "True"
    assert Path(loaded.strip()).resolve().is_relative_to(tmp_path.resolve())
    assert "RuntimeWarning" not in out.stderr
