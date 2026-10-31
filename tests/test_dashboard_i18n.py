"""Tableau de bord bilingue : catalogue de textes, et exécution réelle dans les deux langues."""

import ast
import re
from pathlib import Path

import pytest

from dynamicpricer.ui_texts import LANGUAGES, TEXTS, translate

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
PLACEHOLDER = re.compile(r"<<(\w+)>>")


def test_languages_share_the_same_keys():
    assert set(LANGUAGES) == {"fr", "en"}
    assert TEXTS["fr"].keys() == TEXTS["en"].keys()


def test_placeholders_match_across_languages():
    for key, fr in TEXTS["fr"].items():
        assert set(PLACEHOLDER.findall(fr)) == set(PLACEHOLDER.findall(TEXTS["en"][key])), key


def test_no_empty_translation_and_english_differs_from_french_where_expected():
    assert all(text.strip() for lang in TEXTS.values() for text in lang.values())
    differing = sum(TEXTS["fr"][k] != TEXTS["en"][k] for k in TEXTS["fr"])
    assert differing > 0.85 * len(TEXTS["fr"])  # le reste : nombres, noms propres, « Extensions »


def test_translate_substitutes_variables_and_falls_back_to_french():
    assert "3.50" in translate("en", "r.footer", t="3.50", engine="X", gamma="1", ok="ok")
    assert translate("de", "btn.run") == TEXTS["fr"]["btn.run"]  # langue inconnue : français
    assert translate("en", "btn.run") == "Compute"


def _keys_used_in_app() -> set[str]:
    """Clés passées à `T("...")` dans app.py, y compris le préfixe des clés dynamiques."""
    keys = set()
    for node in ast.walk(ast.parse(APP.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "T" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant):
                keys.add(arg.value)
            elif isinstance(arg, ast.JoinedStr):  # T(f"lab.{k}")
                keys.update(k for k in TEXTS["fr"] if k.startswith(arg.values[0].value))
    return keys


def test_every_key_used_by_the_app_exists_and_none_is_orphaned():
    used = _keys_used_in_app()
    assert used <= set(TEXTS["fr"]), used - set(TEXTS["fr"])
    internal = {"sb.language"}  # lu directement par `translate("fr", ...)` dans app.py
    assert set(TEXTS["fr"]) - used - internal == set()


# --- Exécution réelle de l'application (nécessite l'extra `dashboard`) -----------------------

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def _banners(at):
    return [*at.info, *at.success, *at.warning]


def _run(lang: str, compute: bool = True):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=180)
    at.session_state["lang_choice"] = lang
    at.run()
    assert not at.exception, at.exception
    if compute:
        at.button[0].click().run()
        assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_app_runs_in_each_language(lang):
    at = _run(lang)
    assert at.sidebar.radio[0].value == lang
    assert [t.label for t in at.tabs] == [
        TEXTS[lang][k] for k in ("tab.results", "tab.stock", "tab.reco", "tab.model", "tab.method")
    ]
    assert at.button[0].label == TEXTS[lang]["btn.run"]
    assert len(at.metric) == 4 and len(at.get("image")) + len(at.get("imgs")) >= 2  # nom selon la version  # résultats et figures produits


def test_switching_language_keeps_parameters_and_results():
    """Le libellé traduit ne doit pas changer l'identité des widgets, ni effacer les résultats."""
    at = _run("fr")
    at.sidebar.slider[0].set_value(0.5)
    at.button[0].click().run()  # recalcule avec epsilon = 0.5
    gains_fr = [m.value for m in at.metric]
    assert len(gains_fr) == 4

    at.sidebar.radio[0].set_value("en").run()
    assert not at.exception, at.exception
    assert at.sidebar.slider[0].value == 0.5  # paramètre conservé
    assert at.sidebar.slider[0].label == TEXTS["en"]["sb.eps"]  # libellé traduit
    assert [m.value for m in at.metric] == gains_fr  # résultats toujours affichés, inchangés
    assert any("Recommended strategy" in e.value for e in _banners(at))


def test_switching_language_after_editing_parameters_does_not_recompute():
    """Paramètres modifiés mais non recalculés : on ne lance pas de calcul à l'insu de l'utilisateur."""
    at = _run("fr")
    at.sidebar.slider[0].set_value(0.3)  # sans cliquer sur « Calculer »
    at.sidebar.radio[0].set_value("en").run()
    assert not at.exception, at.exception
    assert len(at.metric) == 0
    assert at.sidebar.slider[0].value == 0.3
