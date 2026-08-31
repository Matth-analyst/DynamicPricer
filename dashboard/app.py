"""Dashboard Streamlit (Étape 5 de la spec).

Lancement : `streamlit run dashboard/app.py` depuis la racine du repo.

Statut : squelette (Étape 0).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="DynamicPricer", layout="wide")
st.title("DynamicPricer")
st.info("Squelette — le dashboard est implémenté à l'Étape 5 de la spec.")
