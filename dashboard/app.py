"""Dashboard interactif DynamicPricer (Étape 5 de la spec).

Lancement, depuis la racine du repo :

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import profit_gain  # noqa: E402
from mdp import solve_mdp  # noqa: E402
from params import Problem  # noqa: E402
from strategies import solve_all  # noqa: E402

LABELS = {
    "s": "S — prix unique, stock unique",
    "sb": "SB — un prix par environnement, stock unique",
    "sp": "SP — prix unique, stock par environnement",
    "edp": "EDP — un prix et un stock par environnement",
    "dp": "DP — un prix par état, stock par environnement",
}

st.set_page_config(page_title="DynamicPricer", layout="wide")


# ---------------------------------------------------------------------------
# Calcul (mis en cache : c'est le poste coûteux)
# ---------------------------------------------------------------------------


def build_problem(N, eps, mu, h, q, a, K, b, m_backlog, m_stock, price_step) -> Problem:
    """Construit l'instance. Pour `N > 2`, les demandes sont étalées régulièrement."""
    if N == 2:
        Lam = (1.0 - eps, 1.0 + eps)
    else:
        Lam = tuple(np.linspace(1.0 - eps, 1.0 + eps, N))
    # Environnement en naissance-mort : la Condition 1 est alors automatiquement
    # satisfaite (Remarque 1 du papier), donc les base-stocks restent ordonnés.
    Q = tuple(
        tuple(q if abs(i - j) == 1 else 0.0 for j in range(N)) for i in range(N)
    )
    return Problem(
        Lam=Lam, Q=Q, mu=mu, h=h, a=a, c=0.0, K=K, b=b,
        m_stock=m_stock, m_backlog=m_backlog, price_step=price_step,
    )


@st.cache_data(show_spinner=False)
def compute(params: tuple):
    """Résout l'instance. Le cache est indexé sur le tuple de paramètres."""
    pb = build_problem(*params)
    if pb.K > 0:
        # Le balayage de prix en lot ne couvre pas le modèle à coût de mise en route
        # (état augmenté `σ`) : seule DP est disponible dans ce régime.
        sol = solve_mdp(pb)
        return dict(mode="setup", g=sol.g, sS=sol.sS, prices=sol.prices,
                    states=pb.states, N=pb.N, Lam=pb.Lam)
    sols = solve_all(pb)
    return dict(
        mode="full",
        gains={k: v.g for k, v in sols.items()},
        base_stock={k: list(v.base_stock) for k, v in sols.items()},
        prices={k: [float(p) for p in v.prices] for k, v in sols.items() if k != "dp"},
        dp_prices=sols["dp"].prices,
        states=pb.states, N=pb.N, Lam=pb.Lam,
    )


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

st.title("DynamicPricer")
st.caption(
    "Tarification et production d'un système make-to-stock à demande fluctuante. "
    "Modèle : Gayon, Talay-Değirmenci, Karaesmen & Örmeci (2009)."
)

sb = st.sidebar
sb.header("Paramètres")

N = sb.number_input("Nombre d'environnements `N`", 2, 4, 2)
eps = sb.slider("Variabilité de la demande `ε`", 0.0, 0.9, 0.8, 0.05)
mu = sb.slider("Taux de service `µ`", 0.01, 1.0, 0.11, 0.01)
h = sb.slider("Coût de stockage `h`", 0.001, 0.10, 0.01, 0.001, format="%.3f")
q = sb.slider("Taux de transition `q`", 0.001, 0.20, 0.01, 0.001, format="%.3f")
a = sb.slider("Sensibilité au prix `a`", 0.5, 2.0, 1.0, 0.1)

sb.subheader("Extensions")
K = sb.slider("Coût de mise en route `K`", 0.0, 2.0, 0.0, 0.05)
m_backlog = sb.number_input("Backlog maximum", 0, 20, 0)
b = sb.slider("Coût de pénurie `b`", 0.0, 0.5, 0.0, 0.005, format="%.3f",
              disabled=m_backlog == 0)

sb.subheader("Numérique")
price_step = sb.select_slider("Pas de la grille de prix", [0.1, 0.05, 0.02, 0.01], 0.05)
m_stock = sb.number_input("Troncature du stock `m`", 20, 200, 60, 10)

sb.caption(
    f"Grille EDP : {int(round(1 / a / price_step)) ** N} combinaisons de prix. "
    "Le coût croît en `(1/pas)^N` — affiner le pas ou augmenter `N` peut être long."
)

params = (N, eps, mu, h, q, a, K, b if m_backlog else 0.0, m_backlog, m_stock, price_step)

if not st.button("Calculer", type="primary"):
    st.info("Réglez les paramètres puis lancez le calcul. Les résultats sont mis en cache.")
    st.stop()

with st.spinner("Résolution des programmes dynamiques…"):
    try:
        res = compute(params)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Le calcul a échoué : {exc}")
        st.stop()

env_names = [f"env {i + 1} (Λ={L:.2f})" for i, L in enumerate(res["Lam"])]

# ---------------------------------------------------------------------------
# Régime à coût de mise en route : seule DP est disponible
# ---------------------------------------------------------------------------

if res["mode"] == "setup":
    st.warning(
        "Avec `K > 0`, l'état du serveur (marche/arrêt) s'ajoute à l'état du système. "
        "Le balayage de prix en lot ne couvre pas ce modèle : seule la stratégie DP "
        "est calculée ici, et la comparaison des cinq stratégies n'est pas disponible. "
        "Mettre `K = 0` pour y revenir."
    )
    st.metric("Profit moyen de long terme (DP)", f"{res['g']:.5f}")
    st.subheader("Bande d'hystérésis par environnement")
    st.dataframe(pd.DataFrame(
        [{"environnement": env_names[e], "redémarrer sous `s_e`": s, "produire jusqu'à `S_e`": S}
         for e, (s, S) in enumerate(res["sS"])]
    ), hide_index=True)
    st.caption(
        "On produit tant que le stock est sous `S_e`, et l'on ne redémarre le serveur "
        "qu'une fois retombé sous `s_e`. L'écart entre les deux seuils est ce qui évite "
        "de payer le coût de mise en route trop souvent."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Comparaison des cinq stratégies
# ---------------------------------------------------------------------------

gains = res["gains"]
pg = {k: 100 * profit_gain(gains[k], gains["s"]) for k in gains}

tab1, tab2, tab3 = st.tabs(["Comparaison", "Stocks et prix", "Recommandation"])

with tab1:
    cols = st.columns(4)
    for col, k in zip(cols, ("sb", "sp", "edp", "dp")):
        col.metric(k.upper(), f"{gains[k]:.5f}", f"{pg[k]:+.2f} % vs S")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    order = ["s", "sb", "sp", "edp", "dp"]
    ax.bar(order, [pg[k] for k in order], color="#4C78A8")
    ax.set_ylabel("profit gain sur S (%)")
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    st.dataframe(pd.DataFrame([
        {"stratégie": LABELS[k], "profit moyen": round(gains[k], 6),
         "gain sur S (%)": round(pg[k], 2), "base-stock": res["base_stock"][k]}
        for k in order
    ]), hide_index=True)

with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("Niveaux de stock par environnement")
        fig, ax = plt.subplots(figsize=(6, 4))
        width = 0.15
        idx = np.arange(res["N"])
        for i, k in enumerate(["s", "sb", "sp", "edp", "dp"]):
            ax.bar(idx + i * width, res["base_stock"][k], width, label=k.upper())
        ax.set_xticks(idx + 2 * width)
        ax.set_xticklabels(env_names, rotation=15)
        ax.set_ylabel("base-stock")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

    with right:
        st.subheader("Prix dynamique en fonction du stock")
        fig, ax = plt.subplots(figsize=(6, 4))
        s_bar = max(res["base_stock"]["dp"])
        rec = res["states"] <= s_bar
        for e in range(res["N"]):
            ax.plot(res["states"][rec], res["dp_prices"][rec, e], marker=".", label=env_names[e])
        for e, p in enumerate(res["prices"]["edp"]):
            ax.axhline(p, ls="--", lw=1, alpha=0.6, color=f"C{e}")
        ax.set_xlabel("niveau de stock `x`")
        ax.set_ylabel("prix")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "Traits pleins : prix DP, décroissants en stock (Théorème 2). "
            "Pointillés : prix EDP correspondants, constants par environnement. "
            "Seuls les états récurrents `x ≤ max_e s^dp_e` sont tracés."
        )

# ---------------------------------------------------------------------------
# Recommandation
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Que faire, sur ce jeu de paramètres")
    gain_edp = pg["edp"]
    extra_dp = 100 * profit_gain(gains["dp"], gains["edp"])

    if gain_edp < 2.0:
        verdict, strategy = "success", "**S — un prix unique**"
        why = (
            f"Différencier les prix ne rapporte que {gain_edp:.1f} % ici. "
            "En dessous de ~2 %, le gain ne couvre pas le coût d'exploitation d'une "
            "politique plus complexe, ni la friction créée côté client."
        )
    elif extra_dp < 1.5:
        verdict, strategy = "info", "**EDP — un prix par environnement**"
        why = (
            f"EDP rapporte {gain_edp:.1f} % sur un prix unique, et DP n'y ajoute que "
            f"{extra_dp:.1f} %. Passer à une tarification pilotée par le stock exige un "
            "suivi en temps réel et multiplie les changements de prix, pour un gain "
            "marginal."
        )
    else:
        verdict, strategy = "warning", "**DP — prix dépendant du stock**"
        why = (
            f"DP dépasse EDP de {extra_dp:.1f} %, ce qui est inhabituel : c'est le "
            "régime où le niveau de stock porte réellement de l'information. À mettre "
            "en balance avec le coût d'un pilotage en temps réel."
        )

    getattr(st, verdict)(f"Stratégie recommandée : {strategy}")
    st.write(why)

    notes = []
    if eps < 0.2:
        notes.append(
            f"**Demande peu variable** (`ε = {eps:.2f}`) — c'est le déterminant principal. "
            "Sans fluctuation, aucune tarification adaptative ne se justifie."
        )
    if q > 0.05:
        notes.append(
            f"**Environnement instable** (`q = {q:.3f}`) — les régimes se succèdent trop "
            "vite pour qu'une politique de prix produise ses effets. Le gain diminue "
            "quand `q` augmente."
        )
    if mu > 0.6:
        notes.append(
            f"**Capacité abondante** (`µ = {mu:.2f}`) — le réapprovisionnement suffit à "
            "absorber les fluctuations, et la tarification perd son intérêt."
        )
    if mu < 0.05:
        notes.append(
            f"**Capacité très contrainte** (`µ = {mu:.2f}`) — le stock reste quasi nul et "
            "l'on vend en permanence au prix fort ; le prix a peu de marge de manœuvre."
        )
    if h > 0.05:
        notes.append(
            f"**Stockage coûteux** (`h = {h:.3f}`) — la tarification gagne en valeur, "
            "car elle sert à écouler avant que la facture de stockage ne tombe."
        )
    for n in notes:
        st.markdown(f"- {n}")

    st.divider()
    st.markdown(
        "**Limite du modèle.** Ni le coût de mise en œuvre d'une politique, ni la "
        "réaction des clients aux variations de prix ne sont facturés. Les gains "
        "affichés pour DP sont donc des **majorants**, et son avantage réel sur EDP "
        "est plus étroit que ce que montrent ces chiffres."
    )
