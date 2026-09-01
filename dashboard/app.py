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

st.markdown("""
<style>
  /* Densité — obtenue par la typographie et le `gap` du flex uniquement.
     Ne pas toucher au `padding` ni aux marges des titres : Streamlit leur applique
     une marge basse négative de -16 px qui compense exactement ce padding. Y toucher
     désaligne toute la colonne et fait chevaucher les blocs. De même, ne pas imposer
     de largeur à la sidebar : cela casse sa mise en page responsive dans un panneau
     étroit, où elle vient alors recouvrir le contenu principal. */
  .block-container { padding-top: 1.6rem !important; padding-bottom: 1.2rem !important; }
  [data-testid="stVerticalBlock"] { gap: .5rem; }

  h1 { font-size: 24px !important; letter-spacing: -.015em; }
  h2 { font-size: 17px !important; }
  h3, h4 { font-size: 14.5px !important; }
  p, li { font-size: 13px !important; line-height: 1.5; }

  [data-testid="stMetricValue"] { font-size: 20px !important; }
  [data-testid="stMetricLabel"] p { font-size: 11px !important; letter-spacing: .04em;
    text-transform: uppercase; opacity: .75; }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }

  /* Curseurs : la barre min/max coûte 22 px de haut par curseur, et la valeur courante
     reste affichée au-dessus du pouce. C'est le seul réglage structurel conservé. */
  [data-testid="stSliderTickBar"] { display: none !important; }
  [data-testid="stSidebar"] label p { font-size: 12px !important; }
  [data-testid="stSidebar"] h2 { font-size: 12px !important; text-transform: uppercase;
    letter-spacing: .08em; opacity: .7; }

  [data-testid="stTabs"] button p { font-size: 13px !important; }
  [data-testid="stCaptionContainer"] p { font-size: 11.5px !important; line-height: 1.5; opacity: .85; }
  .katex { font-size: 1em !important; }
  [data-testid="stTable"] td, [data-testid="stTable"] th { font-size: 12px !important; }
  code { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)


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
    Q = tuple(tuple(q if abs(i - j) == 1 else 0.0 for j in range(N)) for i in range(N))
    return Problem(
        Lam=Lam, Q=Q, mu=mu, h=h, a=a, c=0.0, K=K, b=b,
        m_stock=m_stock, m_backlog=m_backlog, price_step=price_step,
    )


@st.cache_data(show_spinner=False)
def compute(params: tuple):
    """Résout l'instance. Le cache est indexé sur le tuple de paramètres."""
    pb = build_problem(*params)
    common = dict(states=pb.states, N=pb.N, Lam=pb.Lam, gamma=pb.uniformization_constant)
    if pb.K > 0:
        # Le balayage de prix en lot ne couvre pas le modèle à coût de mise en route
        # (état augmenté `σ`) : seule DP est disponible dans ce régime.
        sol = solve_mdp(pb)
        return dict(mode="setup", g=sol.g, sS=sol.sS, prices=sol.prices,
                    sweeps=sol.n_iter, **common)
    sols = solve_all(pb)
    return dict(
        mode="full",
        gains={k: v.g for k, v in sols.items()},
        base_stock={k: list(v.base_stock) for k, v in sols.items()},
        prices={k: [float(p) for p in v.prices] for k, v in sols.items() if k != "dp"},
        dp_prices=sols["dp"].prices,
        **common,
    )


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

st.title("DynamicPricer")
st.caption(
    "Tarification et production d'un système make-to-stock à capacité limitée sous demande "
    "modulée par une chaîne de Markov. Modèle : Gayon, Talay-Değirmenci, Karaesmen & Örmeci "
    "(2009), *PEIS* 23(2), 205-230."
)

sb = st.sidebar
sb.header("Système")
N = sb.number_input("Environnements `N`", 2, 4, 2)
eps = sb.slider("Variabilité `ε`", 0.0, 0.9, 0.8, 0.05)
mu = sb.slider("Taux de service `µ`", 0.01, 1.0, 0.11, 0.01)
h = sb.slider("Coût de stockage `h`", 0.001, 0.10, 0.01, 0.001, format="%.3f")
q = sb.slider("Transition `q`", 0.001, 0.20, 0.01, 0.001, format="%.3f")
a = sb.slider("Sensibilité au prix `a`", 0.5, 2.0, 1.0, 0.1)

sb.header("Extensions")
K = sb.slider("Mise en route `K`", 0.0, 2.0, 0.0, 0.05)
m_backlog = sb.number_input("Backlog maximum", 0, 20, 0)
b = sb.slider("Pénurie `b`", 0.0, 0.5, 0.0, 0.005, format="%.3f", disabled=m_backlog == 0)

sb.header("Numérique")
price_step = sb.select_slider("Pas des prix", [0.1, 0.05, 0.02, 0.01], 0.05)
m_stock = sb.number_input("Troncature `m`", 20, 200, 60, 10)
sb.caption(
    f"Balayage EDP : **{int(round(1 / a / price_step)) ** N}** vecteurs de prix. "
    "Le coût croît en $(1/\\text{pas})^N$."
)

params = (N, eps, mu, h, q, a, K, b if m_backlog else 0.0, m_backlog, m_stock, price_step)
run = st.button("Calculer", type="primary")

# ---------------------------------------------------------------------------
# Onglets « Modèle » et « Méthode » — disponibles sans calcul
# ---------------------------------------------------------------------------

t_res, t_stock, t_reco, t_model, t_method = st.tabs(
    ["Résultats", "Stocks et prix", "Recommandation", "Modèle", "Méthode numérique"]
)

pb_view = build_problem(*params)
gamma = pb_view.uniformization_constant

with t_model:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Demande et revenu")
        st.markdown(
            "La demande est un processus de Poisson **modulé par l'environnement** : son "
            "intensité dépend du régime `e` et du prix affiché."
        )
        st.latex(r"\lambda_e(p) = \Lambda_e\,(1 - a\,p),\qquad p \in [0,\ 1/a]")
        st.latex(r"r_e(\lambda) = \lambda\, p_e(\lambda) = \frac{\lambda}{a}\left(1 - \frac{\lambda}{\Lambda_e}\right)")
        st.caption(
            f"Instance courante : Λ = ({', '.join(f'{L:.2f}' for L in pb_view.Lam)}), "
            f"a = {a:g}, donc prix d'étranglement 1/a = {1 / a:.2f} et prix maximisant "
            f"le revenu p\\* = 1/(2a) = {1 / (2 * a):.2f}."
        )

        st.markdown("### Coût de détention")
        st.latex(r"h(x) = \begin{cases} h\,x & x \ge 0 \\ b\,|x| & x < 0 \end{cases}")
        st.caption(
            "La branche négative n'existe que si le backlog est autorisé. À `b = 0` et "
            "backlog nul, on retrouve l'hypothèse de ventes perdues du papier."
        )

        st.markdown("### Uniformisation")
        st.markdown(
            "Le processus en temps continu est ramené à une chaîne en temps discret en "
            "normalisant la somme de tous les taux à 1."
        )
        st.latex(r"\Gamma = \mu + \sum_e \Lambda_e + \sum_e \sum_{j \neq e} q_{ej}")
        st.caption(
            f"Ici Γ = {gamma:.4f}. Les taux et les coûts **par unité de temps** (`h`, `b`) "
            f"sont divisés par Γ ; les coûts **par événement** (`c`, `K`) ne le sont pas. "
            f"Le gain se reconvertit par g_réel = Γ · g_normalisé. Les profit gains, "
            "étant des ratios, sont invariants."
        )

    with c2:
        st.markdown("### Équation d'optimalité en gain moyen")
        st.latex(
            r"v(x,e) + g \;=\; -h(x) \;+\; \mu\,T_0 v(x,e) \;+\; T_2 v(x,e) \;+\; T_\lambda v(x,e)"
        )
        st.markdown("**Les trois opérateurs**, chacun correspondant à un événement :")
        st.latex(r"T_0 f(x,e) = \max\bigl\{\, f(x,e),\; f(x+1,e) - c \,\bigr\}")
        st.caption("Décision de production : produire une unité, ou non.")
        st.latex(
            r"T_2 f(x,e) = \sum_{j \neq e} q_{ej}\, f(x,j)"
            r"\;+\;\Bigl(\sum_{i \neq e} \Lambda_i + \sum_{i \neq e}\sum_{j \neq i} q_{ij}\Bigr) f(x,e)"
        )
        st.caption(
            "Transitions d'environnement, réelles et **fictives** — ces dernières "
            "complètent la masse pour que l'uniformisation reste valable."
        )
        st.latex(
            r"T_\lambda f(x,e) = \begin{cases}"
            r" r_e(\lambda) + \lambda f(x-1,e) + (\Lambda_e - \lambda) f(x,e) & x > x_{\min} \\"
            r" \Lambda_e f(x,e) & x = x_{\min} \end{cases}"
        )
        st.caption("Arrivée de demande ; à la borne basse la vente est perdue.")
        st.markdown(
            "La masse des trois opérateurs somme exactement à 1, ce qui en fait un "
            "opérateur de Bellman en gain moyen propre."
        )

    st.divider()
    c3, c4 = st.columns([1, 1])

    with c3:
        st.markdown("### Tarification dynamique (DP)")
        st.markdown("L'opérateur de demande devient une maximisation sur le taux :")
        st.latex(r"T f(x,e) = \max_{\lambda \in \mathcal{L}_e} T_\lambda f(x,e)")
        st.markdown(
            "En réécrivant $T_\\lambda f = \\Lambda_e f(x,e) + \\max_\\lambda"
            "\\bigl[r_e(\\lambda) - \\lambda\\,\\Delta f(x-1,e)\\bigr]$, la demande linéaire "
            "donne une **forme fermée** (Appendix E, éq. 11) :"
        )
        st.latex(
            r"p(x,e) = \operatorname{clip}\!\left(\frac{1 + a\,\Delta v(x-1,e)}{2a},\; 0,\; \frac{1}{a}\right)"
        )
        st.caption(
            "avec Δv(x,e) = v(x+1,e) − v(x,e). Aucune recherche numérique n'est nécessaire "
            "pour le prix : il se lit directement dans la fonction de valeur."
        )

    with c4:
        st.markdown("### Résultats structurels du papier")
        st.markdown(
            "- **Théorème 1** — la politique optimale de production est un base-stock "
            "dépendant de l'environnement : produire si et seulement si $x < s_e$.\n"
            "- **Théorème 2** — sous DP, $x < y \\Rightarrow p(x,e) \\ge p(y,e)$ : le prix "
            "décroît avec le stock.\n"
            "- **Théorème 3** — sous SP et EDP, les base-stocks sont ordonnés comme les "
            "environnements, $s_e \\le s_{e+1}$, sous la Condition 1.\n"
            "- **Corollaire 1** — $p(x,e) \\ge p^*$ si $x \\le s_e$, et $\\le p^*$ sinon : "
            "le prix franchit $p^*$ exactement au base-stock."
        )
        st.caption(
            f"Condition 1 vérifiée sur cette instance : "
            f"{'oui' if pb_view.satisfies_condition_1() else 'non'} "
            f"(processus de naissance-mort : {'oui' if pb_view.is_birth_death() else 'non'})."
        )

    if K > 0:
        st.divider()
        st.markdown("### Extension — coût de mise en route (`K > 0`)")
        st.markdown(
            "Hors périmètre du papier. L'état s'augmente d'un indicateur "
            "$\\sigma \\in \\{0,1\\}$ (serveur arrêté / en marche), et $T_0$ se scinde :"
        )
        cc1, cc2 = st.columns(2)
        cc1.latex(r"S f(x,e,\sigma) = \max\bigl\{ f(x,e,0),\; f(x,e,1) - K\,\mathbf{1}_{\sigma=0} \bigr\}")
        cc2.latex(r"P f(x,e,\sigma) = \begin{cases} f(x+1,e,1) - c & \sigma = 1 \\ f(x,e,0) & \sigma = 0\end{cases}")
        st.latex(r"v(x,e,\sigma) + g = S\bigl(-h(x) + \mu\,P v + T_2 v + T_\lambda v\bigr)(x,e,\sigma)")
        st.caption(
            "La production reste **unitaire au taux µ** : la contrainte de capacité, qui "
            "fonde tout le modèle, est préservée. À `K = 0`, `S` devient max{f(·,0), f(·,1)} "
            "et l'équation se réduit exactement à la précédente. La politique optimale est "
            "une bande d'hystérésis, non une politique (s,S) au sens de Chen & Simchi-Levi."
        )

with t_method:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Itération sur les politiques")
        st.markdown(
            "À prix fixé, la politique optimale est entièrement décrite par $N$ entiers — "
            "les base-stocks. L'espace des politiques est donc minuscule, et l'on alterne "
            "**évaluation exacte** et **amélioration**."
        )
        st.markdown("**Évaluation** — un système linéaire, résolu exactement :")
        st.latex(r"(I - P_\pi)\,v \;+\; g\,\mathbf{1} \;=\; r_\pi, \qquad v(x_{\min}, 1) = 0")
        st.markdown("**Amélioration** — les seuils se relisent dans la fonction de valeur :")
        st.latex(r"s_e \;=\; \min\{\, x : \Delta v(x,e) \le c \,\}")
        st.caption(
            "Par concavité de v (Lemme 1), l'ensemble {x : Δv > c} est un segment initial : "
            "la règle gloutonne *est* une politique base-stock. Pour DP, l'amélioration porte "
            "en plus sur le prix, via la forme fermée — l'itération est alors une méthode de "
            "Newton sur l'équation de Bellman."
        )

        st.markdown("### Évaluation stationnaire — contrôle croisé")
        st.markdown(
            "Une seconde méthode, indépendante, calcule le même gain sur le générateur en "
            "temps continu :"
        )
        st.latex(r"\pi Q = 0,\quad \textstyle\sum_{x,e} \pi(x,e) = 1")
        st.latex(
            r"g = \sum_{x,e} \pi(x,e)\Bigl[\lambda_e p_e \mathbf{1}_{x > x_{\min}}"
            r" - h(x) - \mu c\,\mathbf{1}_{x < s_e}\Bigr]"
        )
        st.caption(
            "Les deux méthodes concordent à 10⁻¹⁵. Elles valident du même coup "
            "l'uniformisation, la normalisation du gain et la construction du générateur — "
            "les trois endroits où une erreur de modélisation passerait inaperçue."
        )

    with c2:
        st.markdown("### Pourquoi pas la value iteration")
        st.markdown(
            "C'était l'approche initiale. Elle n'aboutissait pas sur le balayage de prix "
            "d'EDP. La cause est algorithmique, pas linguistique : avec $q$ petit devant "
            "les taux de demande, le mélange de la chaîne est très lent et le module de "
            "contraction en span vaut environ $1 - O(q)$, d'où quelques **milliers** "
            "d'itérations. L'itération sur les politiques en demande 5 à 8."
        )
        st.dataframe(pd.DataFrame([
            {"méthode": "value iteration", "balayage EDP à ε=0.8": "n'aboutit pas"},
            {"méthode": "itération sur les politiques", "balayage EDP à ε=0.8": "87 s"},
            {"méthode": "+ précalcul de la part libre", "balayage EDP à ε=0.8": "59 s"},
            {"méthode": "+ sonde de troncature", "balayage EDP à ε=0.8": "17,7 s"},
        ]), hide_index=True, use_container_width=True)

        st.markdown("### Les cinq stratégies")
        st.markdown(
            "Le prix et le base-stock sont **deux axes indépendants** : cela donne cinq "
            "politiques, et non trois."
        )
        st.table(pd.DataFrame(
            [["**Stock unique**", "S", "SB", "—"],
             ["**Stock par environnement**", "SP", "EDP", "DP"]],
            columns=["", "Prix unique", "Un prix par env.", "Un prix par état"],
        ).set_index(""))
        st.caption(
            "S et SB contraignent le base-stock à être commun : aucune value iteration ne "
            "produit une telle politique, on énumère donc les seuils et l'on évalue chacun "
            "exactement par la loi stationnaire. Tous les gains ci-contre sont rapportés à S."
        )
        st.latex(r"PG^{\phi,\phi'} = \frac{g^{\phi} - g^{\phi'}}{g^{\phi'}}")

# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------

if not run:
    with t_res:
        st.info(
            "Réglez les paramètres dans la barre latérale puis lancez le calcul. Les onglets "
            "**Modèle** et **Méthode numérique** sont consultables sans calcul."
        )
    st.stop()

with st.spinner("Résolution des programmes dynamiques…"):
    try:
        res = compute(params)
    except Exception as exc:  # noqa: BLE001
        with t_res:
            st.error(f"Le calcul a échoué : {exc}")
        st.stop()

env_names = [f"env {i + 1} (Λ={L:.2f})" for i, L in enumerate(res["Lam"])]

if res["mode"] == "setup":
    with t_res:
        st.warning(
            "Avec `K > 0`, l'état du serveur s'ajoute à l'état du système. Le balayage de "
            "prix en lot ne couvre pas ce modèle : seule DP est calculée, et la comparaison "
            "des cinq stratégies n'est pas disponible. Mettre `K = 0` pour y revenir."
        )
        m1, m2 = st.columns([1, 3])
        m1.metric("Profit moyen (DP)", f"{res['g']:.5f}")
        m2.caption(f"Convergence en {res['sweeps']} tours d'itération sur les politiques.")
        st.markdown("#### Bande d'hystérésis par environnement")
        st.dataframe(pd.DataFrame(
            [{"environnement": env_names[e], "redémarrer sous s_e": s, "produire jusqu'à S_e": S,
              "largeur de bande": S - s}
             for e, (s, S) in enumerate(res["sS"])]
        ), hide_index=True, use_container_width=True)
        st.caption(
            "On produit tant que le stock est sous S_e, et l'on ne redémarre qu'une fois "
            "retombé sous s_e. L'écart entre les deux seuils évite de payer K trop souvent."
        )
    st.stop()

gains = res["gains"]
pg = {k: 100 * profit_gain(gains[k], gains["s"]) for k in gains}
order = ["s", "sb", "sp", "edp", "dp"]

with t_res:
    cols = st.columns(4)
    for col, k in zip(cols, ("sb", "sp", "edp", "dp")):
        col.metric(k.upper(), f"{gains[k]:.5f}", f"{pg[k]:+.2f} % vs S")

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        ax.bar(order, [pg[k] for k in order], color="#4C78A8")
        ax.set_ylabel("gain sur S (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.dataframe(pd.DataFrame([
            {"stratégie": LABELS[k], "profit moyen": round(gains[k], 6),
             "gain sur S (%)": round(pg[k], 2), "base-stock": res["base_stock"][k]}
            for k in order
        ]), hide_index=True, use_container_width=True, height=215)

    st.caption(
        f"Uniformisation Γ = {res['gamma']:.4f}. Ordre structurel attendu "
        "g_s ≤ g_sb ≤ g_edp ≤ g_dp et g_s ≤ g_sp ≤ g_edp ≤ g_dp : "
        f"{'vérifié' if all(gains[u] <= gains[v] + 1e-9 for u, v in [('s','sb'),('sb','edp'),('edp','dp'),('s','sp'),('sp','edp')]) else '**violé**'}."
    )

with t_stock:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Base-stock par environnement")
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        width, idx = 0.15, np.arange(res["N"])
        for i, k in enumerate(order):
            ax.bar(idx + i * width, res["base_stock"][k], width, label=k.upper())
        ax.set_xticks(idx + 2 * width)
        ax.set_xticklabels(env_names, rotation=12, fontsize=7.5)
        ax.set_ylabel("base-stock s_e", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, ncol=5, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Théorème 3 : les seuils croissent avec la demande potentielle.")
    with right:
        st.markdown("#### Prix dynamique en fonction du stock")
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        s_bar = max(res["base_stock"]["dp"])
        rec = res["states"] <= s_bar
        for e in range(res["N"]):
            ax.plot(res["states"][rec], res["dp_prices"][rec, e], marker=".", ms=4,
                    label=f"DP {env_names[e]}")
        for e, p in enumerate(res["prices"]["edp"]):
            ax.axhline(p, ls="--", lw=1, alpha=0.65, color=f"C{e}")
        ax.axhline(1 / (2 * a), ls=":", lw=1, color="grey")
        ax.annotate("p*", (res["states"][rec][0], 1 / (2 * a)), fontsize=8,
                    color="grey", va="bottom")
        ax.set_xlabel("niveau de stock x", fontsize=8)
        ax.set_ylabel("prix", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Traits pleins : prix DP, décroissants en stock (Théorème 2). Pointillés : prix "
            "EDP, constants par environnement. Pointillé gris : $p^* = 1/(2a)$, que le prix "
            "DP franchit au base-stock (Corollaire 1). Seuls les états récurrents sont tracés."
        )

with t_reco:
    gain_edp = pg["edp"]
    extra_dp = 100 * profit_gain(gains["dp"], gains["edp"])

    # Le seuil de 3 % n'est pas arbitraire : sur l'ensemble du balayage de capacité,
    # l'écart entre DP et EDP ne dépasse jamais 3,3 % (§5.3.1 du papier, reproduit
    # dans `notebooks/02_sensitivity.ipynb`). Un seuil plus bas ferait recommander DP
    # dans des régimes où l'étude conclut précisément l'inverse.
    if gain_edp < 2.0:
        verdict, strategy = "success", "**S — un prix unique**"
        why = (
            f"Différencier les prix ne rapporte que {gain_edp:.1f} % ici. En dessous de "
            "~2 %, le gain ne couvre pas le coût d'exploitation d'une politique plus "
            "complexe, ni la friction créée côté client."
        )
    elif extra_dp < 3.0:
        verdict, strategy = "info", "**EDP — un prix par environnement**"
        why = (
            f"EDP rapporte {gain_edp:.1f} % sur un prix unique, et DP n'y ajoute que "
            f"{extra_dp:.1f} %. Passer à une tarification pilotée par le stock exige un "
            "suivi en temps réel et multiplie les changements de prix, pour un gain marginal."
        )
    else:
        verdict, strategy = "warning", "**EDP**, mais c'est ici que DP mérite un examen"
        why = (
            f"DP dépasse EDP de {extra_dp:.1f} %, soit le haut de la fourchette observée sur "
            "l'ensemble des régimes — l'écart ne dépasse jamais 3,3 %. C'est le domaine où le "
            "niveau de stock porte le plus d'information, et le seul où le pilotage en temps "
            "réel puisse se discuter. À mettre en balance avec son coût de mise en œuvre, que "
            "le modèle ne facture pas."
        )

    getattr(st, verdict)(f"Stratégie recommandée : {strategy}")
    st.markdown(why)

    notes = []
    if eps < 0.2:
        notes.append(f"**Demande peu variable** (`ε = {eps:.2f}`) — déterminant principal. "
                     "Sans fluctuation, aucune tarification adaptative ne se justifie.")
    if q > 0.05:
        notes.append(f"**Environnement instable** (`q = {q:.3f}`) — les régimes se succèdent "
                     "trop vite pour qu'une politique de prix produise ses effets.")
    if mu > 0.6:
        notes.append(f"**Capacité abondante** (`µ = {mu:.2f}`) — le réapprovisionnement "
                     "absorbe seul les fluctuations.")
    if mu < 0.05:
        notes.append(f"**Capacité très contrainte** (`µ = {mu:.2f}`) — le stock reste quasi "
                     "nul et l'on vend en permanence au prix fort.")
    if h > 0.05:
        notes.append(f"**Stockage coûteux** (`h = {h:.3f}`) — la tarification gagne en "
                     "valeur : elle sert à écouler avant que la facture ne tombe.")
    for n in notes:
        st.markdown(f"- {n}")

    st.divider()
    st.caption(
        "**Limite du modèle.** Ni le coût de mise en œuvre d'une politique, ni la réaction "
        "des clients aux variations de prix ne sont facturés. Les gains affichés pour DP sont "
        "donc des majorants, et son avantage réel sur EDP est plus étroit que ces chiffres."
    )
