"""Tableau de bord interactif DynamicPricer (français / English).

Lancement, depuis la racine du repo (après `pip install -e .[dashboard]`) :

    streamlit run dashboard/app.py
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from dynamicpricer import native
from dynamicpricer.mdp import solve_mdp
from dynamicpricer.metrics import profit_gain
from dynamicpricer.params import Problem
from dynamicpricer.strategies import solve_all
from dynamicpricer.ui_texts import LANGUAGES, translate

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


def default_language() -> str:
    """Langue du navigateur si elle est connue (français sinon, comportement d'origine)."""
    try:
        locale = (st.context.locale or "").lower()
    except Exception:  # noqa: BLE001 — API absente des anciennes versions
        return "fr"
    return "en" if locale and not locale.startswith("fr") else "fr"


st.session_state.setdefault("lang_choice", default_language())
lang = st.sidebar.radio(
    translate("fr", "sb.language"), list(LANGUAGES), format_func=LANGUAGES.get,
    horizontal=True, key="lang_choice",
)


def T(key: str, **values) -> str:
    """Texte de l'interface dans la langue choisie."""
    return translate(lang, key, **values)


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
st.caption(T("app.caption"))

sb = st.sidebar
sb.header(T("sb.system"))
N = sb.number_input(T("sb.N"), 2, 4, 2, key="w_N")
eps = sb.slider(T("sb.eps"), 0.0, 0.9, 0.8, 0.05, key="w_eps")
mu = sb.slider(T("sb.mu"), 0.01, 1.0, 0.11, 0.01, key="w_mu")
h = sb.slider(T("sb.h"), 0.001, 0.10, 0.01, 0.001, format="%.3f", key="w_h")
q = sb.slider(T("sb.q"), 0.001, 0.20, 0.01, 0.001, format="%.3f", key="w_q")
a = sb.slider(T("sb.a"), 0.5, 2.0, 1.0, 0.1, key="w_a")

sb.header(T("sb.extensions"))
K = sb.slider(T("sb.K"), 0.0, 2.0, 0.0, 0.05, key="w_K")
m_backlog = sb.number_input(T("sb.m_backlog"), 0, 20, 0, key="w_m_backlog")
b = sb.slider(
    T("sb.b"), 0.0, 0.5, 0.0, 0.005, format="%.3f", disabled=m_backlog == 0, key="w_b"
)

sb.header(T("sb.numerics"))
price_step = sb.select_slider(
    T("sb.price_step"), [0.1, 0.05, 0.02, 0.01], 0.05, key="w_price_step"
)
m_stock = sb.number_input(T("sb.m_stock"), 20, 200, 60, 10, key="w_m_stock")
ENGINE = T("sb.engine_native") if native.AVAILABLE else T("sb.engine_python")
sb.caption(T("sb.sweep", n=int(round(1 / a / price_step)) ** N, engine=ENGINE))

params = (N, eps, mu, h, q, a, K, b if m_backlog else 0.0, m_backlog, m_stock, price_step)
# Un changement de langue relance le script sans que le bouton soit cliqué : on réaffiche
# alors les résultats déjà calculés, à condition que les paramètres n'aient pas changé.
language_changed = st.session_state.get("_last_lang") not in (None, lang)
st.session_state["_last_lang"] = lang
run = st.button(T("btn.run"), type="primary") or (
    language_changed and st.session_state.get("_results_params") == params
)

# ---------------------------------------------------------------------------
# Onglets « Modèle » et « Méthode » — disponibles sans calcul
# ---------------------------------------------------------------------------

t_res, t_stock, t_reco, t_model, t_method = st.tabs(
    [T("tab.results"), T("tab.stock"), T("tab.reco"), T("tab.model"), T("tab.method")]
)

pb_view = build_problem(*params)
gamma = pb_view.uniformization_constant

with t_model:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(T("m.h_demand"))
        st.markdown(T("m.demand_intro"))
        st.latex(r"\lambda_e(p) = \Lambda_e\,(1 - a\,p),\qquad p \in [0,\ 1/a]")
        st.latex(r"r_e(\lambda) = \lambda\, p_e(\lambda) = \frac{\lambda}{a}\left(1 - \frac{\lambda}{\Lambda_e}\right)")
        st.caption(T("m.instance", Lam=", ".join(f"{L:.2f}" for L in pb_view.Lam), a=f"{a:g}",
                     choke=f"{1 / a:.2f}", pstar=f"{1 / (2 * a):.2f}"))

        st.markdown(T("m.h_holding"))
        st.latex(r"h(x) = \begin{cases} h\,x & x \ge 0 \\ b\,|x| & x < 0 \end{cases}")
        st.caption(T("m.holding"))

        st.markdown(T("m.h_unif"))
        st.markdown(T("m.unif_intro"))
        st.latex(r"\Gamma = \mu + \sum_e \Lambda_e + \sum_e \sum_{j \neq e} q_{ej}")
        st.caption(T("m.unif", gamma=f"{gamma:.4f}"))

    with c2:
        st.markdown(T("m.h_opt"))
        st.latex(
            r"v(x,e) + g \;=\; -h(x) \;+\; \mu\,T_0 v(x,e) \;+\; T_2 v(x,e) \;+\; T_\lambda v(x,e)"
        )
        st.markdown(T("m.ops_intro"))
        st.latex(r"T_0 f(x,e) = \max\bigl\{\, f(x,e),\; f(x+1,e) - c \,\bigr\}")
        st.caption(T("m.T0"))
        st.latex(
            r"T_2 f(x,e) = \sum_{j \neq e} q_{ej}\, f(x,j)"
            r"\;+\;\Bigl(\sum_{i \neq e} \Lambda_i + \sum_{i \neq e}\sum_{j \neq i} q_{ij}\Bigr) f(x,e)"
        )
        st.caption(T("m.T2"))
        st.latex(
            r"T_\lambda f(x,e) = \begin{cases}"
            r" r_e(\lambda) + \lambda f(x-1,e) + (\Lambda_e - \lambda) f(x,e) & x > x_{\min} \\"
            r" \Lambda_e f(x,e) & x = x_{\min} \end{cases}"
        )
        st.caption(T("m.Tl"))
        st.markdown(T("m.ops_mass"))

    st.divider()
    c3, c4 = st.columns([1, 1])

    with c3:
        st.markdown(T("m.h_dp"))
        st.markdown(T("m.dp_intro"))
        st.latex(r"T f(x,e) = \max_{\lambda \in \mathcal{L}_e} T_\lambda f(x,e)")
        st.markdown(T("m.dp_closed"))
        st.latex(
            r"p(x,e) = \operatorname{clip}\!\left(\frac{1 + a\,\Delta v(x-1,e)}{2a},\; 0,\; \frac{1}{a}\right)"
        )
        st.caption(T("m.dp_note"))

    with c4:
        st.markdown(T("m.h_struct"))
        st.markdown(T("m.struct"))
        st.caption(T("m.cond1", c1=T("yes") if pb_view.satisfies_condition_1() else T("no"),
                     bd=T("yes") if pb_view.is_birth_death() else T("no")))

    if K > 0:
        st.divider()
        st.markdown(T("m.h_setup"))
        st.markdown(T("m.setup_intro"))
        cc1, cc2 = st.columns(2)
        cc1.latex(r"S f(x,e,\sigma) = \max\bigl\{ f(x,e,0),\; f(x,e,1) - K\,\mathbf{1}_{\sigma=0} \bigr\}")
        cc2.latex(r"P f(x,e,\sigma) = \begin{cases} f(x+1,e,1) - c & \sigma = 1 \\ f(x,e,0) & \sigma = 0\end{cases}")
        st.latex(r"v(x,e,\sigma) + g = S\bigl(-h(x) + \mu\,P v + T_2 v + T_\lambda v\bigr)(x,e,\sigma)")
        st.caption(T("m.setup_note"))

with t_method:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(T("me.h_pi"))
        st.markdown(T("me.pi_intro"))
        st.markdown(T("me.eval"))
        st.latex(r"(I - P_\pi)\,v \;+\; g\,\mathbf{1} \;=\; r_\pi, \qquad v(x_{\min}, 1) = 0")
        st.markdown(T("me.improve"))
        st.latex(r"s_e \;=\; \min\{\, x : \Delta v(x,e) \le c \,\}")
        st.caption(T("me.pi_note"))

        st.markdown(T("me.h_stat"))
        st.markdown(T("me.stat_intro"))
        st.latex(r"\pi Q = 0,\quad \textstyle\sum_{x,e} \pi(x,e) = 1")
        st.latex(
            r"g = \sum_{x,e} \pi(x,e)\Bigl[\lambda_e p_e \mathbf{1}_{x > x_{\min}}"
            r" - h(x) - \mu c\,\mathbf{1}_{x < s_e}\Bigr]"
        )
        st.caption(T("me.stat_note"))

    with c2:
        st.markdown(T("me.h_vi"))
        st.markdown(T("me.vi"))
        st.dataframe(pd.DataFrame([
            {T("me.col_method"): T("me.row_vi"), T("me.col_sweep"): T("me.row_vi_t")},
            {T("me.col_method"): T("me.row_pi"), T("me.col_sweep"): T("me.t_pi")},
            {T("me.col_method"): T("me.row_pre"), T("me.col_sweep"): T("me.t_pre")},
            {T("me.col_method"): T("me.row_probe"), T("me.col_sweep"): T("me.t_probe")},
        ]), hide_index=True, width="stretch")

        st.markdown(T("me.h_five"))
        st.markdown(T("me.five_intro"))
        st.table(pd.DataFrame(
            [[T("me.five_row1"), "S", "SB", "—"],
             [T("me.five_row2"), "SP", "EDP", "DP"]],
            columns=["", T("me.five_col1"), T("me.five_col2"), T("me.five_col3")],
        ).set_index(""))
        st.caption(T("me.five_note"))
        st.latex(r"PG^{\phi,\phi'} = \frac{g^{\phi} - g^{\phi'}}{g^{\phi'}}")

# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------

if not run:
    with t_res:
        st.info(T("r.info"))
    st.stop()

_t0 = perf_counter()
with st.spinner(T("r.spinner")):
    try:
        res = compute(params)
    except Exception as exc:  # noqa: BLE001
        with t_res:
            st.error(T("r.error", exc=exc))
        st.stop()

_elapsed = perf_counter() - _t0
st.session_state["_results_params"] = params
env_names = [T("r.env", i=i + 1, L=f"{L:.2f}") for i, L in enumerate(res["Lam"])]

if res["mode"] == "setup":
    with t_res:
        st.warning(T("r.setup_warning"))
        m1, m2 = st.columns([1, 3])
        m1.metric(T("r.avg_profit"), f"{res['g']:.5f}")
        m2.caption(T("r.converged", n=res["sweeps"]))
        st.markdown(T("r.h_band"))
        st.dataframe(pd.DataFrame(
            [{T("r.col_env"): env_names[e], T("r.col_restart"): s, T("r.col_produce"): S,
              T("r.col_width"): S - s}
             for e, (s, S) in enumerate(res["sS"])]
        ), hide_index=True, width="stretch")
        st.caption(T("r.band_note"))
    st.stop()

gains = res["gains"]
pg = {k: 100 * profit_gain(gains[k], gains["s"]) for k in gains}
order = ["s", "sb", "sp", "edp", "dp"]

with t_res:
    cols = st.columns(4)
    for col, k in zip(cols, ("sb", "sp", "edp", "dp")):
        col.metric(k.upper(), f"{gains[k]:.5f}", T("r.vs_s", pg=f"{pg[k]:+.2f}"))

    left, right = st.columns([1, 1])
    with left:
        fig, ax = plt.subplots(figsize=(5.2, 2.6))
        ax.bar(order, [pg[k] for k in order], color="#4C78A8")
        ax.set_ylabel(T("r.ylabel_gain"), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.dataframe(pd.DataFrame([
            {T("r.col_strategy"): T(f"lab.{k}"), T("r.col_profit"): round(gains[k], 6),
             T("r.col_gain"): round(pg[k], 2), T("r.col_bs"): res["base_stock"][k]}
            for k in order
        ]), hide_index=True, width="stretch", height=215)

    ok = all(gains[u] <= gains[v] + 1e-9 for u, v in
             [("s", "sb"), ("sb", "edp"), ("edp", "dp"), ("s", "sp"), ("sp", "edp")])
    st.caption(T("r.footer", t=f"{_elapsed:.2f}", engine=ENGINE, gamma=f"{res['gamma']:.4f}",
                 ok=T("r.ok") if ok else T("r.violated")))

with t_stock:
    left, right = st.columns([1, 1])
    with left:
        st.markdown(T("s.h_bs"))
        fig, ax = plt.subplots(figsize=(5.2, 3.1))
        width, idx = 0.15, np.arange(res["N"])
        for i, k in enumerate(order):
            ax.bar(idx + i * width, res["base_stock"][k], width, label=k.upper())
        ax.set_xticks(idx + 2 * width)
        ax.set_xticklabels(env_names, rotation=12, fontsize=7.5)
        ax.set_ylabel(T("s.ylabel_bs"), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, ncol=5, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(T("s.thm3"))
    with right:
        st.markdown(T("s.h_price"))
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
        ax.set_xlabel(T("s.xlabel"), fontsize=8)
        ax.set_ylabel(T("s.ylabel_price"), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(T("s.price_note"))

with t_reco:
    gain_edp = pg["edp"]
    extra_dp = 100 * profit_gain(gains["dp"], gains["edp"])

    # Le seuil de 3 % n'est pas arbitraire : sur l'ensemble du balayage de capacité,
    # l'écart entre DP et EDP ne dépasse jamais 3,3 % (§5.3.1 du papier, reproduit
    # dans `notebooks/02_sensitivity.ipynb`). Un seuil plus bas ferait recommander DP
    # dans des régimes où l'étude conclut précisément l'inverse.
    if gain_edp < 2.0:
        verdict, strategy = "success", T("rc.s_strategy")
        why = T("rc.s_why", g=f"{gain_edp:.1f}")
    elif extra_dp < 3.0:
        verdict, strategy = "info", T("rc.edp_strategy")
        why = T("rc.edp_why", g=f"{gain_edp:.1f}", x=f"{extra_dp:.1f}")
    else:
        verdict, strategy = "warning", T("rc.dp_strategy")
        why = T("rc.dp_why", x=f"{extra_dp:.1f}")

    getattr(st, verdict)(T("rc.verdict", s=strategy))
    st.markdown(why)

    notes = []
    if eps < 0.2:
        notes.append(T("rc.note_eps", eps=f"{eps:.2f}"))
    if q > 0.05:
        notes.append(T("rc.note_q", q=f"{q:.3f}"))
    if mu > 0.6:
        notes.append(T("rc.note_mu_hi", mu=f"{mu:.2f}"))
    if mu < 0.05:
        notes.append(T("rc.note_mu_lo", mu=f"{mu:.2f}"))
    if h > 0.05:
        notes.append(T("rc.note_h", h=f"{h:.3f}"))
    for n in notes:
        st.markdown(f"- {n}")

    st.divider()
    st.caption(T("rc.limit"))
