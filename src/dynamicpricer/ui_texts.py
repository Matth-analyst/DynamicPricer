"""Textes du tableau de bord (`dashboard/app.py`), en français et en anglais.

Chaque entrée est un triplet `(clé, français, anglais)`. Les deux langues partagent donc
les mêmes clés par construction. Les variables s'écrivent `<<nom>>` (et non `{nom}`, qui
entrerait en conflit avec les accolades du LaTeX) ; elles sont substituées par `t()`.

Les formules LaTeX qui ne contiennent aucun mot restent dans `app.py`.
"""

LANGUAGES = {"fr": "Français", "en": "English"}

_ENTRIES = [
    # --- En-tête ------------------------------------------------------------------------
    ("app.caption",
     "Tarification et production d'un système make-to-stock à capacité limitée sous demande "
     "modulée par une chaîne de Markov. Modèle : Gayon, Talay-Değirmenci, Karaesmen & Örmeci "
     "(2009), *PEIS* 23(2), 205-230.",
     "Pricing and production of a capacity-constrained make-to-stock system under "
     "Markov-modulated demand. Model: Gayon, Talay-Değirmenci, Karaesmen & Örmeci (2009), "
     "*PEIS* 23(2), 205-230."),

    # --- Barre latérale -----------------------------------------------------------------
    ("sb.language", "Langue / Language", "Langue / Language"),
    ("sb.system", "Système", "System"),
    ("sb.N", "Environnements `N`", "Environments `N`"),
    ("sb.eps", "Variabilité `ε`", "Variability `ε`"),
    ("sb.mu", "Taux de service `µ`", "Service rate `µ`"),
    ("sb.h", "Coût de stockage `h`", "Holding cost `h`"),
    ("sb.q", "Transition `q`", "Transition rate `q`"),
    ("sb.a", "Sensibilité au prix `a`", "Price sensitivity `a`"),
    ("sb.extensions", "Extensions", "Extensions"),
    ("sb.K", "Mise en route `K`", "Setup cost `K`"),
    ("sb.m_backlog", "Backlog maximum", "Maximum backlog"),
    ("sb.b", "Pénurie `b`", "Shortage cost `b`"),
    ("sb.numerics", "Numérique", "Numerics"),
    ("sb.price_step", "Pas des prix", "Price step"),
    ("sb.m_stock", "Troncature `m`", "Truncation `m`"),
    ("sb.engine_native", "Rust (natif)", "Rust (native)"),
    ("sb.engine_python", "Python", "Python"),
    ("sb.sweep",
     "Balayage EDP : **<<n>>** vecteurs de prix. Le coût croît en $(1/\\text{pas})^N$.  \n"
     "Moteur de balayage : **<<engine>>**.",
     "EDP sweep: **<<n>>** price vectors. Cost grows as $(1/\\text{step})^N$.  \n"
     "Sweep engine: **<<engine>>**."),
    ("btn.run", "Calculer", "Compute"),

    # --- Onglets ------------------------------------------------------------------------
    ("tab.results", "Résultats", "Results"),
    ("tab.stock", "Stocks et prix", "Stock and prices"),
    ("tab.reco", "Recommandation", "Recommendation"),
    ("tab.model", "Modèle", "Model"),
    ("tab.method", "Méthode numérique", "Numerical method"),

    # --- Onglet Modèle ------------------------------------------------------------------
    ("m.h_demand", "### Demande et revenu", "### Demand and revenue"),
    ("m.demand_intro",
     "La demande est un processus de Poisson **modulé par l'environnement** : son "
     "intensité dépend du régime `e` et du prix affiché.",
     "Demand is a Poisson process **modulated by the environment**: its intensity depends "
     "on the regime `e` and on the posted price."),
    ("m.instance",
     "Instance courante : Λ = (<<Lam>>), a = <<a>>, donc prix d'étranglement 1/a = "
     "<<choke>> et prix maximisant le revenu p\\* = 1/(2a) = <<pstar>>.",
     "Current instance: Λ = (<<Lam>>), a = <<a>>, so choke price 1/a = <<choke>> and "
     "revenue-maximizing price p\\* = 1/(2a) = <<pstar>>."),
    ("m.h_holding", "### Coût de détention", "### Holding cost"),
    ("m.holding",
     "La branche négative n'existe que si le backlog est autorisé. À `b = 0` et "
     "backlog nul, on retrouve l'hypothèse de ventes perdues du papier.",
     "The negative branch exists only if backlog is allowed. With `b = 0` and zero "
     "backlog, we recover the paper's lost-sales assumption."),
    ("m.h_unif", "### Uniformisation", "### Uniformization"),
    ("m.unif_intro",
     "Le processus en temps continu est ramené à une chaîne en temps discret en "
     "normalisant la somme de tous les taux à 1.",
     "The continuous-time process is reduced to a discrete-time chain by normalizing the "
     "sum of all rates to 1."),
    ("m.unif",
     "Ici Γ = <<gamma>>. Les taux et les coûts **par unité de temps** (`h`, `b`) "
     "sont divisés par Γ ; les coûts **par événement** (`c`, `K`) ne le sont pas. "
     "Le gain se reconvertit par g_réel = Γ · g_normalisé. Les profit gains, "
     "étant des ratios, sont invariants.",
     "Here Γ = <<gamma>>. Rates and costs **per unit of time** (`h`, `b`) are divided by "
     "Γ; costs **per event** (`c`, `K`) are not. The gain converts back as "
     "g_real = Γ · g_normalized. Profit gains, being ratios, are invariant."),
    ("m.h_opt", "### Équation d'optimalité en gain moyen", "### Average-profit optimality equation"),
    ("m.ops_intro",
     "**Les trois opérateurs**, chacun correspondant à un événement :",
     "**The three operators**, each corresponding to an event:"),
    ("m.T0", "Décision de production : produire une unité, ou non.",
     "Production decision: produce one unit, or not."),
    ("m.T2",
     "Transitions d'environnement, réelles et **fictives** — ces dernières "
     "complètent la masse pour que l'uniformisation reste valable.",
     "Environment transitions, real and **fictitious** — the latter complete the mass so "
     "that uniformization remains valid."),
    ("m.Tl", "Arrivée de demande ; à la borne basse la vente est perdue.",
     "Demand arrival; at the lower bound the sale is lost."),
    ("m.ops_mass",
     "La masse des trois opérateurs somme exactement à 1, ce qui en fait un "
     "opérateur de Bellman en gain moyen propre.",
     "The mass of the three operators sums exactly to 1, which makes this a proper "
     "average-profit Bellman operator."),
    ("m.h_dp", "### Tarification dynamique (DP)", "### Dynamic pricing (DP)"),
    ("m.dp_intro", "L'opérateur de demande devient une maximisation sur le taux :",
     "The demand operator becomes a maximization over the rate:"),
    ("m.dp_closed",
     "En réécrivant $T_\\lambda f = \\Lambda_e f(x,e) + \\max_\\lambda"
     "\\bigl[r_e(\\lambda) - \\lambda\\,\\Delta f(x-1,e)\\bigr]$, la demande linéaire "
     "donne une **forme fermée** (Appendix E, éq. 11) :",
     "Rewriting $T_\\lambda f = \\Lambda_e f(x,e) + \\max_\\lambda"
     "\\bigl[r_e(\\lambda) - \\lambda\\,\\Delta f(x-1,e)\\bigr]$, linear demand yields a "
     "**closed form** (Appendix E, eq. 11):"),
    ("m.dp_note",
     "avec Δv(x,e) = v(x+1,e) − v(x,e). Aucune recherche numérique n'est nécessaire "
     "pour le prix : il se lit directement dans la fonction de valeur.",
     "with Δv(x,e) = v(x+1,e) − v(x,e). No numerical search is needed for the price: it is "
     "read directly from the value function."),
    ("m.h_struct", "### Résultats structurels du papier", "### Structural results of the paper"),
    ("m.struct",
     "- **Théorème 1** — la politique optimale de production est un base-stock "
     "dépendant de l'environnement : produire si et seulement si $x < s_e$.\n"
     "- **Théorème 2** — sous DP, $x < y \\Rightarrow p(x,e) \\ge p(y,e)$ : le prix "
     "décroît avec le stock.\n"
     "- **Théorème 3** — sous SP et EDP, les base-stocks sont ordonnés comme les "
     "environnements, $s_e \\le s_{e+1}$, sous la Condition 1.\n"
     "- **Corollaire 1** — $p(x,e) \\ge p^*$ si $x \\le s_e$, et $\\le p^*$ sinon : "
     "le prix franchit $p^*$ exactement au base-stock.",
     "- **Theorem 1** — the optimal production policy is an environment-dependent "
     "base-stock: produce if and only if $x < s_e$.\n"
     "- **Theorem 2** — under DP, $x < y \\Rightarrow p(x,e) \\ge p(y,e)$: the price "
     "decreases with stock.\n"
     "- **Theorem 3** — under SP and EDP, base-stocks are ordered like the "
     "environments, $s_e \\le s_{e+1}$, under Condition 1.\n"
     "- **Corollary 1** — $p(x,e) \\ge p^*$ if $x \\le s_e$, and $\\le p^*$ otherwise: "
     "the price crosses $p^*$ exactly at the base-stock."),
    ("m.cond1",
     "Condition 1 vérifiée sur cette instance : <<c1>> "
     "(processus de naissance-mort : <<bd>>).",
     "Condition 1 satisfied on this instance: <<c1>> (birth-death process: <<bd>>)."),
    ("yes", "oui", "yes"),
    ("no", "non", "no"),
    ("m.h_setup", "### Extension — coût de mise en route (`K > 0`)",
     "### Extension — setup cost (`K > 0`)"),
    ("m.setup_intro",
     "Hors périmètre du papier. L'état s'augmente d'un indicateur "
     "$\\sigma \\in \\{0,1\\}$ (serveur arrêté / en marche), et $T_0$ se scinde :",
     "Outside the paper's scope. The state is augmented with an indicator "
     "$\\sigma \\in \\{0,1\\}$ (server off / on), and $T_0$ splits into:"),
    ("m.setup_note",
     "La production reste **unitaire au taux µ** : la contrainte de capacité, qui "
     "fonde tout le modèle, est préservée. À `K = 0`, `S` devient max{f(·,0), f(·,1)} "
     "et l'équation se réduit exactement à la précédente. La politique optimale est "
     "une bande d'hystérésis, non une politique (s,S) au sens de Chen & Simchi-Levi.",
     "Production remains **unit-rate at rate µ**: the capacity constraint, on which the "
     "whole model rests, is preserved. At `K = 0`, `S` becomes max{f(·,0), f(·,1)} and "
     "the equation reduces exactly to the previous one. The optimal policy is a "
     "hysteresis band, not an (s,S) policy in the sense of Chen & Simchi-Levi."),

    # --- Onglet Méthode numérique -------------------------------------------------------
    ("me.h_pi", "### Itération sur les politiques", "### Policy iteration"),
    ("me.pi_intro",
     "À prix fixé, la politique optimale est entièrement décrite par $N$ entiers — "
     "les base-stocks. L'espace des politiques est donc minuscule, et l'on alterne "
     "**évaluation exacte** et **amélioration**.",
     "At a fixed price, the optimal policy is fully described by $N$ integers — the "
     "base-stocks. The policy space is therefore tiny, and we alternate **exact "
     "evaluation** and **improvement**."),
    ("me.eval", "**Évaluation** — un système linéaire, résolu exactement :",
     "**Evaluation** — a linear system, solved exactly:"),
    ("me.improve", "**Amélioration** — les seuils se relisent dans la fonction de valeur :",
     "**Improvement** — thresholds are read back from the value function:"),
    ("me.pi_note",
     "Par concavité de v (Lemme 1), l'ensemble {x : Δv > c} est un segment initial : "
     "la règle gloutonne *est* une politique base-stock. Pour DP, l'amélioration porte "
     "en plus sur le prix, via la forme fermée — l'itération est alors une méthode de "
     "Newton sur l'équation de Bellman.",
     "By concavity of v (Lemma 1), the set {x : Δv > c} is an initial segment: the greedy "
     "rule *is* a base-stock policy. For DP, the improvement also acts on the price, via "
     "the closed form — the iteration is then a Newton method on the Bellman equation."),
    ("me.h_stat", "### Évaluation stationnaire — contrôle croisé",
     "### Stationary evaluation — cross-check"),
    ("me.stat_intro",
     "Une seconde méthode, indépendante, calcule le même gain sur le générateur en "
     "temps continu :",
     "A second, independent method computes the same gain on the continuous-time "
     "generator:"),
    ("me.stat_note",
     "Les deux méthodes concordent à 10⁻¹⁵. Elles valident du même coup "
     "l'uniformisation, la normalisation du gain et la construction du générateur — "
     "les trois endroits où une erreur de modélisation passerait inaperçue.",
     "The two methods agree to 10⁻¹⁵. Together they validate the uniformization, the "
     "gain normalization and the construction of the generator — the three places where "
     "a modeling error would go unnoticed."),
    ("me.h_vi", "### Pourquoi pas la value iteration", "### Why not value iteration"),
    ("me.vi",
     "C'était l'approche initiale. Elle n'aboutissait pas sur le balayage de prix "
     "d'EDP. La cause est algorithmique, pas linguistique : avec $q$ petit devant "
     "les taux de demande, le mélange de la chaîne est très lent et le module de "
     "contraction en span vaut environ $1 - O(q)$, d'où quelques **milliers** "
     "d'itérations. L'itération sur les politiques en demande 5 à 8.",
     "This was the initial approach. It did not converge on the EDP price sweep. The "
     "cause is algorithmic, not a matter of programming language: with $q$ small "
     "relative to the demand rates, the chain mixes very slowly and the span contraction "
     "modulus is about $1 - O(q)$, hence a few **thousand** iterations. Policy iteration "
     "needs 5 to 8."),
    ("me.col_method", "méthode", "method"),
    ("me.col_sweep", "balayage EDP à ε=0.8", "EDP sweep at ε=0.8"),
    ("me.row_vi", "value iteration", "value iteration"),
    ("me.row_vi_t", "n'aboutit pas", "does not converge"),
    ("me.row_pi", "itération sur les politiques", "policy iteration"),
    ("me.row_pre", "+ précalcul de la part libre", "+ precomputation of the free part"),
    ("me.row_probe", "+ sonde de troncature", "+ truncation probe"),
    ("me.t_pi", "87 s", "87 s"),
    ("me.t_pre", "59 s", "59 s"),
    ("me.t_probe", "17,7 s", "17.7 s"),
    ("me.h_five", "### Les cinq stratégies", "### The five strategies"),
    ("me.five_intro",
     "Le prix et le base-stock sont **deux axes indépendants** : cela donne cinq "
     "politiques, et non trois.",
     "Price and base-stock are **two independent axes**: this gives five policies, not "
     "three."),
    ("me.five_row1", "**Stock unique**", "**Single stock**"),
    ("me.five_row2", "**Stock par environnement**", "**Stock per environment**"),
    ("me.five_col1", "Prix unique", "Single price"),
    ("me.five_col2", "Un prix par env.", "One price per env."),
    ("me.five_col3", "Un prix par état", "One price per state"),
    ("me.five_note",
     "S et SB contraignent le base-stock à être commun : aucune value iteration ne "
     "produit une telle politique, on énumère donc les seuils et l'on évalue chacun "
     "exactement par la loi stationnaire. Tous les gains ci-contre sont rapportés à S.",
     "S and SB constrain the base-stock to be common: no value iteration produces such a "
     "policy, so we enumerate the thresholds and evaluate each exactly through the "
     "stationary distribution. All gains shown here are expressed relative to S."),

    # --- Stratégies ---------------------------------------------------------------------
    ("lab.s", "S — prix unique, stock unique", "S — single price, single stock"),
    ("lab.sb", "SB — un prix par environnement, stock unique",
     "SB — one price per environment, single stock"),
    ("lab.sp", "SP — prix unique, stock par environnement",
     "SP — single price, stock per environment"),
    ("lab.edp", "EDP — un prix et un stock par environnement",
     "EDP — one price and one stock per environment"),
    ("lab.dp", "DP — un prix par état, stock par environnement",
     "DP — one price per state, stock per environment"),

    # --- Résultats ----------------------------------------------------------------------
    ("r.info",
     "Réglez les paramètres dans la barre latérale puis lancez le calcul. Les onglets "
     "**Modèle** et **Méthode numérique** sont consultables sans calcul.",
     "Set the parameters in the sidebar, then run the computation. The **Model** and "
     "**Numerical method** tabs can be consulted without running anything."),
    ("r.spinner", "Résolution des programmes dynamiques…", "Solving the dynamic programs…"),
    ("r.error", "Le calcul a échoué : <<exc>>", "The computation failed: <<exc>>"),
    ("r.env", "env <<i>> (Λ=<<L>>)", "env <<i>> (Λ=<<L>>)"),
    ("r.setup_warning",
     "Avec `K > 0`, l'état du serveur s'ajoute à l'état du système. Le balayage de "
     "prix en lot ne couvre pas ce modèle : seule DP est calculée, et la comparaison "
     "des cinq stratégies n'est pas disponible. Mettre `K = 0` pour y revenir.",
     "With `K > 0`, the server state is added to the system state. The batch price sweep "
     "does not cover this model: only DP is computed, and the five-strategy comparison is "
     "not available. Set `K = 0` to return to it."),
    ("r.avg_profit", "Profit moyen (DP)", "Average profit (DP)"),
    ("r.converged", "Convergence en <<n>> tours d'itération sur les politiques.",
     "Converged in <<n>> policy-iteration rounds."),
    ("r.h_band", "#### Bande d'hystérésis par environnement",
     "#### Hysteresis band per environment"),
    ("r.col_env", "environnement", "environment"),
    ("r.col_restart", "redémarrer sous s_e", "restart below s_e"),
    ("r.col_produce", "produire jusqu'à S_e", "produce up to S_e"),
    ("r.col_width", "largeur de bande", "band width"),
    ("r.band_note",
     "On produit tant que le stock est sous S_e, et l'on ne redémarre qu'une fois "
     "retombé sous s_e. L'écart entre les deux seuils évite de payer K trop souvent.",
     "Production continues while stock is below S_e, and restarts only once stock has "
     "fallen below s_e. The gap between the two thresholds avoids paying K too often."),
    ("r.vs_s", "<<pg>> % vs S", "<<pg>> % vs S"),
    ("r.ylabel_gain", "gain sur S (%)", "gain over S (%)"),
    ("r.col_strategy", "stratégie", "strategy"),
    ("r.col_profit", "profit moyen", "average profit"),
    ("r.col_gain", "gain sur S (%)", "gain over S (%)"),
    ("r.col_bs", "base-stock", "base-stock"),
    ("r.footer",
     "Résolu en **<<t>> s** avec le moteur <<engine>>. "
     "Uniformisation Γ = <<gamma>>. Ordre structurel attendu "
     "g_s ≤ g_sb ≤ g_edp ≤ g_dp et g_s ≤ g_sp ≤ g_edp ≤ g_dp : <<ok>>.",
     "Solved in **<<t>> s** with the <<engine>> engine. "
     "Uniformization Γ = <<gamma>>. Expected structural order "
     "g_s ≤ g_sb ≤ g_edp ≤ g_dp and g_s ≤ g_sp ≤ g_edp ≤ g_dp: <<ok>>."),
    ("r.ok", "vérifié", "verified"),
    ("r.violated", "**violé**", "**violated**"),

    # --- Stocks et prix -----------------------------------------------------------------
    ("s.h_bs", "#### Base-stock par environnement", "#### Base-stock per environment"),
    ("s.ylabel_bs", "base-stock s_e", "base-stock s_e"),
    ("s.thm3", "Théorème 3 : les seuils croissent avec la demande potentielle.",
     "Theorem 3: the thresholds increase with potential demand."),
    ("s.h_price", "#### Prix dynamique en fonction du stock",
     "#### Dynamic price as a function of stock"),
    ("s.xlabel", "niveau de stock x", "stock level x"),
    ("s.ylabel_price", "prix", "price"),
    ("s.price_note",
     "Traits pleins : prix DP, décroissants en stock (Théorème 2). Pointillés : prix "
     "EDP, constants par environnement. Pointillé gris : $p^* = 1/(2a)$, que le prix "
     "DP franchit au base-stock (Corollaire 1). Seuls les états récurrents sont tracés.",
     "Solid lines: DP price, decreasing in stock (Theorem 2). Dashed: EDP price, constant "
     "per environment. Dotted grey: $p^* = 1/(2a)$, which the DP price crosses at the "
     "base-stock (Corollary 1). Only recurrent states are plotted."),

    # --- Recommandation -----------------------------------------------------------------
    ("rc.s_strategy", "**S — un prix unique**", "**S — a single price**"),
    ("rc.s_why",
     "Différencier les prix ne rapporte que <<g>> % ici. En dessous de "
     "~2 %, le gain ne couvre pas le coût d'exploitation d'une politique plus "
     "complexe, ni la friction créée côté client.",
     "Differentiating prices yields only <<g>> % here. Below ~2 %, the gain covers "
     "neither the operating cost of a more complex policy nor the friction it creates on "
     "the customer side."),
    ("rc.edp_strategy", "**EDP — un prix par environnement**",
     "**EDP — one price per environment**"),
    ("rc.edp_why",
     "EDP rapporte <<g>> % sur un prix unique, et DP n'y ajoute que "
     "<<x>> %. Passer à une tarification pilotée par le stock exige un "
     "suivi en temps réel et multiplie les changements de prix, pour un gain marginal.",
     "EDP yields <<g>> % over a single price, and DP adds only <<x>> %. Moving to "
     "stock-driven pricing requires real-time tracking and multiplies price changes, for "
     "a marginal gain."),
    ("rc.dp_strategy", "**EDP**, mais c'est ici que DP mérite un examen",
     "**EDP**, but this is where DP deserves a closer look"),
    ("rc.dp_why",
     "DP dépasse EDP de <<x>> %, soit le haut de la fourchette observée sur "
     "l'ensemble des régimes — l'écart ne dépasse jamais 3,3 %. C'est le domaine où le "
     "niveau de stock porte le plus d'information, et le seul où le pilotage en temps "
     "réel puisse se discuter. À mettre en balance avec son coût de mise en œuvre, que "
     "le modèle ne facture pas.",
     "DP exceeds EDP by <<x>> %, the upper end of the range observed across all regimes "
     "— the gap never exceeds 3.3 %. This is the domain where the stock level carries the "
     "most information, and the only one where real-time control can be argued for. To be "
     "weighed against its implementation cost, which the model does not charge."),
    ("rc.verdict", "Stratégie recommandée : <<s>>", "Recommended strategy: <<s>>"),
    ("rc.note_eps",
     "**Demande peu variable** (`ε = <<eps>>`) — déterminant principal. "
     "Sans fluctuation, aucune tarification adaptative ne se justifie.",
     "**Low demand variability** (`ε = <<eps>>`) — the main determinant. "
     "Without fluctuation, no adaptive pricing is justified."),
    ("rc.note_q",
     "**Environnement instable** (`q = <<q>>`) — les régimes se succèdent "
     "trop vite pour qu'une politique de prix produise ses effets.",
     "**Unstable environment** (`q = <<q>>`) — regimes alternate too quickly for a "
     "pricing policy to take effect."),
    ("rc.note_mu_hi",
     "**Capacité abondante** (`µ = <<mu>>`) — le réapprovisionnement "
     "absorbe seul les fluctuations.",
     "**Abundant capacity** (`µ = <<mu>>`) — replenishment alone absorbs the "
     "fluctuations."),
    ("rc.note_mu_lo",
     "**Capacité très contrainte** (`µ = <<mu>>`) — le stock reste quasi "
     "nul et l'on vend en permanence au prix fort.",
     "**Very tight capacity** (`µ = <<mu>>`) — stock stays near zero and one sells "
     "permanently at the high price."),
    ("rc.note_h",
     "**Stockage coûteux** (`h = <<h>>`) — la tarification gagne en "
     "valeur : elle sert à écouler avant que la facture ne tombe.",
     "**Costly storage** (`h = <<h>>`) — pricing gains value: it serves to clear stock "
     "before the bill falls due."),
    ("rc.limit",
     "**Limite du modèle.** Ni le coût de mise en œuvre d'une politique, ni la réaction "
     "des clients aux variations de prix ne sont facturés. Les gains affichés pour DP sont "
     "donc des majorants, et son avantage réel sur EDP est plus étroit que ces chiffres.",
     "**Model limitation.** Neither the implementation cost of a policy nor customers' "
     "reaction to price changes is charged. The gains shown for DP are therefore upper "
     "bounds, and its real advantage over EDP is narrower than these figures."),
]

TEXTS = {
    "fr": {key: fr for key, fr, _ in _ENTRIES},
    "en": {key: en for key, _, en in _ENTRIES},
}


def translate(lang: str, key: str, **values) -> str:
    """Texte `key` dans `lang` (français si la clé manque), variables `<<nom>>` substituées."""
    text = TEXTS.get(lang, TEXTS["fr"]).get(key) or TEXTS["fr"][key]
    for name, value in values.items():
        text = text.replace(f"<<{name}>>", str(value))
    return text
