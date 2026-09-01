# DynamicPricer

Simulation et optimisation de politiques de tarification et de production pour un système
make-to-stock en environnement de demande incertain.

Ancrage théorique : Gayon, Talay-Değirmenci, Karaesmen, Örmeci (2009), *Optimal Pricing and
Production Policies of a Make-to-Stock System with Fluctuating Demand*, **Probability in the
Engineering and Informational Sciences**, 23(2), 205-230.

Les spécifications complètes du projet sont dans `../DynamicPricer_SPEC.md`.

## Installation

```bash
pip install -r requirements.txt
```

## Tests

```bash
pytest
```

La validation contre les valeurs publiées (Étape 3) porte la grille de prix complète sur
quatre instances et prend quelques minutes ; elle est donc marquée `slow` et exclue par
défaut. Pour la lancer :

```bash
pytest -m slow
```

Pour recalculer les résultats de validation et alimenter le notebook :

```bash
python -m validation
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

Curseurs sur `µ`, `h`, `ε`, `q`, `a`, plus `N`, `K` et le backlog. Trois onglets :
comparaison des cinq stratégies, niveaux de stock et courbe de prix DP, et une
recommandation rédigée à partir des valeurs calculées.

Deux points d'usage. Le pas de la grille de prix pilote le temps de calcul, qui croît
en `(1/pas)^N` : le pas 0.05 est un bon compromis interactif, le pas 0.01 de l'Appendix E
est réservé à la validation. Et avec `K > 0`, seule la stratégie DP est calculée — le
balayage de prix en lot ne couvre pas l'état augmenté du serveur — le dashboard affiche
alors la bande d'hystérésis et le signale explicitement.

## Structure

```
dynamicpricer/
├── src/
│   ├── params.py       # dataclass Problem : tous les paramètres du modèle
│   ├── mdp.py          # opérateurs T0/T2/Tλ, itération sur les politiques
│   ├── strategies.py   # les cinq stratégies S, SB, SP, EDP, DP
│   ├── metrics.py      # profit gains et contrôles de cohérence
│   ├── validation.py   # comparaison aux tables publiées (Étape 3)
│   ├── native.py       # chargement optionnel du moteur Rust
│   ├── sensitivity.py  # balayages de l'Étape 4
│   └── reference_gayon2009.py  # valeurs publiées, transcrites du papier
├── notebooks/
│   ├── 01_validation.ipynb    # reproduction des Tables 3-5 et Figure 2
│   └── 02_sensitivity.ipynb   # balayages µ, h, ε, q
├── dashboard/app.py
├── rust/               # moteur natif optionnel (balayage de prix)
├── tests/
└── report/
```

## Moteur natif Rust (optionnel)

Le balayage de prix de la stratégie EDP coûte `(1/pas)^N` résolutions. À `N = 2` c'est
supportable ; à `N = 4` au pas 0.05, cela fait 130 321 instances et le moteur Python y
passe une douzaine de minutes. Un moteur natif prend le relais quand il est compilé :

```bash
cd rust && cargo build --release
```

Rien d'autre à faire : `src/native.py` détecte l'artefact, et `solve_batch` retombe
silencieusement sur Python s'il est absent. Le moteur Python reste la **référence** —
c'est lui qui est validé contre le papier — et `tests/test_native.py` fait tenir cette
validation au portage en vérifiant leur équivalence.

Le gain ne vient pas principalement du langage. La matrice d'évaluation `I − P` est
**bloc-tridiagonale** : la production couple le bloc `x` à `x+1`, la demande à `x−1`, et
l'environnement reste dans le bloc. Un algorithme de Thomas par blocs coûte `O(X·N³)` là
où la résolution dense coûte `O((X·N)³)` — pour `N=4`, `X=61`, environ 12 000 opérations
contre 4,8 millions. C'est numpy qui ne sait pas exprimer cela, faute de solveur
bloc-tridiagonal par lot ; Rust est le véhicule, la structure est le levier. `rayon`
répartit ensuite les instances sur les cœurs.

| balayage EDP | Python | Rust | |
|---|---|---|---|
| `N=2`, pas 0.01, `ε=0.8` (9 801) | 47,2 s | 0,8 s | **×59** |
| `N=4`, pas 0.1 (6 561) | 35,4 s | 1,2 s | **×29** |
| `N=4`, pas 0.05 (130 321) | ~12 min | 25,0 s | **×28** |

À l'optimum, les deux moteurs donnent le même gain, la même politique et le même prix,
et les quatre instances de référence retrouvent exactement les Tables 4 et 5.

Une nuance vérifiée et assumée : sur des instances **non optimales**, les base-stocks
peuvent différer d'une ou deux unités entre les deux moteurs. Les gains y sont alors
rigoureusement égaux (écarts de l'ordre de `1e-17`) — ce sont des politiques ex æquo,
que les deux implémentations ne départagent pas de la même façon. Un test dédié vérifie
que toute divergence de seuil s'accompagne d'une égalité de gain.

## Note de performance : itération sur les politiques

Le poste de calcul dominant est le balayage de prix de la stratégie EDP — `(1/pas)^N`
programmes dynamiques, soit 99² = 9801 au pas de 0.01 de l'Appendix E.

En **value iteration**, ce balayage ne terminait pas en un temps raisonnable. La cause
n'est pas l'implémentation : les opérateurs sont vectorisés, et le coût part en nombre
d'itérations. Avec `q = 0.01`, les transitions d'environnement sont deux ordres de
grandeur plus rares que les arrivées de demande, le mélange de la chaîne est très lent
et le module de contraction en span vaut à peu près `1 - O(q)` — il faut donc quelques
milliers d'itérations, quel que soit le langage.

À prix fixé, la politique optimale est pourtant décrite par `N` entiers seulement (les
base-stocks, Théorème 1). L'**itération sur les politiques** exploite cette petitesse :
elle évalue chaque politique exactement en résolvant `(I - P)v + g·1 = r`, puis
l'améliore en relisant les seuils dans `v`. Elle converge en 5 à 8 tours.

Trois points font l'essentiel du gain :

1. la part de la chaîne indépendante de la politique (environnement, demande) est
   précalculée une fois au lieu d'être reconstruite à chaque tour ;
2. la troncature de l'espace d'état est calibrée par une **sonde** sur quelques dizaines
   de vecteurs de prix, au lieu de partir petit et doubler à l'aveugle — une troncature
   trop courte oblige à tout refaire, et le coût d'une résolution est cubique en la
   taille de l'état ;
3. tronquer au-dessus du base-stock optimal n'est pas une approximation : sous une
   politique base-stock, les états supérieurs sont transitoires.

Balayage EDP complet à `ε = 0.8` (9801 combinaisons, `m_stock = 60`) :

| méthode | temps |
|---|---|
| value iteration | n'aboutit pas (arrêtée après plusieurs minutes) |
| itération sur les politiques, version initiale | 87 s |
| + précalcul et indexation directe | 59 s |
| + sonde de troncature | **17,7 s** |

L'itération sur les politiques est aussi **plus exacte** : elle résout l'équation
d'évaluation au lieu de l'approcher, et coïncide avec l'évaluateur stationnaire
`average_profit` au chiffre près, là où la value iteration s'en écarte de l'ordre de sa
tolérance (`1e-5`). `solve_batch_vi` est conservée pour la validation croisée.

## Modèle et limites connues

### `K` est un coût de mise en route, pas un coût de commande par lots

Une première rédaction de la spec posait
`T0 f(x,e) = max{f(x,e), max_{y>x}[f(y,e) - c(y-x) - K]}`, soit une commande de
quantité libre à chaque opportunité. Cette forme a été abandonnée : elle ne dégénère
pas en `max{f(x,e), f(x+1,e) - c}` quand `K = 0` (le maximiseur est le sommet de
`y ↦ f(y,e) - c·y`, pas `y = x+1`), elle supprime la contrainte de capacité qui fonde
le make-to-stock queue, et son gain moyen dépasse celui de l'Étape 1.1 au lieu de le
retrouver — rendant impossible le test de non-régression que la spec exigeait.

`K` est donc le coût de **redémarrage du serveur**. L'état devient `(x, e, σ)` avec
`σ ∈ {0,1}` (arrêté / en marche), la production reste unitaire au taux `µ`, et
l'équation d'optimalité s'écrit
`v = T_switch(-h + µ·T_produce v + T2 v + Tλ v)`.

La politique optimale est une **bande d'hystérésis** : on produit jusqu'à `S_e`, on ne
redémarre qu'une fois retombé sous `s_e`. Ce n'est pas un `(s, S)` au sens de
Chen & Simchi-Levi — ces références portent sur un réapprovisionnement non capacité —
et la structure est donc constatée numériquement (`d(x,e)` décroissante), pas citée.

Les deux validations internes passent : `K = 0` redonne **exactement** l'Étape 1.1
(`g` identique à `1e-12`, bande dégénérée en `(s_e, s_e)`), et la bande s'élargit
pendant que `g` décroît à mesure que `K` augmente.

### Concordance avec les tables publiées

Comparaison directe aux Tables 3 à 5 pour `ε ∈ {0, 0.3, 0.6, 0.8}`, `µ = 0.11`,
au pas de prix 0.01 de l'Appendix E. La validation formelle est l'objet de l'Étape 3 ;
ce qui suit est l'état constaté à l'issue de l'Étape 2.

* **Table 4 — base-stocks : les 16 valeurs concordent exactement**, pour les cinq
  stratégies et les quatre valeurs de `ε`.
* **Table 5 — prix : toutes les valeurs de S, SB, SP et EDP concordent exactement**
  (par exemple `p^edp = (0.57, 0.84)` et `p^sb = (0.55, 0.84)` à `ε = 0.8`).
* **Table 3 — profit gains : écart maximum de 0.2 point** (`PG^{dp,s} = 15.4 %`
  contre 15.2 % publié à `ε = 0.8`), soit ~1.3 % en relatif, dans la tolérance visée.
  Le biais est systématiquement positif et de l'ordre de l'arrondi du papier, qui ne
  publie qu'une décimale.

Deux valeurs de prix **DP** restent en écart, déjà relevées à l'Étape 1 :

| | papier | solveur |
|---|---|---|
| `ε=0.6`, `p^dp_L` (min) | 0.33 | 0.31 |
| `ε=0.8`, `p̄^dp_H` (max) | 0.99 | 0.88 |

La seconde est suspecte côté papier : la suite publiée `0.85 → 0.87 → 0.88 → 0.99`
rompt une progression que le solveur rend régulière (`0.85 → 0.87 → 0.88 → 0.88`),
et les quatre autres valeurs DP à `ε = 0.8` coïncident au centime près. Les résultats
sont insensibles à la troncature (`m` de 60 à 400) et à la tolérance (`1e-5` à `1e-11`).

## Avancement

| Étape | Statut |
|---|---|
| 0 — Cadrage & structure du repo | ✅ |
| 1 — Solveur MDP (opérateurs, K>0, backlog) | ✅ |
| 2 — Cinq stratégies de tarification | ✅ |
| 3 — Validation | ✅ |
| 4 — Analyse de sensibilité | 🔄 |
| 5 — Dashboard | ✅ |
| 6 — Rapport & présentation | ⬜ |
