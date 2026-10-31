# DynamicPricer

Solveur et étude numérique des politiques conjointes de tarification et de production
d'un système make-to-stock à capacité limitée, dont la demande est modulée par une
chaîne de Markov.

Le projet reproduit puis étend les résultats de Gayon, Talay-Değirmenci, Karaesmen et
Örmeci (2009), *Optimal Pricing and Production Policies of a Make-to-Stock System with
Fluctuating Demand*, *Probability in the Engineering and Informational Sciences*, 23(2),
205-230.

---

## Question étudiée

Un fabricant produit un article unique, le stocke, et le vend à une clientèle sensible
au prix. Sa capacité de production est bornée (un serveur unique traitant une unité à la
fois) et la demande alterne entre plusieurs régimes qu'il observe sans les contrôler.

Il dispose de deux leviers : le niveau de stock jusqu'auquel produire, et le prix affiché.
La question posée est celle du second :

> Jusqu'où faut-il complexifier une politique de prix, et quel gain en attendre ?

Trois degrés de sophistication sont envisageables, du prix unique au prix recalculé à
chaque mouvement de stock, chacun plus coûteux à exploiter que le précédent. L'étude
quantifie le gain de chacun et détermine où s'arrêter.

## Modèle

L'état est le couple $(x, e)$ : niveau de stock et régime de demande courant.

| Composante | Spécification |
|---|---|
| Production | serveur exponentiel de taux $\mu$, une unité à la fois, coût unitaire $c$ |
| Environnement | chaîne de Markov à $N$ états, taux de transition $q_{ej}$ |
| Demande | processus de Poisson modulé, intensité $\lambda_e(p) = \Lambda_e \, (1 - a p)$ |
| Coûts | stockage $h$ par unité et unité de temps, pénurie $b$ sous zéro |

Après uniformisation, c'est-à-dire sous la normalisation

$$
\mu + \sum_{e} \Lambda_e + \sum_{e} \sum_{j \neq e} q_{ej} = 1,
$$

l'équation d'optimalité en gain moyen s'écrit

$$
v(x,e) + g = -h(x) + \mu \, T_0 v(x,e) + T_2 v(x,e) + T_\lambda v(x,e),
$$

où $T_0$ porte la décision de production, $T_2$ les transitions d'environnement réelles et
fictives, et $T_\lambda$ l'arrivée de demande. Pour la tarification dynamique, $T_\lambda$ est
remplacé par

$$
T = \max_{\lambda} T_\lambda,
$$

dont le maximiseur admet une forme fermée pour la demande linéaire.

### Les cinq stratégies

Le prix et le base-stock constituent deux axes de liberté indépendants, ce qui donne cinq
politiques et non trois. Tous les gains rapportés le sont relativement à **S**.

| | Prix unique | Un prix par régime | Un prix par état |
|---|---|---|---|
| **Base-stock unique** | S | SB | non considérée |
| **Un base-stock par régime** | SP | EDP | DP |

### Résultats structurels

Quatre propriétés du papier sont vérifiées numériquement sur chaque instance résolue :
la politique de production optimale est un base-stock dépendant du régime (Théorème 1) ;
le prix DP décroît avec le stock (Théorème 2) ; les base-stocks sont ordonnés comme les
régimes sous la Condition 1 (Théorème 3) ; et le prix DP franchit $p^* = \dfrac{1}{2a}$
exactement au niveau de base-stock (Corollaire 1).

## Méthode numérique

Le solveur repose sur l'**itération sur les politiques**. À prix fixés, la politique
optimale est entièrement décrite par $N$ entiers, ce qui rend l'espace des politiques
minuscule : chaque tour résout exactement

$$
(I - P)\, v + g \, \mathbf{1} = r,
$$

puis relit les seuils dans la fonction de valeur. La convergence demande 5 à 8 tours.

La value iteration, employée initialement, ne convenait pas : avec $q$ faible devant les
taux de demande, le mélange de la chaîne est lent, le module de contraction en span vaut
environ $1 - O(q)$, et plusieurs milliers d'itérations sont nécessaires.

Un second solveur, indépendant, calcule le même gain moyen par la loi stationnaire du
générateur en temps continu :

$$
\pi Q = 0.
$$

Les deux méthodes concordent à la précision machine, ce qui valide simultanément
l'uniformisation, la normalisation du gain et la construction du générateur.

### Moteur natif (optionnel)

Deux noyaux dominent le temps de calcul et sont portés en Rust : le balayage de prix de la
stratégie EDP, dont le coût croît en $(1/\text{pas})^N$, et la recherche de seuil commun des
stratégies S et SB, qui évalue chaque niveau candidat pour chaque vecteur de prix.

Le levier principal n'est pas le langage mais la structure. La matrice d'évaluation
$I - P$ est bloc-tridiagonale (la production couple le bloc $x$ au bloc $x+1$, la demande
au bloc $x-1$, l'environnement reste dans le bloc) et ses blocs hors-diagonale sont
eux-mêmes diagonaux, ni la vente ni la production ne changeant de régime. Un algorithme de
Thomas par blocs coûte donc $O(X N^3)$ au lieu de $O\big((X N)^3\big)$ : à $N = 4$ et
$X = 61$, environ 12 000 opérations contre 4,8 millions. numpy ne peut pas l'exprimer, faute
de solveur bloc-tridiagonal par lot. Les instances du balayage étant indépendantes, elles
sont ensuite réparties sur les cœurs disponibles.

| | Python | Rust | |
|---|---|---|---|
| Balayage EDP, $N=2$, pas 0,01, $\varepsilon=0{,}8$ (9 801 vecteurs) | 47,2 s | 0,8 s | x59 |
| Balayage EDP, $N=4$, pas 0,1 (6 561 vecteurs) | 35,4 s | 1,2 s | x29 |
| Seuil commun SB, $N=4$, pas 0,1 | 89,7 s | 3,8 s | x23 |
| `solve_all`, $N=4$, pas 0,1 | 90,6 s | 5,5 s | x16 |
| `solve_all`, $N=4$, pas 0,05 (130 321 vecteurs) | environ 40 min | 104 s | x23 |
| Validation complète contre le papier | 135 s | 16,5 s | x8 |

Le moteur Python demeure la référence : c'est lui qui est validé contre le papier, et le
portage n'hérite de cette validation que par les tests d'équivalence de
`tests/test_native.py`. En l'absence du module compilé, le solveur retombe silencieusement
sur Python.

## Validation

### Contre le papier

Périmètre : $N = 2$, sans coût de mise en route, ventes perdues (le cas exact de la
source).

| Élément | Résultat |
|---|---|
| Table 4, base-stocks optimaux | 20 valeurs sur 20 exactes |
| Table 3, profit gains | écart maximum de 0,14 point de pourcentage |
| Table 5, prix S / SB / SP / EDP | toutes les valeurs exactes |
| Table 2, menu de prix | les 15 valeurs concordent |
| Figure 1, gain maximal mono-environnement | 3,80 % contre 3,81 % publié |
| Figure 2, $\max_\mu PG(\text{dp},\text{edp})$ | 3,22 % contre 3,24 % publié |

La Table 4 constitue le test le plus discriminant : les base-stocks étant des entiers,
aucun écart ne peut s'y dissimuler derrière un arrondi. Les valeurs de la Figure 1 et de
la Figure 2 proviennent de balayages indépendants et n'ont jamais servi de cible.

### Cohérence interne des extensions

Trois volets dépassent le périmètre de la source et ne disposent d'aucune référence
publiée. Ils sont vérifiés par des cas limites, ce qui constitue une garantie
sensiblement plus faible.

| Extension | Cas limite | Écart |
|---|---|---|
| $N > 2$ régimes | un régime dédoublé dans un modèle à 3 états reproduit le cas à 2 états | nul |
| Coût de mise en route $K > 0$ | $K = 0$ redonne le modèle sans coût de mise en route | nul |
| Backlog $b > 0$ | $b \to \infty$ converge vers le comportement à ventes perdues | $3{,}5 \times 10^{-8}$ |

### Écarts relevés dans la source

La fidélité de la reproduction a permis d'identifier trois points sur lesquels le papier
est imprécis ou contredit par ses propres données. Les trois sont figés dans la suite de
tests, de sorte qu'une évolution du solveur les ferait apparaître plutôt que de les
dissimuler dans une tolérance élargie.

1. Le paragraphe 5.3.2 observe, « dans toutes les expériences », que
   $s(\varphi,e) \leq s(\text{dp},e)$ pour
   $\varphi \in \{\text{s}, \text{sb}, \text{sp}, \text{edp}\}$. La Table 4 du même article
   publie pourtant $s(\text{sb}) = 4$ et $s(\text{dp},L) = 3$ à $\varepsilon = 0{,}8$. Le
   solveur reproduit ces deux valeurs exactement : la contradiction est dans le texte. La
   propriété n'est bien posée que pour SP et EDP, dont le base-stock dépend du régime ; S et
   SB n'en ont qu'un, compromis entre les régimes, qui n'a pas de raison de rester sous le
   seuil retenu par DP dans le régime le plus faible.
2. Table 5, $\varepsilon = 0{,}6$ : $p(\text{dp},L)$ publié à 0,33, calculé à 0,31. La valeur
   0,33 figure aussi dans la version de conférence de 2006. Le solveur applique la
   définition du papier (minimum sur les états récurrents, atteint au stock 22) ; 0,33
   correspond au prix du stock 20, mais ce choix de fenêtre contredirait les autres colonnes
   de la même ligne. L'écart reste inexpliqué.
3. Table 5, $\varepsilon = 0{,}8$ : $p(\text{dp},H)$ publié à 0,99, calculé à 0,88. Ici la
   source diverge d'une version à l'autre : la version de conférence de 2006 (mêmes
   paramètres) publie 0,88, que le solveur retrouve (0,883). La version de revue publie 0,99
   et son texte donne un écart de prix de DP de 0,8, cohérent avec 0,99 : ce n'est donc pas
   une simple coquille de tableau, et on ignore si elle provient d'un nouveau calcul ou d'une
   erreur propagée. Le solveur ne retrouve 0,99 nulle part : le prix maximal de DP reste à
   0,883 pour toute variabilité de 0,7 à 0,95, et le résultat est insensible à la troncature
   de l'espace d'état comme à la tolérance numérique.

## Principaux résultats

**Sans fluctuation de la demande, la tarification dynamique ne rapporte presque rien.**
En régime unique, S, SB, SP et EDP se confondent, et le gain de DP mesure exactement
l'apport de la dépendance au niveau de stock. Le maximum atteignable sur l'ensemble du plan
$(\mu, h)$ est de 3,81 %. Ce gain est en outre non monotone en capacité : nul aux deux
extrémités, maximal à capacité intermédiaire.

**Avec fluctuation, l'ordre de grandeur change, mais pas là où on l'attend.** Au point de
référence ($\varepsilon = 0{,}8$, $h = 0{,}01$, $\mu = 0{,}11$), le gain atteint 15,3 % pour DP
et 13,7 % pour EDP (il croît fortement avec le coût de stockage : 201 % pour DP à
$h = 0{,}1$, relativement à S), dont l'essentiel provient de l'adaptation au régime et non au niveau de
stock : l'écart entre DP et EDP reste sous 3,3 % sur tout le balayage de capacité.

**Le classement de SP et SB s'inverse avec la capacité.** À capacité contrainte
($\mu = 0{,}11$), ajuster les prix sans les stocks rapporte 12,0 % quand l'inverse ne rapporte
que 2,5 %. À $\mu = 0{,}51$, le rapport s'inverse : 9,1 % contre 2,0 %.

**Deux ou trois tarifs suffisent, dans le cadre étudié.** Pour le système à un seul régime
de la Table 2 ($h = 0{,}01$, cinq niveaux de capacité, troisième prix imposé à la moyenne
des deux premiers, donc borne inférieure), un menu de deux prix capte environ 77 % du
bénéfice d'un continuum, trois prix environ 93 %. Cette part n'a pas été mesurée pour le
système à deux régimes. Le rendement décroît très vite, et l'essentiel se
joue entre un prix unique et deux prix.

## Installation

Python 3.11 ou plus récent (testé sous 3.12).

```bash
pip install -e ".[dashboard,notebooks,dev]"   # ou : pip install -r requirements.txt
```

L'installation en mode éditable (`-e`) est requise : le solveur retrouve le moteur natif et
les résultats dans `rust/` et `report/`, à la racine du dépôt. Le cœur du solveur ne dépend
que de `numpy` ; les extras `dashboard`, `notebooks` et `dev` ajoutent le reste.

Le moteur natif est optionnel et requiert une chaîne Rust :

```bash
cd rust && cargo build --release
```

Aucune autre étape n'est nécessaire : `dynamicpricer/native.py` détecte l'artefact et le solveur
l'utilise automatiquement.

Pour utiliser un moteur compilé ailleurs (autre machine, conteneur, artefact de CI), pointez la
variable d'environnement `DYNAMICPRICER_NATIVE_DIR` vers un dossier `target/` de cargo ou vers
le dossier contenant directement la bibliothèque. Elle est prioritaire sur `rust/target/` ; si
elle ne contient rien d'utilisable, un avertissement est émis et le solveur retombe sur le
moteur Python (mêmes résultats, plus lent).

## Utilisation

Tableau de bord interactif :

```bash
streamlit run dashboard/app.py
```

Curseurs sur $\mu$, $h$, $\varepsilon$, $q$, $a$, ainsi que sur $N$, le coût de mise en route et le
backlog. Cinq onglets : comparaison des cinq stratégies, niveaux de stock et courbe de prix,
recommandation rédigée à partir des valeurs calculées, formulation mathématique du modèle,
et méthode numérique. Le pas de la grille de prix pilote le temps de calcul, qui croît en
$(1/\text{pas})^N$ ; le pas 0,05 constitue un bon compromis interactif.

Tests unitaires et d'intégration :

```bash
pytest
```

Validation contre les valeurs publiées, plus lente et donc exclue par défaut :

```bash
pytest -m slow
```

Recalcul des résultats alimentant les notebooks :

```bash
python -m dynamicpricer.validation
python -m dynamicpricer.sensitivity
```

## Structure

```
dynamicpricer/
├── src/dynamicpricer/           # le paquet Python
│   ├── params.py                # dataclass Problem, uniformisation, demande linéaire
│   ├── mdp/                     # solveur en gain moyen
│   │   ├── operators.py         #   opérateurs de Bellman
│   │   ├── policies.py          #   extraction des seuils base-stock
│   │   ├── value_iteration.py   #   solve_mdp (instance unique)
│   │   ├── batch.py             #   solve_batch, average_profit (grille de prix)
│   │   └── dynamic_pricing.py   #   itération sur les politiques pour DP
│   ├── strategies.py            # les cinq stratégies S, SB, SP, EDP, DP
│   ├── metrics.py               # profit gains, états récurrents, contrôles de cohérence
│   ├── validation.py            # comparaison aux tables publiées
│   ├── sensitivity.py           # balayages de µ, h, ε, q et menus de prix
│   ├── native.py                # chargement optionnel du moteur Rust
│   └── reference_gayon2009.py   # valeurs publiées, transcrites de la source
├── rust/                        # moteur natif : balayage de prix et seuil commun
├── notebooks/
│   ├── 01_validation.ipynb      # reproduction des Tables 2-5 et des Figures 1-2
│   └── 02_sensitivity.ipynb     # analyses de sensibilité et enseignements
├── dashboard/app.py
├── tests/
├── requirements.txt
└── report/                      # rapport HTML autonome, résultats calculés (JSON)
```

## Périmètre et limites

**Du modèle.** Ni le coût de mise en œuvre d'une politique, ni la réaction des clients aux
variations de prix ne sont facturés. Les gains affichés pour la tarification dynamique sont
donc des majorants, et son avantage réel sur la tarification par régime est plus étroit que
ces chiffres. Le modèle suppose par ailleurs le régime parfaitement observable au moment de
décider, hypothèse favorable qui majore elle aussi le gain.

**De la validation.** Seul le cœur du modèle est validé contre une source publiée. Les
trois extensions ne le sont que par cohérence interne : elles sont cohérentes avec
elles-mêmes, ce qui n'établit pas leur correction au regard d'une référence externe.

**Du coût de mise en route.** $K$ est modélisé comme un coût de redémarrage du serveur,
avec un état augmenté marche/arrêt, et non comme un coût de commande par lots. Ce second
choix, envisagé initialement, supprimait la contrainte de capacité qui fonde le modèle et
ne dégénérait pas en production unitaire quand $K = 0$. La politique optimale obtenue est
une bande d'hystérésis, non une politique $(s, S)$ au sens de Chen et Simchi-Levi, dont les
résultats portent sur un réapprovisionnement non capacité et ne s'appliquent donc pas ici.
Le balayage de prix en lot ne couvre pas ce modèle : la comparaison des cinq stratégies
suppose $K = 0$.

**Du coût de calcul.** Le balayage de prix croît en $(1/\text{pas})^N$. Au pas de 0,01 de la
source, $N = 3$ et au-delà imposent une grille plus grossière.

## Référence

Gayon, J.-P., Talay-Değirmenci, I., Karaesmen, F., Örmeci, E. L. (2009). Optimal Pricing
and Production Policies of a Make-to-Stock System with Fluctuating Demand. *Probability in
the Engineering and Informational Sciences*, 23(2), 205-230.

## Licence

Code publié sous licence MIT (voir `LICENSE`). Les valeurs publiées de Gayon *et al.* (2009)
reprises dans `src/dynamicpricer/reference_gayon2009.py` restent attribuées à leurs auteurs ; les articles
eux-mêmes ne sont pas redistribués.
