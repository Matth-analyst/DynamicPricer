# DynamicPricer — rapport

**Tarification et production d'un système make-to-stock en environnement de demande incertain.**

Ancrage théorique : Gayon, Talay-Değirmenci, Karaesmen & Örmeci (2009), *Optimal Pricing
and Production Policies of a Make-to-Stock System with Fluctuating Demand*, *Probability
in the Engineering and Informational Sciences*, 23(2), 205-230.

---

## 1. Contexte et question

Un fabricant produit un article unique, le stocke, et le vend à une clientèle sensible au
prix. Sa capacité de production est **limitée** — un serveur unique, un article à la fois —
et la demande **fluctue** entre plusieurs régimes qu'il observe mais ne contrôle pas.

Il dispose de deux leviers : *quand produire*, et *à quel prix vendre*. La question posée
est celle du second levier, et elle est d'abord économique avant d'être technique :

> Jusqu'où faut-il complexifier sa politique de prix, et qu'est-ce que cela rapporte
> réellement ?

Trois degrés de sophistication sont possibles, du prix unique au prix recalculé à chaque
mouvement de stock. Chacun coûte plus cher à exploiter que le précédent — systèmes
d'information, réétiquetage, réaction des clients. L'enjeu est de savoir où s'arrêter.

## 2. Le modèle

### 2.1 Système

L'état est le couple `(x, e)` : `x` le niveau de stock, `e` le régime de demande courant.

- **Production.** Un serveur exponentiel de taux `µ`, une unité à la fois. C'est cette
  contrainte de capacité qui distingue le modèle d'un problème d'inventaire classique et
  qui rend le couplage prix/stock non trivial.
- **Environnement.** Une chaîne de Markov à `N` états, de taux de transition `q_ej`.
- **Demande.** Un processus de Poisson modulé, d'intensité `λ_e(p) = Λ_e(1 - a·p)`.
- **Coûts.** Stockage `h` par unité et par unité de temps, production `c` par unité.

### 2.2 Les cinq stratégies

Le papier croise **deux** axes indépendants : le degré de liberté du prix, et celui du
base-stock. Cela donne cinq stratégies, et non trois — distinction essentielle, car les
gains publiés se comparent tous à la plus contrainte d'entre elles.

| | prix unique | un prix par régime | un prix par état |
|---|---|---|---|
| **stock unique** | **S** | **SB** | non considérée |
| **un stock par régime** | **SP** | **EDP** | **DP** |

### 2.3 Équations d'optimalité

Après uniformisation (`µ + Σ_e Λ_e + Σ_e Σ_{j≠e} q_ej = 1`) :

```
v(x,e) + g = -h(x) + µ·T0 v(x,e) + T2 v(x,e) + Tλ v(x,e)
```

où `T0` porte la décision de production, `T2` les transitions d'environnement réelles et
fictives, et `Tλ` l'arrivée de demande. Pour DP, `Tλ` est remplacé par `T = max_λ Tλ`.

Les résultats structurels du papier, tous vérifiés numériquement ici :

- **Théorème 1** — la politique de réapprovisionnement optimale est un base-stock
  dépendant de l'environnement, pour les trois stratégies ;
- **Théorème 2** — sous DP, le prix est décroissant en niveau de stock ;
- **Théorème 3** — sous SP et EDP, les base-stocks sont ordonnés comme les régimes ;
- **Corollaire 1** — le prix DP franchit `p* = 1/(2a)` exactement au base-stock.

### 2.4 Extensions hors périmètre du papier

Trois volets dépassent la source, et leur statut de validation est plus faible (§5.2) :

| extension | traitement |
|---|---|
| `N > 2` régimes | les opérateurs sont déjà génériques en `N` ; seule la Condition 1 (monotonicité des base-stock) demande une vérification explicite hors du cas naissance-mort |
| coût fixe `K > 0` | modélisé en **coût de mise en route du serveur**, avec état augmenté `σ ∈ {marche, arrêt}` |
| backlog `b > 0` | espace d'état étendu à `x ∈ ℤ`, coût de pénurie `b·|x|` sous zéro |

Le choix sur `K` mérite d'être explicité. Une formulation par **lots** — commander une
quantité libre à chaque opportunité — avait d'abord été retenue. Elle a été abandonnée :
elle ne dégénère pas en production unitaire quand `K = 0`, supprime la contrainte de
capacité qui fonde tout le modèle, et produit un profit *supérieur* au cas sans coût fixe.
Le modèle à mise en route préserve la capacité, et sa politique optimale est une **bande
d'hystérésis** : produire jusqu'à `S_e`, ne redémarrer qu'une fois retombé sous `s_e`.
Ce n'est pas une politique `(s, S)` au sens de Chen & Simchi-Levi, dont les résultats
portent sur un réapprovisionnement non capacité et ne s'appliquent donc pas ici.

## 3. Méthodologie numérique

### 3.1 Deux solveurs, qui se contrôlent l'un l'autre

Le projet dispose de **deux méthodes indépendantes** pour le même gain moyen, ce qui
constitue la première ligne de validation :

1. **l'itération sur les politiques**, qui résout exactement `(I - P)v + g·1 = r` pour la
   politique courante puis l'améliore ;
2. **l'évaluation stationnaire**, qui résout `πQ = 0` sur le générateur en temps continu
   et calcule directement `g = Σ π·récompense`.

Les deux concordent à `10⁻¹⁵` sur toutes les instances testées. Elles valident du même
coup l'uniformisation, la normalisation du gain et la construction du générateur, qui
sont les trois endroits où une erreur de modélisation se loge sans se voir.

### 3.2 Pourquoi pas la value iteration

La value iteration était l'approche initiale. Elle ne terminait pas en temps raisonnable
sur le balayage de prix d'EDP — 9801 programmes dynamiques au pas de 0.01.

La cause est algorithmique et non liée au langage. Avec `q = 0.01`, les transitions
d'environnement sont deux ordres de grandeur plus rares que les arrivées de demande : le
mélange de la chaîne est très lent, le module de contraction en span vaut environ
`1 - O(q)`, et il faut donc quelques milliers d'itérations. Or à prix fixé, la politique
optimale est entièrement décrite par `N` entiers. L'itération sur les politiques exploite
cette petitesse et converge en 5 à 8 tours.

Balayage EDP complet à `ε = 0.8` :

| méthode | temps |
|---|---|
| value iteration | n'aboutit pas |
| itération sur les politiques | 87 s |
| + précalcul de la part indépendante de la politique | 59 s |
| + calibration de la troncature par sonde | **17,7 s** |

La suite de tests complète est passée de 5 min 52 s à 32 s. La méthode est de surcroît
**plus exacte** : elle résout l'équation d'évaluation au lieu de l'arrêter à une tolérance.

### 3.3 Deux pièges de convergence, et ce qu'ils enseignent

L'analyse de sensibilité, en explorant des régimes de paramètres que la validation
n'atteignait pas, a révélé deux défauts qui méritent d'être documentés.

**Politiques à valeur identique.** À faible coût de stockage, deux base-stocks voisins
peuvent donner *exactement* le même gain. L'amélioration bascule alors indéfiniment de
l'un à l'autre. Dans un balayage en lot, qui n'aboutit que lorsque tous ses éléments ont
convergé, un seul élément dans ce cas bloque l'ensemble. La détection de **cycle** — une
politique déjà visitée — résout le problème exactement, sans seuil arbitraire.

**Monotonie du gain et bruit numérique.** L'itération sur les politiques garantit un gain
non décroissant, ce qui suggère d'arrêter dès qu'il cesse de progresser. Ce critère est
faux en pratique : avec un coût de pénurie élevé, la fonction de valeur atteint `4·10⁶` et
le bruit numérique sur `g` dépasse les progrès des premiers tours. Appliqué seul, ce
critère concluait à la convergence alors que le gain final était mille fois supérieur.
Il n'est donc retenu que subordonné à la stabilité de la politique de production.

## 4. Validation

### 4.1 Contre le papier — cœur du modèle

Périmètre : `N = 2`, `K = 0`, ventes perdues — le cas exact de la source.

| | résultat |
|---|---|
| **Table 4** — base-stocks | **20 / 20 exacts** |
| **Table 3** — profit gains | écart maximum **0,14 point de pourcentage** |
| **Table 5** — prix S, SB, SP, EDP | tous exacts |
| **Figure 2** — `max_µ PG^{dp,edp}` | **3,22 %** contre 3,24 % publié |
| Propriétés structurelles | Lemme 1, Théorèmes 2 et 3 vérifiés sur les 4 instances |

La Table 4 est le test le plus discriminant : les base-stocks sont des entiers, donc aucun
écart ne peut s'y dissimuler derrière un arrondi. La valeur de la Figure 2 n'a jamais servi
de cible — elle provient d'un balayage indépendant sur `µ` — et constitue donc une
confirmation croisée de toute la chaîne de calcul.

### 4.2 Cohérence interne — extensions

Aucune de ces trois extensions n'a de référence publiée. Elles sont vérifiées par des cas
limites, ce qui est **une garantie réellement plus faible** :

| extension | cas limite | écart |
|---|---|---|
| `N > 2` | un régime dédoublé dans un modèle à 3 états reproduit le cas à 2 états | nul |
| `K > 0` | `K = 0` redonne le modèle sans coût de mise en route | nul |
| backlog | `b → ∞` converge vers le comportement à ventes perdues | `3,5·10⁻⁸` |

### 4.3 Trois écarts trouvés dans la source

La fidélité de la reproduction a permis de relever trois points sur lesquels le papier
est soit imprécis, soit contredit par ses propres données.

**Un.** Le §5.3.2 affirme `s^φ_e ≤ s^dp_e` pour `φ = s, sb, sp, edp`. La Table 4 du papier
publie pourtant `s^sb = 4` et `s^dp_L = 3` à `ε = 0.8`. Le solveur reproduit ces deux
valeurs exactement : la contradiction est dans le texte. L'explication tient à la
classification du §2.2 — S et SB n'ont qu'un seul base-stock, compromis entre les régimes,
qui n'a aucune raison de rester sous le seuil que DP retient dans le régime le plus faible.
La propriété n'est bien posée que pour SP et EDP, où elle est vérifiée partout.

**Deux et trois.** Deux prix DP de la Table 5 restent en écart : `p^dp_L = 0.33` publié
contre 0.31 calculé à `ε = 0.6`, et `p̄^dp_H = 0.99` contre 0.88 à `ε = 0.8`. Le second
est suspect côté source : la suite publiée `0.85 → 0.87 → 0.88 → 0.99` rompt une
progression que le solveur rend régulière (`0.85 → 0.87 → 0.88 → 0.88`), alors que les
quatre autres valeurs DP à `ε = 0.8` coïncident au centime près. Les résultats sont
insensibles à la troncature de l'espace d'état comme à la tolérance numérique.

Ces trois écarts sont **figés dans la suite de tests** : ils échoueraient si une évolution
du solveur les faisait disparaître, plutôt que de se fondre discrètement dans une
tolérance élargie.

## 5. Résultats et enseignements

*(Cette section reprend les analyses de `notebooks/02_sensitivity.ipynb`.)*

### 5.1 Sans fluctuation, la tarification dynamique ne vaut presque rien

En régime unique, S, SB, SP et EDP se confondent, et l'écart de DP mesure exactement ce que
rapporte le fait de faire dépendre le prix du stock. Le papier établit que le maximum
atteignable sur *tout* le plan `(µ, h)` est de **3,81 %** — un plafond, et il est bas.

Le gain est en outre non monotone en capacité : nul aux deux extrémités, maximal au milieu.
À capacité très faible, le stock reste quasi nul et l'on vend en permanence au prix fort ;
à capacité abondante, le réapprovisionnement absorbe seul l'aléa. Le prix n'a de valeur que
lorsque le stock fluctue assez pour porter de l'information.

### 5.2 Avec fluctuation, le gain change d'ordre de grandeur — mais pas où on l'attend

Dès que la demande alterne entre régimes, l'écart dépasse 13 %. L'essentiel ne vient
cependant **pas** de la dépendance au stock : l'écart entre DP et EDP reste sous 3,3 % sur
tout le balayage de capacité.

Autrement dit, la valeur vient de l'**adaptation au régime de demande**, pas de
l'adaptation au niveau de stock. Le premier axe rapporte gros, le second marginalement.
Symétriquement, SP — ajuster ses stocks sans ajuster ses prix — ne rapporte presque rien.

### 5.3 Ce qui fait varier le gain

- **Variabilité de la demande** — déterminant principal. Nul à `ε = 0`, croissant ensuite.
  Le gain de la tarification est une prime à l'incertitude.
- **Coût de stockage** — plus il est élevé, plus la tarification vaut, car elle sert à
  écouler avant que la facture ne tombe.
- **Vitesse des transitions** — le gain **décroît** quand les régimes se succèdent vite :
  aucune politique de prix n'a le temps de produire ses effets.

### 5.4 Deux ou trois prix suffisent

Question la plus opérationnelle de l'étude : que perd-on à se limiter à quelques tarifs
plutôt qu'à un continuum ? Environ **22 % du bénéfice avec deux prix, 8 % avec trois**.

Le rendement décroît très vite, et l'essentiel se joue entre « un seul prix » et « deux
prix ». Une grille à deux ou trois tarifs, lisible pour le client et simple à administrer,
est donc préférable à un moteur de prix continu dans la quasi-totalité des cas.

## 6. Recommandations

| situation | stratégie | justification |
|---|---|---|
| demande stable | **S** — prix unique | le gain de toute alternative reste sous ~2 % |
| régimes marqués et durables | **EDP** — un prix par régime | capte l'essentiel du gain, avec des changements de prix rares |
| capacité très contrainte ou très abondante | **S** | le gain retombe aux deux extrémités |
| capacité intermédiaire et stockage coûteux | **EDP**, éventuellement DP | seul domaine où le stock porte de l'information |
| contrainte sur les changements de prix | **menu à 2 ou 3 tarifs** | ~78 % et ~92 % du bénéfice d'un continuum |

Le fil conducteur est constant sur les quatre balayages : **DP n'apporte que marginalement
plus qu'EDP**, tout en exigeant un pilotage en temps réel, des changements de prix
fréquents, et l'exposition à la réaction des clients. C'est aussi la conclusion du papier
(§5.3.4).

## 7. Limites

**Du modèle.** Ni le coût de mise en œuvre d'une politique, ni la réaction des clients aux
variations de prix ne sont facturés. Les gains affichés pour DP sont donc des **majorants**,
et son avantage réel sur EDP est plus étroit encore que ce que montrent les courbes. Le
modèle suppose par ailleurs l'environnement parfaitement observable ; le papier note que
cela fournit une borne supérieure du gain réellement atteignable.

**De la validation.** Seul le cœur du modèle est validé contre une source publiée. Les
trois extensions — `N > 2`, coût de mise en route, backlog — ne le sont que par cohérence
interne. Elles sont cohérentes avec elles-mêmes, ce qui n'établit pas qu'elles sont
correctes au regard d'une référence externe.

**Du périmètre technique.** Le balayage de prix croît en `(1/pas)^N` : au pas de 0.01 de la
source, `N = 3` est hors d'atteinte et impose une grille plus grossière. Le modèle à coût
de mise en route n'a pas de variante en lot, ce qui limite la comparaison des cinq
stratégies au cas `K = 0`.
