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

## Dashboard

```bash
streamlit run dashboard/app.py
```

## Structure

```
dynamicpricer/
├── src/
│   ├── params.py       # dataclass Problem : tous les paramètres du modèle
│   ├── mdp.py          # opérateurs T0/T2/Tλ + value iteration average-cost
│   ├── strategies.py   # les cinq stratégies S, SB, SP, EDP, DP
│   └── metrics.py      # profit gains et contrôles de cohérence
├── notebooks/
│   ├── 01_validation.ipynb    # reproduction des Tables 3-5 et Figure 2
│   └── 02_sensitivity.ipynb   # balayages µ, h, ε, q
├── dashboard/app.py
├── tests/
└── report/
```

## Modèle et limites connues

### Coût fixe de commande (`K > 0`) — modèle distinct, non comparable

L'opérateur prescrit par la spec,
`T0 f(x,e) = max{f(x,e), max_{y>x}[f(y,e) - c(y-x) - K]}`, autorise une quantité
commandée **illimitée** à chaque opportunité de réapprovisionnement. Il ne dégénère
donc pas en `max{f(x,e), f(x+1,e) - c}` quand `K = 0` : le maximiseur est le sommet
de `y ↦ f(y,e) - c·y`, pas `y = x+1`.

Deux conséquences :

1. le **test de non-régression `K=0`** prescrit par l'Étape 1.2 de la spec est
   mathématiquement impossible sous cette forme d'opérateur ;
2. le modèle « lot » relâche la contrainte de capacité qui fonde le make-to-stock
   queue, et son gain moyen **dépasse** celui de l'Étape 1.1 au lieu de le retrouver.

`T0_setup` implémente néanmoins la formule de la spec telle qu'écrite, et produit
bien une politique `(s_e, S_e)`. Les tests comparent des instances **à modèle
constant** (`K` croissant), seule comparaison qui ait un sens. Le choix entre
conserver ce modèle et repasser à un coût de mise en route avec état marche/arrêt
du serveur reste ouvert.

### Écarts résiduels avec la Table 5 du papier

Le solveur reproduit **exactement** la Table 4 (base-stocks DP) pour
`ε ∈ {0, 0.3, 0.6, 0.8}` : `(17,17)`, `(12,20)`, `(7,22)`, `(3,23)`. Sur la Table 5,
deux valeurs isolées diffèrent, à confirmer à l'Étape 3 :

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
| 1 — Solveur MDP (opérateurs, K>0, backlog) | ✅ (1.2 : voir limites ci-dessus) |
| 2 — Cinq stratégies de tarification | ⬜ |
| 3 — Validation | ⬜ |
| 4 — Analyse de sensibilité | ⬜ |
| 5 — Dashboard | ⬜ |
| 6 — Rapport & présentation | ⬜ |
