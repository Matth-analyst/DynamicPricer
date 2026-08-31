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

## Avancement

| Étape | Statut |
|---|---|
| 0 — Cadrage & structure du repo | ✅ |
| 1 — Solveur MDP (opérateurs, K>0, backlog) | ⬜ |
| 2 — Cinq stratégies de tarification | ⬜ |
| 3 — Validation | ⬜ |
| 4 — Analyse de sensibilité | ⬜ |
| 5 — Dashboard | ⬜ |
| 6 — Rapport & présentation | ⬜ |
