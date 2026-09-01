# Moteur natif

Prend en charge le **seul** poste coûteux du solveur : le balayage de prix à prix fixés
(`solve_batch`), qui croît en `(1/pas)^N`. Tout le reste — stratégie DP, modèle à coût
de mise en route, évaluation stationnaire — reste en Python.

```bash
cargo build --release
```

`src/native.py` détecte l'artefact automatiquement. Sous Windows, `cargo` produit un
`.dll` que CPython refuse de charger : le chargeur en dépose une copie `.pyd` à côté,
ce qui évite d'exiger `maturin`.

Deux points d'implémentation valent d'être connus avant toute modification.

**La structure bloc-tridiagonale est le cœur du gain.** `I - P` a des blocs sous- et
sur-diagonaux **diagonaux** (la demande et la production ne changent pas
d'environnement) et un bloc diagonal plein `N×N` (l'environnement). Perdre cette
structure — par exemple en autorisant la production à changer d'environnement — ramène
au coût dense.

**La ligne 0 est remplacée, pas supprimée.** `A = I - P` est singulière (`A·1 = 0`). On
remplace la ligne 0 par `v_0 = 0`, ce qui préserve les bandes, on résout deux seconds
membres, et l'on récupère `g` par l'équation de rang 0 laissée de côté. C'est ce qui
permet de rester bloc-tridiagonal malgré la colonne dense de `g`.

La sémantique reproduit `src/mdp.py` à l'identique — mêmes opérateurs, même règle
d'amélioration, même critère d'arrêt. `tests/test_native.py` vérifie l'équivalence ;
c'est par ce test seul que le portage hérite de la validation contre le papier.
