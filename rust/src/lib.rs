//! Moteur de calcul natif pour DynamicPricer — balayage de prix à prix fixés.
//!
//! Ce module ne réimplémente pas tout le solveur : il ne prend en charge que le
//! **poste dominant**, le balayage `(1/pas)^N` de la stratégie EDP. Une résolution
//! isolée coûte 17 ms côté Python et n'a aucun intérêt à être portée.
//!
//! Deux leviers, dans cet ordre d'importance.
//!
//! **1. Structure bloc-tridiagonale.** À politique fixée, la matrice d'évaluation
//! `A = I - P` a une structure très particulière, indexée par `idx = x*N + e` :
//!
//! * la production envoie `(x,e)` vers `(x+1,e)` — bloc sur-diagonal, **diagonal** ;
//! * la demande envoie `(x,e)` vers `(x-1,e)` — bloc sous-diagonal, **diagonal** ;
//! * l'environnement reste en `x` — bloc diagonal, plein `N×N`.
//!
//! Un algorithme de Thomas par blocs coûte donc `O(X·N³)` là où une résolution dense
//! coûte `O((X·N)³)`. Pour `N=4`, `X=61` : ~12 000 opérations contre ~4,8 millions.
//! C'est le gain principal, et il est indépendant du langage — mais numpy ne sait pas
//! l'exprimer, faute de solveur bloc-tridiagonal par lot.
//!
//! **2. Parallélisme.** Les instances du balayage sont indépendantes : `rayon` les
//! répartit sur les cœurs disponibles.
//!
//! La sémantique reproduit exactement celle de `src/mdp.py` — mêmes opérateurs, même
//! règle d'amélioration, même critère d'arrêt — pour que le test d'équivalence ait un
//! sens. Le modèle à coût de mise en route (`K > 0`, état augmenté `σ`) n'est **pas**
//! couvert ici : il reste traité côté Python.

use pyo3::prelude::*;
use rayon::prelude::*;

/// Seuil relatif de non-amélioration du gain (identique à `_GAIN_STALL_TOL` en Python).
const GAIN_STALL_TOL: f64 = 1e-12;

/// Paramètres communs à toutes les instances d'un balayage, déjà normalisés.
struct Ctx {
    n: usize,
    /// Nombre de niveaux de stock représentés.
    x_len: usize,
    /// Niveau de stock du premier indice (négatif si backlog autorisé).
    x_min: i32,
    lam_pot: Vec<f64>,
    q: Vec<f64>,
    /// Masse des transitions fictives, par environnement (éq. 2 du papier).
    fictitious: Vec<f64>,
    mu: f64,
    c: f64,
    a: f64,
    /// Coût de détention par niveau de stock, déjà évalué.
    holding: Vec<f64>,
    max_sweeps: usize,
}

impl Ctx {
    fn level(&self, x: usize) -> i32 {
        self.x_min + x as i32
    }
}

/// Résout `M X = B` en place par élimination de Gauss avec pivot partiel.
///
/// `m` est `n×n` en ligne-major, `b` est `n×ncols`. Renvoie `false` si la matrice est
/// numériquement singulière — le solveur bascule alors sur l'instance suivante plutôt
/// que de propager des `NaN`.
fn solve_dense(m: &mut [f64], b: &mut [f64], n: usize, ncols: usize) -> bool {
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * n + col].abs();
        for row in (col + 1)..n {
            let val = m[row * n + col].abs();
            if val > best {
                best = val;
                pivot = row;
            }
        }
        if best < 1e-300 {
            return false;
        }
        if pivot != col {
            for k in 0..n {
                m.swap(col * n + k, pivot * n + k);
            }
            for k in 0..ncols {
                b.swap(col * ncols + k, pivot * ncols + k);
            }
        }
        let inv = 1.0 / m[col * n + col];
        for row in (col + 1)..n {
            let factor = m[row * n + col] * inv;
            if factor == 0.0 {
                continue;
            }
            for k in col..n {
                m[row * n + k] -= factor * m[col * n + k];
            }
            for k in 0..ncols {
                b[row * ncols + k] -= factor * b[col * ncols + k];
            }
        }
    }
    for col in (0..n).rev() {
        let inv = 1.0 / m[col * n + col];
        for k in 0..ncols {
            let mut acc = b[col * ncols + k];
            for j in (col + 1)..n {
                acc -= m[col * n + j] * b[j * ncols + k];
            }
            b[col * ncols + k] = acc * inv;
        }
    }
    true
}

/// Évaluation exacte d'une politique base-stock, par élimination bloc-tridiagonale.
///
/// Résout `(I - P) v + g·1 = r` sous la contrainte `v[0] = 0`.
///
/// `A` étant singulière (`A·1 = 0`), on ne peut pas inverser directement. On remplace
/// la ligne 0 par `e_0ᵀ` — ce qui préserve la structure en bandes, la ligne 0 vivant
/// dans le bloc 0 — et l'on résout **deux** seconds membres :
///
/// * `A₁ y = r` (avec `r_0` mis à zéro) ;
/// * `A₁ z = 1` (idem).
///
/// Alors `v = y - g·z` vérifie `v_0 = 0` et les équations de rang `≥ 1`. Le gain se
/// déduit de l'équation de rang 0, restée inutilisée :
/// `g = (r_0 - (A y)_0) / (1 - (A z)_0)`.
fn evaluate(ctx: &Ctx, lam: &[f64], s: &[i32], v_out: &mut [f64]) -> Option<f64> {
    let n = ctx.n;
    let x_len = ctx.x_len;
    let nn = n * n;

    // Blocs diagonaux (pleins) et seconds membres, deux colonnes : r puis 1.
    let mut diag = vec![0.0; x_len * nn];
    let mut rhs = vec![0.0; x_len * n * 2];
    // Sous-diagonale et sur-diagonale : diagonales, un coefficient par environnement.
    let mut sub = vec![0.0; x_len * n];
    let mut sup = vec![0.0; x_len * n];

    for x in 0..x_len {
        let level = ctx.level(x);
        for e in 0..n {
            let produces = level < s[e] && x + 1 < x_len;
            let mut self_mass = ctx.fictitious[e];
            if !produces {
                self_mass += ctx.mu;
            } else {
                sup[x * n + e] = -ctx.mu;
            }
            if x > 0 {
                self_mass += ctx.lam_pot[e] - lam[e];
                sub[x * n + e] = -lam[e];
            } else {
                self_mass += ctx.lam_pot[e];
            }

            let base = x * nn + e * n;
            for j in 0..n {
                diag[base + j] = if j == e {
                    1.0 - self_mass
                } else {
                    -ctx.q[e * n + j]
                };
            }

            let mut reward = -ctx.holding[x];
            if produces {
                reward -= ctx.mu * ctx.c;
            }
            if x > 0 {
                reward += lam[e] * (1.0 - lam[e] / ctx.lam_pot[e]) / ctx.a;
            }
            rhs[(x * n + e) * 2] = reward;
            rhs[(x * n + e) * 2 + 1] = 1.0;
        }
    }

    // Ligne 0 d'origine, conservée avant remplacement : elle sert à déterminer `g`.
    let mut row0 = vec![0.0; nn.min(n * n)];
    row0.copy_from_slice(&diag[0..nn][0..nn]);
    let row0_diag: Vec<f64> = diag[0..n].to_vec();
    let row0_sup = sup[0];
    let row0_r = rhs[0];

    // Remplacement : la ligne 0 devient `v_0 = 0`.
    for j in 0..n {
        diag[j] = if j == 0 { 1.0 } else { 0.0 };
    }
    sup[0] = 0.0;
    rhs[0] = 0.0;
    rhs[1] = 0.0;

    // Élimination avant. `w` stocke `C_x^{-1} U_x`, `U_x` étant diagonal.
    let mut w = vec![0.0; x_len * nn];
    let mut d = vec![0.0; x_len * n * 2];
    let mut c_block = vec![0.0; nn];
    let mut work = vec![0.0; n * (n + 2)];

    for x in 0..x_len {
        c_block.copy_from_slice(&diag[x * nn..(x + 1) * nn]);
        if x > 0 {
            // C_x <- D_x - L_x · W_{x-1}, avec L_x diagonal : mise à l'échelle des lignes.
            for e in 0..n {
                let l = sub[x * n + e];
                if l != 0.0 {
                    for j in 0..n {
                        c_block[e * n + j] -= l * w[(x - 1) * nn + e * n + j];
                    }
                }
            }
        }
        // Second membre : n colonnes pour U_x (diagonal), puis les 2 colonnes du rhs.
        let ncols = n + 2;
        for e in 0..n {
            for j in 0..n {
                work[e * ncols + j] = if j == e { sup[x * n + e] } else { 0.0 };
            }
            let mut r0 = rhs[(x * n + e) * 2];
            let mut r1 = rhs[(x * n + e) * 2 + 1];
            if x > 0 {
                let l = sub[x * n + e];
                r0 -= l * d[((x - 1) * n + e) * 2];
                r1 -= l * d[((x - 1) * n + e) * 2 + 1];
            }
            work[e * ncols + n] = r0;
            work[e * ncols + n + 1] = r1;
        }
        if !solve_dense(&mut c_block, &mut work, n, ncols) {
            return None;
        }
        for e in 0..n {
            for j in 0..n {
                w[x * nn + e * n + j] = work[e * ncols + j];
            }
            d[(x * n + e) * 2] = work[e * ncols + n];
            d[(x * n + e) * 2 + 1] = work[e * ncols + n + 1];
        }
    }

    // Remontée : y et z simultanément.
    let mut y = vec![0.0; x_len * n];
    let mut z = vec![0.0; x_len * n];
    for x in (0..x_len).rev() {
        for e in 0..n {
            let mut acc_y = d[(x * n + e) * 2];
            let mut acc_z = d[(x * n + e) * 2 + 1];
            if x + 1 < x_len {
                for j in 0..n {
                    let wij = w[x * nn + e * n + j];
                    if wij != 0.0 {
                        acc_y -= wij * y[(x + 1) * n + j];
                        acc_z -= wij * z[(x + 1) * n + j];
                    }
                }
            }
            y[x * n + e] = acc_y;
            z[x * n + e] = acc_z;
        }
    }

    // Équation de rang 0, laissée de côté par le remplacement, qui fixe `g`.
    let mut ay0 = 0.0;
    let mut az0 = 0.0;
    for j in 0..n {
        ay0 += row0_diag[j] * y[j];
        az0 += row0_diag[j] * z[j];
    }
    if x_len > 1 {
        ay0 += row0_sup * y[n];
        az0 += row0_sup * z[n];
    }
    let denom = 1.0 - az0;
    if denom.abs() < 1e-300 {
        return None;
    }
    let g = (row0_r - ay0) / denom;
    for i in 0..(x_len * n) {
        v_out[i] = y[i] - g * z[i];
    }
    let _ = row0;
    Some(g)
}

/// Seuils de production relus dans la fonction de valeur : `s_e = min{x : Δv(x,e) ≤ c}`.
fn base_stock_levels(ctx: &Ctx, v: &[f64], out: &mut [i32]) {
    let n = ctx.n;
    for e in 0..n {
        let mut found = false;
        for x in 0..(ctx.x_len - 1) {
            if v[(x + 1) * n + e] - v[x * n + e] <= ctx.c {
                out[e] = ctx.level(x);
                found = true;
                break;
            }
        }
        if !found {
            out[e] = ctx.level(ctx.x_len - 1);
        }
    }
}

/// Itération sur les politiques pour une instance, à prix fixés.
fn solve_instance(ctx: &Ctx, lam: &[f64]) -> (f64, Vec<i32>) {
    let n = ctx.n;
    let mut s = vec![0i32; n];
    let mut s_next = vec![0i32; n];
    let mut s_best = vec![0i32; n];
    let mut v = vec![0.0; ctx.x_len * n];
    let mut g_best = f64::NEG_INFINITY;
    let mut g_prev = f64::NEG_INFINITY;

    for _ in 0..ctx.max_sweeps {
        let g = match evaluate(ctx, lam, &s, &mut v) {
            Some(g) => g,
            None => break,
        };
        if g >= g_best {
            g_best = g;
            s_best.copy_from_slice(&s);
        }
        // Deux politiques de valeur identique feraient osciller l'amélioration
        // indéfiniment ; la non-amélioration du gain tranche ce cas.
        let stalled = g <= g_prev + GAIN_STALL_TOL * g.abs().max(1.0);
        if g > g_prev {
            g_prev = g;
        }
        base_stock_levels(ctx, &v, &mut s_next);
        if s_next == s || stalled {
            break;
        }
        s.copy_from_slice(&s_next);
    }
    (g_best, s_best)
}

/// Balayage complet : `b` vecteurs de prix, résolus en parallèle.
///
/// `lam_grid` est aplati en `(b, n)` — les `n` taux de demande de l'instance 0, puis
/// ceux de l'instance 1, etc. Les taux sont donnés en **unités réelles** et normalisés
/// ici, comme le fait `Problem.normalized()` côté Python.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn solve_batch_py(
    lam_grid: Vec<f64>,
    lam_pot: Vec<f64>,
    q: Vec<f64>,
    mu: f64,
    c: f64,
    h: f64,
    b_cost: f64,
    m_stock: i32,
    m_backlog: i32,
    a: f64,
    max_sweeps: usize,
) -> PyResult<(Vec<f64>, Vec<i32>)> {
    let n = lam_pot.len();
    if n == 0 || q.len() != n * n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dimensions incohérentes entre `lam_pot` et `q`",
        ));
    }
    if lam_grid.len() % n != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "`lam_grid` doit être un multiple de N",
        ));
    }
    let batch = lam_grid.len() / n;

    // Uniformisation, identique à `Problem.normalized()` : les taux et les coûts par
    // unité de temps sont divisés par Γ ; `c` est un coût par événement et ne l'est pas.
    let mut gamma = mu;
    for value in &lam_pot {
        gamma += *value;
    }
    for (idx, value) in q.iter().enumerate() {
        if idx / n != idx % n {
            gamma += *value;
        }
    }

    let lam_pot_n: Vec<f64> = lam_pot.iter().map(|v| v / gamma).collect();
    let q_n: Vec<f64> = q.iter().map(|v| v / gamma).collect();
    let total_lam: f64 = lam_pot_n.iter().sum();
    let mut fictitious = vec![0.0; n];
    for e in 0..n {
        let mut acc = total_lam - lam_pot_n[e];
        for i in 0..n {
            if i == e {
                continue;
            }
            for j in 0..n {
                if j != i {
                    acc += q_n[i * n + j];
                }
            }
        }
        fictitious[e] = acc;
    }

    let x_min = -m_backlog;
    let x_len = (m_stock + m_backlog + 1) as usize;
    let holding: Vec<f64> = (0..x_len)
        .map(|x| {
            let level = x_min + x as i32;
            if level >= 0 {
                (h / gamma) * level as f64
            } else {
                (b_cost / gamma) * (-level) as f64
            }
        })
        .collect();

    let ctx = Ctx {
        n,
        x_len,
        x_min,
        lam_pot: lam_pot_n,
        q: q_n,
        fictitious,
        mu: mu / gamma,
        c,
        a,
        holding,
        max_sweeps,
    };

    let results: Vec<(f64, Vec<i32>)> = (0..batch)
        .into_par_iter()
        .map(|i| {
            let lam: Vec<f64> = lam_grid[i * n..(i + 1) * n]
                .iter()
                .map(|v| (v / gamma).clamp(0.0, ctx.lam_pot[0].max(f64::MAX)))
                .collect();
            let lam: Vec<f64> = lam
                .iter()
                .enumerate()
                .map(|(e, v)| v.clamp(0.0, ctx.lam_pot[e]))
                .collect();
            solve_instance(&ctx, &lam)
        })
        .collect();

    let mut gains = Vec::with_capacity(batch);
    let mut stocks = Vec::with_capacity(batch * n);
    for (g, s) in results {
        gains.push(g * gamma);
        stocks.extend_from_slice(&s);
    }
    Ok((gains, stocks))
}

#[pymodule]
fn dynamicpricer_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_batch_py, m)?)?;
    Ok(())
}
