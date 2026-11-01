[Français](README.md) · **English**

# DynamicPricer

Solver and numerical study of joint pricing and production policies for a
capacity-constrained make-to-stock system whose demand is modulated by a Markov chain.

The project reproduces, then extends, the results of Gayon, Talay-Değirmenci, Karaesmen and
Örmeci (2009), *Optimal Pricing and Production Policies of a Make-to-Stock System with
Fluctuating Demand*, *Probability in the Engineering and Informational Sciences*, 23(2),
205-230.

---

## Research question

A manufacturer produces a single item, stores it, and sells it to price-sensitive customers.
Its production capacity is bounded (a single server processing one unit at a time) and demand
alternates between several regimes that it observes but does not control.

It has two levers: the stock level up to which it produces, and the posted price. The
question addressed concerns the second:

> How far should a pricing policy be complicated, and what gain should be expected from it?

Three degrees of sophistication are conceivable, from a single price to a price recomputed at
every stock movement, each more costly to operate than the previous one. The study quantifies
the gain of each and determines where to stop.

## Model

The state is the pair $(x, e)$: stock level and current demand regime.

| Component | Specification |
|---|---|
| Production | exponential server with rate $\mu$, one unit at a time, unit cost $c$ |
| Environment | Markov chain with $N$ states, transition rates $q_{ej}$ |
| Demand | modulated Poisson process, intensity $\lambda_e(p) = \Lambda_e \, (1 - a p)$ |
| Costs | holding cost $h$ per unit and unit of time, shortage cost $b$ below zero |

After uniformization, that is under the normalization

$$
\mu + \sum_{e} \Lambda_e + \sum_{e} \sum_{j \neq e} q_{ej} = 1,
$$

the average-profit optimality equation reads

$$
v(x,e) + g = -h(x) + \mu \, T_0 v(x,e) + T_2 v(x,e) + T_\lambda v(x,e),
$$

where $T_0$ carries the production decision, $T_2$ the real and fictitious environment
transitions, and $T_\lambda$ the demand arrival. For dynamic pricing, $T_\lambda$ is replaced
by

$$
T = \max_{\lambda} T_\lambda,
$$

whose maximizer has a closed form for linear demand.

### The five strategies

Price and base-stock are two independent degrees of freedom, which gives five policies rather
than three. All reported gains are relative to **S**.

| | Single price | One price per regime | One price per state |
|---|---|---|---|
| **Single base-stock** | S | SB | not considered |
| **One base-stock per regime** | SP | EDP | DP |

### Structural results

Four properties of the paper are checked numerically on every solved instance: the optimal
production policy is a regime-dependent base-stock (Theorem 1); the DP price decreases with
stock (Theorem 2); base-stocks are ordered like the regimes under Condition 1 (Theorem 3);
and the DP price crosses $p^* = \dfrac{1}{2a}$ exactly at the base-stock level
(Corollary 1).

## Numerical method

The solver relies on **policy iteration**. At fixed prices, the optimal policy is fully
described by $N$ integers, which makes the policy space tiny: each round solves exactly

$$
(I - P)\, v + g \, \mathbf{1} = r,
$$

then reads the thresholds back from the value function. Convergence takes 5 to 8 rounds.

Value iteration, used initially, was not suitable: with $q$ small relative to the demand
rates, the chain mixes slowly, the span contraction modulus is about $1 - O(q)$, and several
thousand iterations are needed.

A second, independent solver computes the same average profit from the stationary
distribution of the continuous-time generator:

$$
\pi Q = 0.
$$

The two methods agree to machine precision, which simultaneously validates the
uniformization, the gain normalization and the construction of the generator.

### Native engine (optional)

Two kernels dominate the computing time and are ported to Rust: the price sweep of the EDP
strategy, whose cost grows as $(1/\text{step})^N$, and the common-threshold search of
strategies S and SB, which evaluates every candidate level for every price vector.

The main lever is not the language but the structure. The evaluation matrix $I - P$ is
block-tridiagonal (production couples block $x$ to block $x+1$, demand couples it to block
$x-1$, and the environment stays within the block) and its off-diagonal blocks are themselves
diagonal, since neither a sale nor a production changes the regime. A block Thomas algorithm
therefore costs $O(X N^3)$ instead of $O\big((X N)^3\big)$: at $N = 4$ and $X = 61$, about
12,000 operations against 4.8 million. numpy cannot express it, for lack of a batched
block-tridiagonal solver. Since the instances of the sweep are independent, they are then
spread across the available cores.

| | Python | Rust | |
|---|---|---|---|
| EDP sweep, $N=2$, step 0.01, $\varepsilon=0.8$ (9,801 vectors) | 47.2 s | 0.8 s | x59 |
| EDP sweep, $N=4$, step 0.1 (6,561 vectors) | 35.4 s | 1.2 s | x29 |
| SB common threshold, $N=4$, step 0.1 | 89.7 s | 3.8 s | x23 |
| `solve_all`, $N=4$, step 0.1 | 90.6 s | 5.5 s | x16 |
| `solve_all`, $N=4$, step 0.05 (130,321 vectors) | about 40 min | 104 s | x23 |
| Full validation against the paper | 135 s | 16.5 s | x8 |

The Python engine remains the reference: it is the one validated against the paper, and the
port inherits this validation only through the equivalence tests of `tests/test_native.py`.
When the compiled module is absent, the solver silently falls back to Python.

## Validation

### Against the paper

Scope: $N = 2$, no setup cost, lost sales (the exact case of the source).

| Item | Result |
|---|---|
| Table 4, optimal base-stocks | 20 values out of 20 exact |
| Table 3, profit gains | maximum gap of 0.14 percentage point |
| Table 5, S / SB / SP / EDP prices | all values exact |
| Table 2, price menu | all 15 values agree |
| Figure 1, maximum single-environment gain | 3.80 % against 3.81 % published |
| Figure 2, $\max_\mu PG(\text{dp},\text{edp})$ | 3.22 % against 3.24 % published |

Table 4 is the most discriminating test: since base-stocks are integers, no discrepancy can
hide behind rounding. The values of Figure 1 and Figure 2 come from independent sweeps and
were never used as a target.

### Internal consistency of the extensions

Three components go beyond the scope of the source and have no published reference. They are
checked through limiting cases, which is a markedly weaker guarantee.

| Extension | Limiting case | Gap |
|---|---|---|
| $N > 2$ regimes | one regime duplicated in a 3-state model reproduces the 2-state case | zero |
| Setup cost $K > 0$ | $K = 0$ recovers the model without setup cost | zero |
| Backlog $b > 0$ | $b \to \infty$ converges to the lost-sales behavior | $3.5 \times 10^{-8}$ |

### Discrepancies found in the source

The reproduction brings out three points on which the published values or text differ from the
solver's results, or from themselves. Their interpretation remains to be clarified with the
authors. All three are frozen in the test suite, so that
a change in the solver would bring them to light rather than hide them in a widened tolerance.

1. Section 5.3.2 observes, "in all experiments", that
   $s(\varphi,e) \leq s(\text{dp},e)$ for
   $\varphi \in \{\text{s}, \text{sb}, \text{sp}, \text{edp}\}$. Yet Table 4 of the same
   article publishes $s(\text{sb}) = 4$ and $s(\text{dp},L) = 3$ at $\varepsilon = 0.8$. The
   solver reproduces these two values exactly: the gap does not come from the
   computation. The property is natural only for SP and EDP, whose base-stock depends on the regime; S and
   SB have a single one, a compromise between regimes, which has no reason to stay below the
   threshold chosen by DP in the weakest regime.
2. Table 5, $\varepsilon = 0.6$: $p(\text{dp},L)$ published as 0.33, computed as 0.31. The
   value 0.33 also appears in the 2006 conference version. The solver applies the paper's
   definition (minimum over the recurrent states, reached at stock 22); 0.33 corresponds to
   the price at stock 20, but this choice of window would contradict the other columns of the
   same row. The gap remains unexplained.
3. Table 5, $\varepsilon = 0.8$: $p(\text{dp},H)$ published as 0.99, computed as 0.88. Here
   the source diverges from one version to the other: the 2006 conference version (same
   parameters) publishes 0.88, which the solver recovers (0.883). The journal version
   publishes 0.99 and its text gives a DP price gap of 0.8, consistent with 0.99: this is
   therefore not a mere table typo, and it is unknown whether it comes from a new computation
   or from a propagated error. The solver recovers 0.99 nowhere: the maximum DP price stays at
   0.883 for any variability from 0.7 to 0.95, and the result is insensitive to the state-space
   truncation as well as to the numerical tolerance.

## Main results

**Without demand fluctuation, dynamic pricing yields almost nothing.** In a single regime, S,
SB, SP and EDP coincide, and the gain of DP measures exactly the contribution of dependence on
the stock level. The maximum attainable over the whole $(\mu, h)$ plane is 3.81 %. This gain is
moreover non-monotonic in capacity: zero at both extremes, maximal at intermediate capacity.

**With fluctuation, the order of magnitude changes, but not where one expects.** At the
reference point ($\varepsilon = 0.8$, $h = 0.01$, $\mu = 0.11$), the gain reaches 15.3 % for DP
and 13.7 % for EDP (it grows strongly with the holding cost: 201 % for DP at $h = 0.1$,
relative to S), most of which comes from adapting to the regime rather than to the stock
level: the gap between DP and EDP stays below 3.3 % over the whole capacity sweep.

**The ranking of SP and SB reverses with capacity.** With constrained capacity
($\mu = 0.11$), adjusting prices without stocks yields 12.0 % whereas the converse yields only
2.5 %. At $\mu = 0.51$, the ratio reverses: 9.1 % against 2.0 %.

**Two or three prices suffice, within the framework studied.** For the single-regime system of
Table 2 ($h = 0.01$, five capacity levels, third price constrained to the average of the first
two, hence a lower bound), a two-price menu captures about 77 % of the benefit of a continuum,
and three prices about 93 %. This share has not been measured for the two-regime system. The
returns diminish very quickly, and what matters is the step from one price to two.

## Installation

Python 3.11 or later (tested under 3.12).

```bash
pip install -e ".[dashboard,notebooks,dev]"   # or: pip install -r requirements.txt
```

An editable install (`-e`) is required: the solver finds the native engine and the results in
`rust/` and `report/`, at the root of the repository. The core of the solver depends only on
`numpy`; the `dashboard`, `notebooks` and `dev` extras add the rest.

The native engine is optional and requires a Rust toolchain:

```bash
cd rust && cargo build --release
```

No other step is needed: `dynamicpricer/native.py` detects the artifact and the solver uses it
automatically.

To use an engine compiled elsewhere (another machine, a container, a CI artifact), point the
environment variable `DYNAMICPRICER_NATIVE_DIR` to a cargo `target/` directory or directly to
the directory containing the library. It takes priority over `rust/target/`; if it contains
nothing usable, a warning is issued and the solver falls back to the Python engine (same
results, slower).

## Usage

Interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The interface is available in **French and English**: a selector at the top of the sidebar
switches the language without losing the parameters or the results already computed. By
default, the browser language is used.

Sliders for $\mu$, $h$, $\varepsilon$, $q$, $a$, as well as for $N$, the setup cost and the
backlog. Five tabs: comparison of the five strategies, stock levels and price curve, a
recommendation written from the computed values, the mathematical formulation of the model,
and the numerical method. The step of the price grid drives the computing time, which grows
as $(1/\text{step})^N$; a step of 0.05 is a good interactive compromise.

Unit and integration tests:

```bash
pytest
```

Validation against the published values, slower and therefore excluded by default:

```bash
pytest -m slow
```

Recomputing the results that feed the notebooks:

```bash
python -m dynamicpricer.validation
python -m dynamicpricer.sensitivity
```

## Structure

```
dynamicpricer/
├── src/dynamicpricer/           # the Python package
│   ├── params.py                # Problem dataclass, uniformization, linear demand
│   ├── mdp/                     # average-profit solver
│   │   ├── operators.py         #   Bellman operators
│   │   ├── policies.py          #   extraction of base-stock thresholds
│   │   ├── value_iteration.py   #   solve_mdp (single instance)
│   │   ├── batch.py             #   solve_batch, average_profit (price grid)
│   │   └── dynamic_pricing.py   #   policy iteration for DP
│   ├── strategies.py            # the five strategies S, SB, SP, EDP, DP
│   ├── metrics.py               # profit gains, recurrent states, consistency checks
│   ├── validation.py            # comparison with the published tables
│   ├── sensitivity.py           # sweeps over µ, h, ε, q and price menus
│   ├── native.py                # optional loading of the Rust engine
│   ├── ui_texts.py              # dashboard texts (French / English)
│   └── reference_gayon2009.py   # published values, transcribed from the source
├── rust/                        # native engine: price sweep and common threshold
├── notebooks/
│   ├── 01_validation.ipynb      # reproduction of Tables 2-5 and Figures 1-2
│   └── 02_sensitivity.ipynb     # sensitivity analyses and lessons (in French)
├── dashboard/app.py
├── tests/
├── requirements.txt
└── report/                      # self-contained HTML report (in French), computed results (JSON)
```

## Scope and limitations

**Of the model.** Neither the implementation cost of a policy nor customers' reaction to price
changes is charged. The gains shown for dynamic pricing are therefore upper bounds, and its
real advantage over pricing by regime is narrower than these figures. The model also assumes
the regime is perfectly observable at decision time, a favorable assumption that likewise
overstates the gain.

**Of the validation.** Only the core of the model is validated against a published source. The
three extensions are validated only by internal consistency: they are consistent with
themselves, which does not establish their correctness against an external reference.

**Of the setup cost.** $K$ is modeled as a server restart cost, with an augmented on/off
state, and not as a batch ordering cost. This second choice, considered initially, removed the
capacity constraint on which the model rests and did not degenerate into unit production when
$K = 0$. The resulting optimal policy is a hysteresis band, not an $(s, S)$ policy in the
sense of Chen and Simchi-Levi, whose results concern uncapacitated replenishment and therefore
do not apply here. The batch price sweep does not cover this model: the comparison of the five
strategies assumes $K = 0$.

**Of the computing cost.** The price sweep grows as $(1/\text{step})^N$. At the source's step
of 0.01, $N = 3$ and beyond require a coarser grid.

## Reference

Gayon, J.-P., Talay-Değirmenci, I., Karaesmen, F., Örmeci, E. L. (2009). Optimal Pricing
and Production Policies of a Make-to-Stock System with Fluctuating Demand. *Probability in
the Engineering and Informational Sciences*, 23(2), 205-230.

## License

Code released under the MIT license (see `LICENSE`). The published values of Gayon *et al.*
(2009) reproduced in `src/dynamicpricer/reference_gayon2009.py` remain attributed to their
authors; the articles themselves are not redistributed.
