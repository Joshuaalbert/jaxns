# Mathematical notes: classic samples, phantom samples, plateaus, and shrinkage posteriors in nested sampling

This document is self-contained and focuses only on the mathematics. It clarifies:

- what nested sampling (NS) needs from *classic* (dead/live) samples,
- what *phantom* samples are and when they can be reused safely,
- what plateaus (equal likelihood values) imply for shrinkage and identifiability,
- how to condition shrinkage using phantom exceedances, including plateau-safe blockwise updates,
- how to treat correlated phantom chains via effective sample size (ESS) or robust variance estimators,
- **Route 2 (inclusive constraints)**: a discrete-level re-derivation where shrinkage is formulated with `{=, >}` categories and MCMC may accept `>=`, enabling direct plateau exploration.

> **Conventions.**  
> - Let \(L(\theta)\ge 0\) denote the likelihood (or any nonnegative score proportional to it).  
> - In computation one often stores \(\ell(\theta)=\log L(\theta)\). Ordering-only statements are invariant to strictly monotone transforms (e.g. \(L\leftrightarrow \ell\)), but **evidence and posterior weights depend on the actual values of \(L\)** (if using \(\ell\), substitute \(L=\exp(\ell)\)).

---

## 1. Core NS objects and identities

Let \((\Theta,\mathcal F,\mu)\) be a probability space with prior measure \(\mu\) (\(\mu(\Theta)=1\)). Let \(L:\Theta\to[0,\infty)\) be measurable.

### 1.1 Survival prior mass (strict)

Define the *strict survival prior mass* above threshold \(\lambda\ge 0\):

\[
X(\lambda) \;=\; \mu\{\theta : L(\theta) > \lambda\}.
\]

This is non-increasing in \(\lambda\).

Define the *strict constrained prior*:

\[
\pi_\lambda(d\theta)
\;=\;
\mu(d\theta \mid L(\theta)>\lambda)
\;=\;
\frac{\mathbf 1\{L(\theta)>\lambda\}}{X(\lambda)}\,\mu(d\theta).
\]

> Under continuous \(L\), strict vs inclusive constraints coincide a.s.  
> Under discrete/quantized \(L\), strict vs inclusive differs materially (this is the plateau setting).

### 1.2 Evidence as an integral over survival mass

The evidence is

\[
Z = \int_\Theta L(\theta)\,\mu(d\theta).
\]

For \(L\ge 0\), the layer-cake identity gives:

\[
Z = \int_0^\infty X(\lambda)\,d\lambda,
\quad X(\lambda)=\mu(L>\lambda).
\]

Equivalently, using a generalized inverse \( \lambda(x) := \sup\{\lambda : X(\lambda)\ge x\}\) for \(x\in[0,1]\),

\[
Z = \int_0^1 \lambda(x)\,dx.
\]

Nested sampling approximates this by constructing likelihood levels and associated masses.

---

## 2. Classic NS step (strict) and the shrinkage ratio

### 2.1 Shrinkage ratio between successive strict thresholds

Let NS (in the ideal strict-threshold story) produce an increasing sequence \(\lambda_0 < \lambda_1 < \lambda_2 < \cdots\).
Define:

\[
X_i := X(\lambda_i),\qquad r_i := \frac{X_i}{X_{i-1}} \in (0,1].
\]

Then \(X_i = r_i X_{i-1}\) and the shell mass removed at step \(i\) is:

\[
\mu(\Delta_i)
= X_{i-1} - X_i
= (1-r_i)X_{i-1},
\quad \Delta_i := \{\lambda_{i-1} < L \le \lambda_i\}.
\]

Define log-volume coordinates:

\[
s_i := -\log X_i
\quad\Rightarrow\quad
s_i - s_{i-1} = -\log r_i.
\]

> **Plateau warning.** If \(\lambda_i=\lambda_{i-1}\) (a plateau in the sorted dead list), then \(X(\lambda_i)=X(\lambda_{i-1})\) and the strict survival-mass ratio is 1, even though the classic NS discard process may still “advance indices”. This mismatch is the core plateau tension in the strict formulation.

### 2.2 The key place NS uses (approximate) iid

Assume at step \(i-1\), NS maintains \(K\) live points that are (approximately)

\[
\theta^{(1)},\dots,\theta^{(K)} \overset{\text{iid}}{\sim} \pi_{\lambda_{i-1}}.
\]

Define the *prior-mass coordinate*:

\[
U(\theta) := X(L(\theta)).
\]

If \(\theta \sim \pi_{\lambda_{i-1}}\), then marginally \(U(\theta)\) is uniform on \((0,X_{i-1})\). If the \(K\) points are iid, then

\[
\frac{\max_{j\le K} U(\theta^{(j)})}{X_{i-1}}
\sim \max\{K \text{ iid Unif}(0,1)\}.
\]

The next threshold \(\lambda_i\) is the minimum likelihood among the \(K\) live points, which corresponds to the *largest* \(U\). Therefore (under the iid approximation):

\[
r_i = \frac{X_i}{X_{i-1}}
= \max_{j\le K} V_j,
\quad V_j \overset{\text{iid}}{\sim}\mathrm{Unif}(0,1).
\]

Hence:

\[
r_i \sim \mathrm{Beta}(K,1),
\quad
\mathbb P(r_i \le u)=u^K,
\quad
-\log r_i \sim \mathrm{Exp}(K).
\]

**Important:** this distribution depends on the *joint* iid property of the live set, not just correct marginals.

---

## 3. Phantom samples: definition and what they can estimate

### 3.1 What is a phantom sample?

At a constraint \(\lambda\), a constrained MCMC kernel \(K_\lambda(\theta,\cdot)\) is used to produce new points targeting \(\pi_\lambda\).
During one replacement move, the sampler may generate intermediate states

\[
\theta_1,\theta_2,\dots,\theta_M
\]

that are (after burn-in within that move) approximately **stationary** for \(\pi_\lambda\) but are **correlated**.

We call these intermediate states **phantom samples**. They are not inserted as live points and do not affect the NS discard schedule.

### 3.2 Fundamental identity (strict): exceedances equal compression ratios

Fix two strict thresholds \(\lambda < \lambda'\). Then:

\[
\frac{X(\lambda')}{X(\lambda)}
=
\frac{\mu(L>\lambda')}{\mu(L>\lambda)}
=
\mu(L>\lambda' \mid L>\lambda)
=
\mathbb P_{\theta\sim\pi_\lambda}(L(\theta)>\lambda').
\]

Thus:

\[
\boxed{
r(\lambda\to\lambda') := \frac{X(\lambda')}{X(\lambda)} = \mathbb E_{\pi_\lambda}\left[\,\mathbf 1\{L>\lambda'\}\,\right].
}
\]

So, if we have phantom samples \(\theta_m\sim\pi_\lambda\) (stationary, possibly correlated), then:

\[
\hat r(\lambda\to\lambda')
=
\frac{1}{M}\sum_{m=1}^M \mathbf 1\{L(\theta_m)>\lambda'\}
\]

is a consistent estimator of \(r(\lambda\to\lambda')\). Correlation affects variance, not the target mean (under stationarity/ergodicity).

---

## 4. Reusing looser samples to get more out of phantoms

This section is about using **as many phantom points as possible**, including those generated under looser constraints.

### 4.1 Conditioning lemma (strict)

If \(\lambda^* \le \lambda\) and \(\theta \sim \pi_{\lambda^*}\), then

\[
\theta \mid (L(\theta)>\lambda) \sim \pi_\lambda.
\]

**Derivation.** For measurable \(B\subseteq\Theta\),

\[
\mathbb P_{\pi_{\lambda^*}}(\theta\in B\mid L>\lambda)
=
\frac{\mu(B\cap\{L>\lambda\})/X(\lambda^*)}{X(\lambda)/X(\lambda^*)}
=
\frac{\mu(B\cap\{L>\lambda\})}{X(\lambda)}
=
\pi_\lambda(B).
\]

### 4.2 Implication (strict)

A phantom draw generated under a **looser** constraint \(\lambda^*\le \lambda\) can be used as a draw from \(\pi_\lambda\) **only after conditioning/filtering** on \(L>\lambda\).

This enables pooling of phantom samples from earlier (looser) constraints to estimate quantities under tighter constraints, but acceptance probability becomes small deep in the run.

### 4.3 Ratio-of-means form (strict, no filtering required)

For fixed \(\lambda^*\le \lambda < \lambda'\), define indicators under \(\pi_{\lambda^*}\):

\[
A := \mathbf 1\{L>\lambda\},\qquad B := \mathbf 1\{L>\lambda'\}.
\]

Then:

\[
\mathbb E_{\pi_{\lambda^*}}[A] = \frac{X(\lambda)}{X(\lambda^*)},\qquad
\mathbb E_{\pi_{\lambda^*}}[B] = \frac{X(\lambda')}{X(\lambda^*)}.
\]

Therefore:

\[
\boxed{
\frac{X(\lambda')}{X(\lambda)} = \frac{\mathbb E[B]}{\mathbb E[A]}.
}
\]

A pooled estimator using samples from \(\pi_{\lambda^*}\) is:

\[
\hat r(\lambda\to\lambda') = \frac{\sum B_m}{\sum A_m},
\]

which can be preferable because it uses the full phantom series (no thinning by filtering).

---

## 5. Plateaus under strict constraints: what fails and what still works

### 5.1 What is a plateau?

A plateau is a set of dead points with identical likelihood value that appears repeatedly in the ordered dead-point list.
Formally, a plateau block is a maximal set of indices \(i\in\{a,\dots,b\}\) such that

\[
L_i = L_{i+1} = \cdots = L_b = \tilde \lambda.
\]

### 5.2 Two different “masses” that diverge on plateaus

- **Strict survival mass** \(X(\lambda)=\mu(L>\lambda)\) is a function of threshold \(\lambda\) only.
- **Index-associated mass** in classic NS tracks a sequence of shrinkages produced by repeatedly discarding points, even if the observed likelihood does not change.

On a plateau:

\[
L_i=L_{i-1}=\tilde\lambda
\quad\Rightarrow\quad
X(\tilde\lambda)=X(\tilde\lambda),
\]
so the strict survival-mass ratio is exactly 1:

\[
\frac{X(\tilde\lambda)}{X(\tilde\lambda)} = 1.
\]

But classic NS still discards live points and therefore advances its index-based construction.

### 5.3 Identifiability failure from strict exceedances

Strict phantom conditioning uses identities of the form:

\[
\frac{X(\lambda')}{X(\lambda)} = \mathbb E_{\pi_\lambda}[\mathbf 1\{L>\lambda'\}].
\]

If \(\lambda'=\lambda\), then under \(\pi_\lambda\) we have \(L>\lambda\) almost surely, so \(\mathbf 1\{L>\lambda\}\equiv 1\). Thus phantom exceedances contain **no information** about any “micro-shrinkage” that classic NS implicitly assigns while discarding multiple equal-likelihood points.

What strict phantoms *can* still do on plateaus:

- infer **between distinct likelihood values** (between blocks),
- infer **equality mass at a boundary** *from the previous block* (via the event \(L=\tilde\lambda\) under a looser constraint),
- estimate **expectations within shells** using mixture identities (see §10.1).

---

## 6. Boundary equality mass and Dirichlet modelling (strict)

When plateaus represent a genuine atom at \(\tilde\lambda_g\), the event \(L=\tilde\lambda_g\) can have non-negligible probability under \(\pi_{\tilde\lambda_{g-1}}\).
Then boundary outcomes can be categorized into three classes relative to \(\tilde\lambda_g\):

- \(>\): \(L>\tilde\lambda_g\)
- \(=\): \(L=\tilde\lambda_g\)
- \(<\): \(\tilde\lambda_{g-1}<L<\tilde\lambda_g\)

Let their probabilities under \(\pi_{\tilde\lambda_{g-1}}\) be \((p_>,p_=,p_<)\).

Between-block strict shrinkage is:

\[
r_g = \frac{X(\tilde\lambda_g)}{X(\tilde\lambda_{g-1})} = p_>.
\]

A conjugate Bayesian model uses a Dirichlet prior:

\[
(p_>,p_=,p_<) \sim \mathrm{Dirichlet}(\alpha_>^{(0)},\alpha_=^{(0)},\alpha_<^{(0)}).
\]

If phantom outcomes were iid multinomial with counts \((n_>,n_=,n_<)\), the posterior is:

\[
(p_>,p_=,p_<)\mid \text{data} \sim
\mathrm{Dirichlet}(\alpha_>^{(0)}+n_>,\;\alpha_=^{(0)}+n_=,\;\alpha_<^{(0)}+n_<).
\]

Marginally, \(p_>\) has a Beta distribution:

\[
p_> \sim \mathrm{Beta}\left(\alpha_>,\;\alpha_0-\alpha_>\right),
\quad \alpha_0=\alpha_>+\alpha_=\ +\alpha_<.
\]

### 6.1 Connecting classic NS to prior strength (strict)

In the standard iid-live-set idealization with \(K\) live points at the boundary, one often uses:

\[
r_g=p_> \sim \mathrm{Beta}(K,1).
\]

To embed this as a Dirichlet prior marginal on \(p_>\), choose:

\[
\alpha_>^{(0)} = K,\qquad
\alpha_=^{(0)}+\alpha_<^{(0)} = 1,
\]

with a small split (e.g. \(\alpha_=^{(0)}=\varepsilon,\alpha_<^{(0)}=1-\varepsilon\)).

### 6.2 Correlated phantoms: ESS-adjusted Dirichlet (strict)

If phantom samples are correlated, the multinomial likelihood is not exact. A pragmatic approximation is:

- estimate a scalar effective sample size \(M_{\mathrm{eff}}\) (via batch means or cluster-robust variance),
- replace raw counts by fractional effective counts \(n_{k,\mathrm{eff}} = M_{\mathrm{eff}} \hat p_k\),
- update Dirichlet parameters with \(n_{k,\mathrm{eff}}\).

(See §9 for ESS mechanics.)

---

## 7. Route 2 (inclusive constraints): `{=, >}` level ratios, plateau exploration, and posterior mass for all points

This is the alternative re-derivation you asked for.

### 7.1 Inclusive survival mass and inclusive constrained prior

Define the *inclusive survival mass*:

\[
Y(\lambda) \;=\; \mu\{\theta : L(\theta) \ge \lambda\}.
\]

Define the *inclusive constrained prior*:

\[
\bar\pi_\lambda(d\theta)
\;=\;
\mu(d\theta \mid L(\theta)\ge\lambda)
\;=\;
\frac{\mathbf 1\{L(\theta)\ge\lambda\}}{Y(\lambda)}\,\mu(d\theta).
\]

Relationship to strict survival mass:

\[
Y(\lambda) = X(\lambda) + \mu(L=\lambda).
\]

- If \(\mu(L=\lambda)=0\) for all \(\lambda\) (continuous case), then \(X(\lambda)=Y(\lambda)\) and Route 2 collapses to Route 1.
- If \(\mu(L=\lambda)>0\) for some \(\lambda\) (discrete/quantized plateaus), Route 2 keeps that mass explicit.

### 7.2 Discrete levels, plateau mass, and a “level-ratio” shrinkage

Assume (for Route 2’s cleanest form) that the likelihood takes values on a discrete increasing grid of *attainable levels*:

\[
\lambda_0 < \lambda_1 < \cdots < \lambda_G,
\]

where \(\lambda_{g+1}\) is the next attainable value above \(\lambda_g\).

Define the *level mass* (the atom at the level):

\[
w_g := \mu(L=\lambda_g).
\]

Then inclusive survival at the level is:

\[
Y_g := Y(\lambda_g) = \sum_{h=g}^{G} w_h,
\quad \text{with } Y_{G+1}:=0.
\]

Define the *level-ratio* (Route 2 “shrinkage”) as:

\[
s_g := \frac{Y_{g+1}}{Y_g} \in [0,1].
\]

Then the level mass is:

\[
\boxed{
w_g = Y_g - Y_{g+1} = (1-s_g)\,Y_g.
}
\]

This is the key simplification: **plateau mass is explicit and does not require any within-plateau micro-shrinkage model.**

### 7.3 Fundamental identity (inclusive): `{=, >}` as a Bernoulli under \(\bar\pi_{\lambda_g}\)

Under the inclusive constrained prior \(\bar\pi_{\lambda_g}\),

\[
\mathbb P_{\theta\sim\bar\pi_{\lambda_g}}(L(\theta)\ge\lambda_{g+1})
=
\frac{\mu(L\ge\lambda_{g+1})}{\mu(L\ge\lambda_g)}
=
\frac{Y_{g+1}}{Y_g}
=
s_g.
\]

Equivalently, if \(\lambda_{g+1}\) is the next attainable level, then
\(\{L\ge\lambda_{g+1}\}\) is the same event as \(\{L>\lambda_g\}\), and the complement inside \(\bar\pi_{\lambda_g}\) is exactly \(\{L=\lambda_g\}\).
So Route 2 turns “progress beyond the plateau” into a binary classification:

- `>` (advance): \(L\ge\lambda_{g+1}\)
- `=` (stay on plateau): \(L=\lambda_g\)

with probabilities \((s_g,\;1-s_g)\) under \(\bar\pi_{\lambda_g}\).

### 7.4 Why Route 2 helps plateau MCMC mixing

Under strict constraints \(\pi_{\lambda_g}\propto \mathbf 1\{L>\lambda_g\}\), the plateau set \(L=\lambda_g\) has probability 0 and an MCMC kernel cannot “walk along the plateau”.

Under inclusive constraints \(\bar\pi_{\lambda_g}\propto \mathbf 1\{L\ge\lambda_g\}\), the plateau set \(L=\lambda_g\) has positive probability \(w_g/Y_g\). An MCMC kernel may accept proposals with \(L=\lambda_g\), allowing the chain to explore the plateau geometry and potentially find routes to higher levels more efficiently.

Crucially, Route 2 also makes plateau exploration *statistically useful*:
- time spent at \(L=\lambda_g\) informs both \(w_g\) (via \(s_g\)) and expectations conditional on \(L=\lambda_g\).

### 7.5 Estimating \(s_g\) with classic live points and phantoms

**Data source A: live-set membership (optional but natural in Route 2).**  
If at some time you have a snapshot of \(K_g\) live points that are approximately iid from \(\bar\pi_{\lambda_g}\),
then the count

\[
C^{\text{live}}_g := \#\{j\le K_g : L(\theta^{(j)})\ge\lambda_{g+1}\}
\]

satisfies

\[
C^{\text{live}}_g \mid s_g \sim \mathrm{Binomial}(K_g, s_g),
\]

in the iid-live-set idealization.

**Data source B: phantom chains at constraint \(\lambda_g\).**  
Given phantom states \(\theta_m \sim \bar\pi_{\lambda_g}\) (stationary, correlated), define the indicator

\[
I_m := \mathbf 1\{L(\theta_m)\ge\lambda_{g+1}\}.
\]

Then \(\mathbb E[I_m]=s_g\). In the iid idealization, \(\sum I_m\) is Binomial; with correlation, use ESS (see §9).

**Pooling across looser constraints to “get the most out of phantoms.”**  
For any looser constraint \(\lambda^*\le\lambda_g\), and draws from \(\bar\pi_{\lambda^*}\), define:

\[
A := \mathbf 1\{L\ge\lambda_g\},\qquad B := \mathbf 1\{L\ge\lambda_{g+1}\}.
\]

Then

\[
\mathbb E[A]=\frac{Y_g}{Y(\lambda^*)},\quad
\mathbb E[B]=\frac{Y_{g+1}}{Y(\lambda^*)}
\quad\Rightarrow\quad
\boxed{s_g = \frac{Y_{g+1}}{Y_g} = \frac{\mathbb E[B]}{\mathbb E[A]}.}
\]

So you can estimate \(s_g\) via a ratio-of-means using phantom samples from earlier constraints without discarding most samples.

### 7.6 A Beta posterior for \(s_g\) (ESS-adjusted)

Choose a Beta prior (weak unless you have a reason otherwise):

\[
s_g \sim \mathrm{Beta}(a_g^{(0)}, b_g^{(0)}).
\]

Aggregate effective Bernoulli counts from:
- live-set membership (if used): \(C^{\text{live}}_g\) successes out of \(K_g\),
- phantom indicators: \(C^{\text{ph}}_{g,\mathrm{eff}}\) effective successes out of \(M^{\text{ph}}_{g,\mathrm{eff}}\) effective trials.

Then update:

\[
\boxed{
s_g \mid \text{data} \;\approx\;
\mathrm{Beta}\Big(
a_g^{(0)} + C^{\text{live}}_g + C^{\text{ph}}_{g,\mathrm{eff}},\;
b_g^{(0)} + (K_g - C^{\text{live}}_g) + (M^{\text{ph}}_{g,\mathrm{eff}} - C^{\text{ph}}_{g,\mathrm{eff}})
\Big).
}
\]

This is exact under iid Bernoulli sampling; it is a moment-matched approximation under correlation via ESS.

### 7.7 From \(\{s_g\}\) to level masses \(\{w_g\}\)

Fix \(Y_0 = 1\) (since \(Y(\lambda_0)=\mu(L\ge\lambda_0)=1\) if \(\lambda_0\) is the minimum attainable level; if not, add a sentinel \(\lambda_{-1}=-\infty\) and start there).

Recurrence:

\[
Y_{g+1} = s_g Y_g
\quad\Rightarrow\quad
Y_g = \prod_{h=0}^{g-1} s_h.
\]

Then:

\[
\boxed{
w_g = Y_g - Y_{g+1} = (1-s_g)\,Y_g
= (1-s_g)\prod_{h=0}^{g-1}s_h.
}
\]

These \(w_g\) are the prior-mass weights assigned to each discrete likelihood level.

### 7.8 Evidence and posterior decomposition (Route 2)

Evidence:

\[
\boxed{
Z = \sum_{g=0}^{G} \lambda_g\,w_g.
}
\]

(If you store log-likelihood \(\ell_g\), replace \(\lambda_g=\exp(\ell_g)\).)

Posterior expectation for any integrable \(f\):

\[
\mathbb E[f\mid \text{data}]
=
\frac{1}{Z}\int f(\theta)L(\theta)\,\mu(d\theta)
=
\frac{1}{Z}\sum_{g=0}^{G}\lambda_g
\int_{L=\lambda_g} f(\theta)\,\mu(d\theta).
\]

Define the equality-conditional mean:

\[
m_g(f) := \mathbb E_\mu[f(\theta)\mid L(\theta)=\lambda_g].
\]

Then

\[
\boxed{
\mathbb E[f\mid \text{data}]
=
\frac{1}{Z}\sum_{g=0}^{G}\lambda_g\,w_g\,m_g(f).
}
\]

So Route 2 separates inference into:
1) **mass inference** via \(\{w_g\}\) (driven by \(\{s_g\}\)), and  
2) **conditional expectations on each plateau** via \(m_g(f)\).

### 7.9 Estimating \(m_g(f)\) using classic and phantom plateau samples

Under Route 2, phantom chains at constraint \(\lambda_g\) target \(\bar\pi_{\lambda_g}\), which includes the plateau \(L=\lambda_g\).
Conditioning gives:

\[
\theta\sim\bar\pi_{\lambda_g}
\quad\Rightarrow\quad
\theta\mid (L=\lambda_g) \sim \mu(\cdot\mid L=\lambda_g).
\]

So any phantom state with \(L=\lambda_g\) is a valid (correlated) draw from the plateau-conditional distribution.

Let:
- \(C_g\): the set of classic dead samples with \(L=\lambda_g\), size \(|C_g|=m_g^{\text{cl}}\).
- \(P_g\): the set of phantom states (collected across relevant chains) with \(L=\lambda_g\), size \(|P_g|=m_g^{\text{ph}}\).

A simple pooled estimator is:

\[
\widehat m_g(f)
=
\frac{\sum_{\theta\in C_g} f(\theta) + \rho_g \sum_{\theta\in P_g} f(\theta)}
{m_g^{\text{cl}} + \rho_g m_g^{\text{ph}}},
\]

where \(\rho_g\in[0,1]\) discounts phantom correlation. A natural choice is

\[
\rho_g := \frac{m_{g,\mathrm{eff}}^{\text{ph}}}{m_g^{\text{ph}}},
\]

with \(m_{g,\mathrm{eff}}^{\text{ph}}\) an ESS for plateau-hit phantoms (batch means on the subsequence of plateau hits, or cluster-robust across get-sample calls).

If you want “maximum reuse” without tuning, a conservative default is:
- \(m_{g,\mathrm{eff}}^{\text{ph}} := \min(m_g^{\text{ph}},\; \#\text{phantom clusters contributing to }P_g)\).

### 7.10 Assigning posterior mass to **all points** (classic + phantoms) under Route 2

Route 2 assigns posterior mass at the **level** first, then distributes it over representative samples for that level.

Define the unnormalized posterior mass carried by level \(g\):

\[
q_g := \lambda_g\,w_g.
\quad\text{(So } Z=\sum_g q_g\text{.)}
\]

Now distribute \(q_g\) across the point set \(C_g \cup P_g\).

Define effective counts:
- classic effective count: \(n_g^{\text{cl}} := m_g^{\text{cl}}\) (or an ESS if you want to discount classic correlation),
- phantom effective count: \(n_g^{\text{ph}} := m_{g,\mathrm{eff}}^{\text{ph}}\).

Total effective count:
\[
n_g := n_g^{\text{cl}} + n_g^{\text{ph}}.
\]

Then assign **unnormalized weights**:

- For each classic point \(\theta\in C_g\):
  \[
  \boxed{
  \tilde w(\theta) := \frac{q_g}{n_g}.
  }
  \]

- For each phantom point \(\theta\in P_g\):
  \[
  \boxed{
  \tilde w(\theta) := \frac{q_g}{n_g}\cdot \frac{n_g^{\text{ph}}}{m_g^{\text{ph}}}.
  }
  \]

Checks:
- Total weight over classics: \(m_g^{\text{cl}}\cdot q_g/n_g = q_g\cdot n_g^{\text{cl}}/n_g\).
- Total weight over phantoms: \(m_g^{\text{ph}}\cdot q_g/n_g\cdot (n_g^{\text{ph}}/m_g^{\text{ph}})=q_g\cdot n_g^{\text{ph}}/n_g\).
- Combined total: \(q_g\).

Finally normalize across all levels to get posterior weights:
\[
\boxed{
w_{\text{post}}(\theta) := \frac{\tilde w(\theta)}{\sum_{\theta'} \tilde w(\theta')} = \frac{\tilde w(\theta)}{Z}.
}
\]

This yields a particle approximation that:
- preserves correct level mass \(q_g\),
- uses phantoms as additional plateau representatives without double-counting correlated draws.

### 7.11 Practical note: Route 2 does **not** need within-plateau micro-shrinkage

Because plateau mass \(w_g\) is explicit and the evidence is a discrete sum, Route 2 avoids the strict-route identifiability question:
“how much volume does each repeated dead point at the same \(L\) consume?”

Instead:
- repeated dead points at \(L=\lambda_g\) are simply additional representatives for estimating \(m_g(f)\),
- \(w_g\) is inferred via \(s_g\) using `{=, >}` occupancy under \(\bar\pi_{\lambda_g}\),
- phantoms can directly improve both \(s_g\) (mass) and \(m_g(f)\) (within-plateau expectations).

---

## 8. Shrinkage posterior using phantom exceedances (strict, no plateaus at that step)

This section remains the clean strict “one step” story.

Set \(\lambda=\lambda_{i-1}\) and \(\lambda'=\lambda_i\). Define indicator under \(\pi_\lambda\):

\[
I_m = \mathbf 1\{L(\theta_m)>\lambda'\},\qquad \theta_m\sim\pi_\lambda.
\]

Then \(r_i = \mathbb E[I_m]\).

### 8.1 Exact conjugate update (iid phantom indicators)

Prior (from classic NS live count \(K\)):

\[
r_i \sim \mathrm{Beta}(K,1).
\]

Likelihood:

\[
C := \sum_{m=1}^M I_m \;\big|\; r_i \sim \mathrm{Binomial}(M, r_i).
\]

Posterior:

\[
\boxed{
r_i \mid C \sim \mathrm{Beta}(\alpha,\beta),
\quad \alpha = K+C,\quad \beta = 1 + (M-C).
}
\]

---

## 9. Correlated phantoms: why Binomial is wrong and how to approximate

Phantom samples are correlated because they come from an MCMC trajectory. Then:

- \(\mathbb E[\hat r] = r\) still holds under stationarity,
- but \(\mathrm{Var}(\hat r)\) is larger than in the iid Binomial model.

There is no exact conjugate posterior without specifying a dependence model.

Two practical approaches are:

1) **Moment-matched Beta/Dirichlet** using an ESS computed from the variance of the mean estimator.
2) **Resampling/bootstrapping** at the cluster level (when multiple independent trajectories exist).

### 9.1 ESS defined by variance matching (indicator mean)

Let \(\hat r = \frac1M\sum_{m=1}^M I_m\). Define \(M_{\mathrm{eff}}\) by:

\[
\mathrm{Var}(\hat r) \approx \frac{r(1-r)}{M_{\mathrm{eff}}}.
\]

Replace unknown \(r\) by \(\hat r\) and estimate \(\mathrm{Var}(\hat r)\) (e.g. batch means). Then:

\[
\boxed{
M_{\mathrm{eff}} := \frac{\hat r(1-\hat r)}{\widehat{\mathrm{Var}}(\hat r)}.
}
\]

Clip \(M_{\mathrm{eff}}\in[1,M]\) for stability.

Define effective successes:

\[
C_{\mathrm{eff}} := M_{\mathrm{eff}}\hat r.
\]

### 9.2 Moment-matched Beta update

Use fractional counts in the conjugate form:

\[
\boxed{
r \mid \text{phantoms} \approx
\mathrm{Beta}\big(K + C_{\mathrm{eff}},\; 1 + (M_{\mathrm{eff}}-C_{\mathrm{eff}})\big).
}
\]

### 9.3 Multicategory boundary case (Dirichlet with ESS)

For boundaries with \((>,=,<)\) categories, let empirical proportions be \(\hat p\) and estimate \(\mathrm{Cov}(\hat p)\) using cluster-robust or batch-means methods. For iid multinomial proportions:

\[
\mathrm{Cov}(\hat p) \approx \frac{\mathrm{diag}(p)-pp^\top}{M}.
\]

Match a scalar ESS via the trace:

\[
\boxed{
M_{\mathrm{eff}} :=
\frac{1-\|\hat p\|_2^2}{\mathrm{tr}\left(\widehat{\mathrm{Cov}}(\hat p)\right)}.
}
\]

Then effective counts \(n_{k,\mathrm{eff}} = M_{\mathrm{eff}}\hat p_k\) are added to a Dirichlet prior.

---

## 10. Compatibility with exponential-race view (strict): distribution of \(-\log r\)

Classic NS prior with \(r\sim \mathrm{Beta}(K,1)\) implies:

\[
\Delta s = -\log r \sim \mathrm{Exp}(K).
\]

After phantom conditioning, \(r\) becomes Beta(\(\alpha,\beta\)) (or marginal Beta from Dirichlet). Then:

\[
p(\Delta s)
=
\frac{1}{B(\alpha,\beta)} e^{-\alpha \Delta s}\left(1-e^{-\Delta s}\right)^{\beta-1},
\qquad \Delta s>0.
\]

If \(\beta\) is an integer, then \(\Delta s\) has an exact sum-of-exponentials representation:

\[
\boxed{
\Delta s \overset{d}{=} \sum_{j=0}^{\beta-1} E_j,
\quad E_j \sim \mathrm{Exp}(\alpha+j)\ \text{independent}.
}
\]

With fractional ESS (\(\beta\) non-integer), this exact decomposition does not hold; one can sample Beta directly or approximate.

---

## 11. Live count recursion with out-degrees (tree view)

Let \(K[i]\) be the number of live points **before** discarding node/sample \(i\) in sorted discard order.
Let \(d[i]\) be the number of children created when discarding node \(i\). Let \(K[0]\) be the virtual root out-degree (the initial live count).

Then the live count recursion is:

\[
\boxed{
K[i+1] = K[i] + d[i] - 1.
}
\]

This matches:
- the \(-1\) for removing one live point,
- the \(+d[i]\) for adding its children.

---

## 12. Summary: what can and cannot use phantoms, and why

### 12.1 Where phantoms are valid (most general view)

Phantoms can be used to estimate any quantity expressible as an expectation under a constrained prior:

- strict compression ratios:
  \[
  \frac{X(\lambda')}{X(\lambda)} = \mathbb E_{\pi_\lambda}[\mathbf 1\{L>\lambda'\}]
  \]
- inclusive level ratios (Route 2):
  \[
  \frac{Y(\lambda')}{Y(\lambda)} = \mathbb E_{\bar\pi_\lambda}[\mathbf 1\{L\ge\lambda'\}]
  \]
- boundary category probabilities (>,=,<) when equality mass is meaningful
- shell-conditional expectations, via mixture identities such as:
  \[
  \pi_\lambda = (1-r)\,\mu(\cdot\mid \Delta) + r\,\pi_{\lambda'}
  \Rightarrow
  \mathbb E_\Delta[f] = \frac{\mathbb E_{\pi_\lambda}[f] - r\,\mathbb E_{\pi_{\lambda'}}[f]}{1-r}.
  \]

Correlation reduces effective information (ESS) but does not change the target mean under stationarity.

### 12.2 Strict route vs Route 2 on plateaus

- **Strict route (`>`):** phantoms cannot sample the equality set \(L=\lambda\), so they cannot directly estimate plateau-conditional means \(m_g(f)=\mathbb E[f\mid L=\lambda_g]\). They can still estimate between-level mass ratios and boundary equality probabilities from looser constraints.

- **Route 2 (`>=`):** the equality set is inside the constrained support. Phantoms can:
  - estimate level ratios \(s_g=Y_{g+1}/Y_g\) using `{=, >}` occupancy,
  - estimate plateau-conditional means by conditioning on \(L=\lambda_g\),
  - contribute directly to a weighted posterior particle set at each plateau, with ESS control.

### 12.3 Correlations: practical approximation

- iid phantom indicators yield exact conjugate Beta/Dirichlet posteriors.
- correlated phantoms invalidate Binomial/Multinomial likelihoods; approximate by:
  - estimating variance of the mean/proportions (batch means or cluster-robust),
  - mapping to \(M_{\mathrm{eff}}\),
  - performing a moment-matched fractional-count Beta/Dirichlet update.

This yields a coherent approximation that extracts substantial information from phantoms while remaining stable on plateaus.
