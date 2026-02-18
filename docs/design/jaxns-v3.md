# JAXNS Version 3 Design

## History of JAXNS

JAXNS version 1 started off as an offshoot from my PhD, as I needed a tool that could perform robust inference on
likelihood with very complex structure. In particular, it was inference over angular quantities where absolute angle was
related to the physics of a system. Phase wrapping created many interesting structures in the posterior, and shed light
on the limitations of inference of such physical quantities from the type of data I was considering. I began exploring
more elaborate physical priors that could ultimately provide enough regularisation to resolve the angular degeneracies,
and was quite successful in that. JAXNS was the tool that made it possible.

I then discovered that others had started using JAXNS, much to my pleasant surprise. I had never lead an open source
project before, though I had participated in several in a satellite manner. So, I began to take serious the job of
upholding JAXNS for the general use of the scientific community. JAXNS v2 was born of this attempt. Along with it came a
few offshoot ideas of my own, with no free time to properly publish them. You see, I have long since stepped out of
formal academia. I no longer need to publish or perish. I am now publish for pleasure. In any case, the original paper
compared JAXNS in pure computational speed to the then-available frameworks. The second paper proposed an idea for
safely retaining phantom points without degrading evidence calculations. Neither of those papers made it past pre-print,
mostly out of a lack of desire to dedicate the time. However, now as JAXNS v3 sits on the doorstep, I may finally make
the effort to put JAXNS out there on equal footing (i.e. formally citable).

## JAXNS version 3

JAXNS version 3 shall address a laundry list of tasks that will position JAXNS as a leader in scientific computing for
the future. This is an age where large scale computing of expensive likelihoods happens on massive clusters, where
simulations are being converted to JAX, where delicate physical signals are being teased out of massive amounts of data.
The solution needs to satisfy some basic principles:

1. Built to scale from laptop to cluster.
2. Efficient dynamic refinement of the posterior.
3. Make use of gradients to efficiently explore high dimensional spaces.
4. A powerful probabilistic programming framework for constructing models.
5. Make use of all information in likelihood evaluations.

### Built to scale from laptop to cluster

We shall make scaling JAXNS effortless. We make this distinction, not to confuse the user with JAX's modality of
distributed computing. I find JAX's distributed computing framework obtuse and too tailored to certain workflows. JAXNS
v3 provides its own distributed computing framework. Why? Because it's easier and more maintainable. Some terminology is
important.

**Device**: a compute unit: CPU/GPU/TPU.

**Node**: a host with devices and network address. Optionally, it may have attached JAX compatible accelerators (
GPUs/TPUs).

**System**: A non-empty set of nodes. Examples: a single laptop; two machines connected behind a firewall; a set of
virtual cloud computing instances on a common VPN.

**Replicated Model**: A probability model that can be identically copied between nodes, and identically produce the same
results given the same inputs and random seed. In practice this will require that the probability model and all inputs
can be pickled. Helper tools are provided for edge cases where the user may have strange objects that need help being
pickled.

**Distributed design**: We shall use Zero-MQ (ZMQ) to construct a peer-to-peer (P2P) network of processes which perform
a distributed computation. We start by defining an interactive command interface.

#### Interactive Flow

Jupyter notebook is the ideal interface for running distributed computations. The user is so familiar with the stateful
system of Jupyter that it is a no-brainer. We define the following flow:

1. `register` Each node that will participate in the computation will join a cluster, and make known its available
   devices.
2. `define` A probabilistic model will be defined on a central host, and the compute devices required to run a single
   model defined. E.g. if 2 GPUs are required to execute the model once, then this is part of the model definition.
3. `launch` A nested sampling run with given parameters is launched, and run until completion.
4. `refine` The results of the previous run may be refined dynamically, e.g. to achieve a better evidence estimate, or
   posterior estimate.
5. `finalise` The results are processed into a final result.
6. `stop` The cluster is torn down.

In JAXNS v1 and v2 only steps 2, 3, and 5 are present. The introduction of a distributed setup adds a few extra things
for managing that, and the largescale nature of the processing naturally implies that a form of resumption/refinement
should be possible. Refinement here means the user is able to perform small steps of computation, analyse the results,
and then go deeper if needed. This iterative flow is good for large scale runs where delicate scientific signals are
being teased out, and the user does not know how deep they should go. Many current frameworks require the user to simply
start from scratch with new 'depth' parameters, i.e. number of live points. JAXNS v3 will leverage the mathematics of
nested smapling to enable dynamic depth sampling, essentially removing the requirement of the user to know ahead of time
the depth hyperparameters.

#### Processes of the distributed system

There are three processes that main up the distributed JAXNS system:

1. `CoordinatorProcess` Load balances likelihood evaluations over cluster using a ROUTER-ROUTER broker. Issues control
   commands over lifecyle of cluster.
2. `NodeProcess` Executes model runs with given inputs and reports results. Control is maintained by PUB/SUB.
3. `UserProcess` A jupyter notebook process, where the user executes a particular flow. The `CoordinatorProcess` and
   `NodeProcess` can both be a subprocess of the `UserProcess`, e.g. in a single node setup.

#### Public API

These are primitives that are made available to the system after installing JAXNS v3, as the executable `jaxns`

`start [-e <env file>]` Starts the process, and starts listening for nodes who wish to join the cluster. It assigns
unique IDs to each node that joins. This yields control immediately. Use `-e` option to provide an environment file. See
environment variables section. Otherwise variables read from environment.

`report [-l <period>]` Reports the current status of cluster, i.e. for each joined node prints the latest reported
stats (uptime,
likelihood evaluations, etc.). Can be done periodically on a loop using `-l` option.

`stop` Issues graceful shutdown to cluster, including coordinator process. Saves any results.

`join [-d -e <env file> -t <cpu|gpu> -g <device list>]` Starts a node process and joins the cluster. Once it has
joined, it performs
work and periodically sends stats to the coordinator process. Without `-d` option, starts process in terminal. Ctrl-C (
SIGINT or SIGTERM) are caught
and lead to graceful shutdown. If `-d` given then start as daemon in background. Use `stop` on the coordinator node to
gracefully shutdown a cluster. Use `-e` to provide an environment file. Otherwise, variables read from environment. Use
`-t` to specify device type CPU, GPU. Use `-g` to specify the GPU devices to be used by node process, which will be
accomplished using `CUDA_DEVICES_VISIBLE`.

#### Environment variables

`CTL_SOCKET` the ZMQ TCP socket endpoint for coordination. E.g. `tcp://123.123.123.123:5087`.

#### How this breaks v2

Currently, in JAXNS v2 parallel computing is achieved using JAX primitives to in parallel refill a discarded shell of
live points. When we discard `n` live points, we refill them in parallel using sharded-map. This tends not to make the
most out of available hardware because all samples need to be realised before moving on. It was also possible to fully
JIT compile the nested sampling code, besides just the likelihood. This is no longer the case, nor needed. JAXNS v3 is
composed of smaller sub-programs that are compiled, but the main utility of JAX functionality is that it operates on
JAX-based likelihoods, getting access to gradients. It also means that we don't need to separate dynamic from static
code. Performance is still a priority, so we provide an entirely JAX functional call path to minmic v2, however we
endevour to achieve this performance withit its use.

## Efficient dynamic refinement of the posterior.

We make use of nested samplings representation as a birth/death tree to allow arbitrarily adding new live points from
any previous sample point. This representation is as follows. Define a sample as a tuple (`lambda`, `x`, `L`), where
`lambda < L`. `x` is the sample point, and `L` is the likelihood at the point. `x` is sampled uniformly and
independently from within `lambda < L`. The birth/death tree representation requires that each sample `lambda`
corresponds to a likelihood of another sample, or else be zero. Formally for `(lambda, x, L) in S`, there exists
`(lambda', x', L') in S` such that `lambda in {0, L'}`.

When this property is true then we can construct a birth/death process which enables enumerating the joint probability
distribution of the shrinkage variables from `L=0` to `L=L_max`. We construct a tree with edges `(lambda, L)` and place
the nodes with horizontal positions equal to their label, or any monotonic transformation of them. This horizontal
placement means that we can draw a vertical line at any `L` and see the number of alive points, by counting the number
of edges crossings.

Equivalently, we can construct a set `U = {(lambda, +1): (lambda, x, L) in S} U {(L, -1): (lambda, x, L) in S}`.
Then the number of alive points at any `L` is given by `n(l) = sum_{(L, sign) in U, L < l} sign`.
Then, we can compute the enclosed prior volumes `V(l) = prod_{(L, sign) in H, L <= l} X(n(L))` and `X(n) ~ Beta(n, 1)`.

So, if we add one new sample (`lambda'`, `x'`, `L'`) and already have computed `{n(L) for L in U}` then we can update
this set by `+1` to `n(L)` if `lambda' < L` and `-1` to `n(L)` if `L' < L`. That allows `O(|S|)` updates per new sample.
Or, after collecting `m` new samples, we can sort and compute all `n(L)` in `O(|S + m| log|S + m|)` time.

## Make use of gradients to efficiently explore high dimensional spaces.

Exploration of a constrained space using gradients can be done by using momentum preserving reflections from likelihood
contours.
This means using a form of adaptive step-size to not overshot boundaries. This is a two-step process:

1. For a contour `lambda` step `x' = x + dt n` along a direction an amount `dt`, and measure likelihood, `L` and
   gradient `g`. If `L < lambda` then set `x=x'` repeat step 1.
2. Reflect `x'' = x' - 2 * dt * g.n / ||g||^2` and measure likelihood, `L'` and gradient `g'`. If `L' < lambda` then
   decrease `dt` (`x` stays same). Otherwise, set `x = x''`. Set `dt = total distance / num steps`. Go to step 1.

We see than `dt` is adaptive, shrinking

## A powerful probabilistic programming framework for constructing models.

JAXNS v2 already provides a powerful probabilistic programming framework for constructing models. It uses TFP as a
foundation to provide distributions. Any distribution with a quantile and CDF is suitable. JAXNS v3 will simply make
this framework better and more compact, by outsource most of the lifting to `jaxctx`, a sister package used for making
parametrised models using the memoisation model.

## Make use of all information in likelihood evaluations.

The likelihoods are useful for learning representations of the likelihood. One way is to use a GP to predict the
log-likelihood at any point in the domain, and then perform a delayed MH acceptance test to accept or reject the point.
In particular, suppose we have a GP representation of the log-likelihood, and wish to sample a new point within a target
likelihood constraint.
Then, construct the probability of improvement, `w(theta) = eps + (1 - eps) Pr(L(theta) > lambda)`, where `eps ~ 1e-2`
is some default probabiltiy preventing accidental collapse. Perform one step of slice sampling
on `p(theta) w(theta)` or HMC since we have gradients of them all.
Accept with probability `min(1, w(theta') / w(theta))`. This leaves `p(theta) w(theta)`. Then accept again if
`L(theta') > lambda`.

This allows quickly exploring the target space using the surrogate and then accepting only if the expensive likelihood
match is made. The better the GP learns the likelihood, the more efficient the exploration. Instead of GPs, we could use
neural networks to learn a different weight function that minimises the loss `|I(L(theta) > lambda) - w(theta)|`.


# Phantom-augmented shrinkage, shell quadrature, and posterior reservoir sampling

This note describes an **online** way to incorporate “phantom” MCMC states (correlated constrained-prior samples) into nested sampling’s **prior-mass shrinkage** and (optionally) **shell quadrature**, while keeping the core nested-sampling schedule unchanged.

We work in likelihood space but only use ordering, so replacing (L) by (\log L) is fine (monotone transforms preserve all inequalities).

---

## 0. Definitions

Let (\mu) be the prior measure on parameter space (\Theta), and (L:\Theta\to\mathbb R) a (possibly log-)likelihood.

For a threshold (\ell), define the constrained set and constrained prior:
[
A(\ell) = {\theta: L(\theta)>\ell},\qquad
\pi_\ell(d\theta) = \mu(d\theta \mid L>\ell)=\frac{\mathbf 1{L>\ell}}{X(\ell)},\mu(d\theta),
]
where (X(\ell) := \mu(A(\ell))\in(0,1]) is the remaining prior mass above (\ell).

Nested sampling produces an increasing sequence of thresholds (\ell_0<\ell_1<\ell_2<\cdots) (in standard NS: (\ell_i) is the likelihood of the discarded worst live point at step (i)).

Define:

* prior mass at step (i): (X_i := X(\ell_i))
* shrinkage / compression ratio:
  [
  r_i := \frac{X_i}{X_{i-1}} \in (0,1)
  ]
* negative log-volume:
  [
  s_i := -\log X_i \quad \Rightarrow \quad s_i - s_{i-1} = -\log r_i.
  ]
* shell (discarded mass slab):
  [
  \Delta_i := A(\ell_{i-1})\setminus A(\ell_i)={\ell_{i-1}<L\le \ell_i}.
  ]
  Then the shell prior mass is:
  [
  \mu(\Delta_i) = X_{i-1}-X_i = (1-r_i)X_{i-1}.
  ]

---

## 1. Baseline shrinkage model (race / order-statistic prior)

Let (K_i) be the number of active live points **just before** producing (\ell_i) (i.e. before discarding the worst point at step (i)). In the standard iid-live-set idealization:

* the shrinkage ratio has distribution:
  [
  r_i \sim \mathrm{Beta}(K_i,,1),
  ]
* equivalently the log-shrink increment is exponential:
  [
  \Delta s_i := s_i-s_{i-1} = -\log r_i \sim \mathrm{Exp}(K_i).
  ]

In a lineage/out-degree formulation, (K_i) evolves via
[
K_{i+1} = K_i - 1 + d(i)
]
where (d(i)) is out-degree / number of spawned children from node (i).

**Interpretation**: (\mathrm{Beta}(K_i,1)) is a *prior* on shrinkage implied by “(K_i) iid constrained-prior live points and we remove the worst”.

---

## 2. Phantom samples as direct information about shrinkage (r_i)

### 2.1 Key identity

For (\ell_i>\ell_{i-1}) (strictly), we have:
[
r_i = \frac{X(\ell_i)}{X(\ell_{i-1})}
= \mathbb P_{\theta\sim \pi_{\ell_{i-1}}}(L(\theta)>\ell_i)
= \mathbb E_{\pi_{\ell_{i-1}}}\left[\mathbf 1{L>\ell_i}\right].
]

Therefore, any samples approximately distributed as (\pi_{\ell_{i-1}}) provide a Monte Carlo estimate of (r_i) by exceedance counting.

### 2.2 What “phantoms” are (for this note)

At level (\ell), phantom points are correlated draws (\theta_m) produced by your constrained MCMC kernel targeting (\pi_\ell). They are **identically distributed (stationary)** but not independent.

For shrinkage at step (i), the relevant phantoms are those produced at the **previous constraint** (\ell_{i-1}):
[
\theta_{i-1,1},\dots,\theta_{i-1,M_i} \approx \pi_{\ell_{i-1}}.
]

Define indicator sequence:
[
I_{i,m} := \mathbf 1{L(\theta_{i-1,m})>\ell_i},\quad m=1..M_i.
]
Then ( \mathbb E[I_{i,m}] = r_i).

---

## 3. Estimating (r_i) from correlated indicators

### 3.1 Point estimate

[
\hat r_i = \frac{1}{M_i}\sum_{m=1}^{M_i} I_{i,m} = \frac{C_i}{M_i},\qquad C_i := \sum_m I_{i,m}.
]

### 3.2 Effective sample size for correlation (indicator ESS)

Correlation inflates (\mathrm{Var}(\hat r_i)) relative to iid Binomial.

Define an ESS by matching the variance of an iid Bernoulli mean:
[
\mathrm{Var}(\hat r_i);\approx;\frac{r_i(1-r_i)}{M_{\mathrm{eff},i}}
\quad\Rightarrow\quad
M_{\mathrm{eff},i} := \frac{\hat r_i(1-\hat r_i)}{\widehat{\mathrm{Var}}(\hat r_i)}.
]

A robust implementation is **batch means** on the indicator sequence:

* Choose (B) blocks, block length (m=\lfloor M_i/B\rfloor).
* Compute block means (\bar I_b = \frac{1}{m}\sum_{t=(b-1)m+1}^{bm} I_t).
* Estimate variance of the mean:
  [
  \widehat{\mathrm{Var}}(\hat r_i) := \frac{1}{B(B-1)}\sum_{b=1}^B (\bar I_b - \bar I_{\cdot})^2,
  \qquad \bar I_{\cdot}=\frac{1}{B}\sum_b \bar I_b.
  ]
* Then set:
  [
  M_{\mathrm{eff},i} := \frac{\hat r_i(1-\hat r_i)}{\widehat{\mathrm{Var}}(\hat r_i)}.
  ]

Practical guards:

* clip (\hat r_i) into ([\epsilon,1-\epsilon]) for ESS computation (e.g. (\epsilon=10^{-6}))
* enforce (M_{\mathrm{eff},i}\in[1,M_i])
* if (M_i) too small for batching, fall back to (M_{\mathrm{eff},i}=M_i) (or skip phantom update).

### 3.3 Pooling multiple phantom fragments

If the phantom data at level (\ell_{i-1}) consists of multiple **independent** fragments (e.g. replacement chains started from different live points / devices), compute (M_{\mathrm{eff}}) per fragment and sum:
[
M_{\mathrm{eff},i} = \sum_{f} M_{\mathrm{eff},i}^{(f)},\qquad
\hat r_i = \frac{\sum_f C_i^{(f)}}{\sum_f M_i^{(f)}}.
]
Then define an “effective exceedance count”:
[
C_{\mathrm{eff},i} := M_{\mathrm{eff},i},\hat r_i.
]

---

## 4. Bayesian shrinkage update: Beta prior + (effective) exceedance data

### 4.1 Prior from live count

Baseline:
[
r_i \sim \mathrm{Beta}(\alpha_i^{(0)},\beta_i^{(0)}),\qquad
\alpha_i^{(0)}=K_i,\ \beta_i^{(0)}=1.
]

### 4.2 Approximate posterior using fractional counts

Under iid Bernoulli exceedances, conjugacy would give:
[
\alpha_i = K_i + C_i,\qquad \beta_i = 1 + (M_i - C_i).
]
For correlated phantoms, replace ((C_i,M_i)) by ((C_{\mathrm{eff},i},M_{\mathrm{eff},i})):
[
\boxed{
\alpha_i := K_i + C_{\mathrm{eff},i},\qquad
\beta_i := 1 + (M_{\mathrm{eff},i} - C_{\mathrm{eff},i}).
}
]
This is a moment-matching “ESS-adjusted Beta” update: it preserves the mean (\hat r_i) while inflating uncertainty.

### 4.3 Two computation modes

**Sampling-based NS (recommended if you already do evidence particles):**

* For each evidence replicate (b=1..B), draw
  [
  r_i^{(b)} \sim \mathrm{Beta}(\alpha_i,\beta_i),
  ]
  then update volumes.

**Expectation-based NS (closed-form moments for log-shrink):**
For (r\sim\mathrm{Beta}(\alpha,\beta)):
[
\mathbb E[\log r] = \psi(\alpha)-\psi(\alpha+\beta),\qquad
\mathrm{Var}(\log r)=\psi_1(\alpha)-\psi_1(\alpha+\beta),
]
where (\psi) is digamma and (\psi_1) trigamma.

Assuming stepwise independence of (r_i) (approximation):
[
\mathbb E[s_i] = \mathbb E[s_{i-1}] - \mathbb E[\log r_i],\qquad
\mathrm{Var}(s_i) = \mathrm{Var}(s_{i-1}) + \mathrm{Var}(\log r_i).
]

---

## 5. Volume recursion and shell mass

Given (X_0=1):

**Sampling-based:**
[
X_i^{(b)} = X_{i-1}^{(b)},r_i^{(b)},\qquad
\mu_i^{(b)} := \mu(\Delta_i)^{(b)} = X_{i-1}^{(b)} - X_i^{(b)} = (1-r_i^{(b)})X_{i-1}^{(b)}.
]

**Expectation-based (plug-in; less exact):**
Use ( \bar r_i = \mathbb E[r_i]=\alpha_i/(\alpha_i+\beta_i)) and propagate ( \bar X_i=\bar X_{i-1}\bar r_i). This is not the same as (\mathbb E[X_i]) unless further assumptions hold; for evidence uncertainty you generally want sampling-based.

---

## 6. Evidence and posterior moments: shell quadrature options

Evidence:
[
Z = \int L(\theta),\mu(d\theta) = \sum_i \int_{\Delta_i} L(\theta),\mu(d\theta)
= \sum_i \mu(\Delta_i),\mathbb E_{\Delta_i}[L].
]

Posterior expectations for any (g(\theta)):
[
\mathbb E_{\text{post}}[g] = \frac{1}{Z}\sum_i \mu(\Delta_i),\mathbb E_{\Delta_i}[gL].
]

We need (i) shell masses (\mu(\Delta_i)) (from shrinkage) and (ii) shell conditional means (\mathbb E_{\Delta_i}[\cdot]).

### 6.1 Minimal (classic) shell representative

Use only boundary likelihoods (\ell_{i-1},\ell_i), e.g.

* right Riemann: (\mathbb E_{\Delta_i}[L] \approx L(\ell_i))
* trapezoid in (X)-space: (\int_{\Delta_i} L,d\mu \approx \frac{L(\ell_{i-1})+L(\ell_i)}{2},(X_{i-1}-X_i))

This needs no phantom parameter storage.

### 6.2 Phantom-based shell means via direct shell hits (fast when shells are not tiny)

If you can collect samples (\theta\sim \pi_{\ell_{i-1}}) (phantoms at level (i-1)), then conditioning to the shell gives correct shell distribution:
[
\theta \mid (\ell_{i-1}<L\le \ell_i) \sim \mu(\cdot\mid \Delta_i).
]
So you can estimate shell means by filtering phantoms:
[
\widehat{\mathbb E_{\Delta_i}[f]} = \frac{1}{|\mathcal S_i|}\sum_{\theta\in\mathcal S_i} f(\theta),
\quad \mathcal S_i := {\theta_{i-1,m} : \ell_{i-1}<L(\theta_{i-1,m})\le \ell_i}.
]
If (|\mathcal S_i|=0), this fails (common when (1-r_i) is tiny).

### 6.3 Phantom-based shell means without shell hits (mixture identity)

Use the exact decomposition:
[
\pi_{\ell_{i-1}} = (1-r_i),\mu(\cdot\mid \Delta_i) + r_i,\pi_{\ell_i}.
]
Therefore for any integrable (f):
[
\boxed{
\mathbb E_{\Delta_i}[f] = \frac{\mathbb E_{\pi_{\ell_{i-1}}}[f] - r_i,\mathbb E_{\pi_{\ell_i}}[f]}{1-r_i}.
}
]

Implementation pattern:

* Maintain running estimates (m_i(f)\approx \mathbb E_{\pi_{\ell_i}}[f]) from phantoms generated at level (\ell_i).
* When (r_i) becomes available, compute (\mathbb E_{\Delta_i}[f]) using (m_{i-1}(f)) and (m_i(f)).

**Streaming note (1-step lag):**

* At time (\ell_i) is created, (m_i(f)) may be unavailable (you haven’t generated phantoms at (\ell_i) yet).
* So finalize shell means for (\Delta_i) after you’ve collected enough phantoms at level (i). This introduces a one-step delay but remains online.

**Numerical note:** when (1-r_i) is extremely small, the formula involves subtracting close numbers and dividing by a small number; in that regime, the shell’s prior mass is tiny anyway, and classic trapezoid may be adequate.

---

## 7. Pooling looser samples to estimate shrinkage (optional)

Sometimes level (i-1) has few phantoms; you can pool samples from any looser constraint (\ell^*\le \ell_{i-1}) using conditioning.

Let pooled samples be (\theta_m\sim\pi_{\ell_m^*}) with (\ell_m^*\le \ell_{i-1}). Define:
[
A_m=\mathbf 1{L(\theta_m)>\ell_{i-1}},\quad
B_m=\mathbf 1{L(\theta_m)>\ell_i}.
]
Then:
[
r_i = \frac{X(\ell_i)}{X(\ell_{i-1})} = \frac{\mathbb E[B]}{\mathbb E[A]},
]
and a pooled estimator is:
[
\hat r_i = \frac{\sum_m B_m}{\sum_m A_m}.
]

**Rules:**

* only include sources with (\ell^*\le \ell_{i-1})
* never include samples generated at tighter constraints (\ell^*>\ell_{i-1}) (they bias upward)
* ESS for pooled ratio is most safely computed per independent source/fragment after filtering, then aggregated conservatively.

Given the complexity of accurate ESS for ratios, the simplest stable approach is:

* use pooling for point estimate (\hat r_i),
* use a conservative (M_{\mathrm{eff}}) (e.g. sum of fragment ESS on (A_m)-filtered subsequences, clipped).

---

## 8. Posterior sample output without storing everything: exponential-race reservoir

Goal: maintain a bounded-memory set of (R) representative posterior samples from a stream of weighted candidates whose *unnormalized* weights become available online.

### 8.1 Candidate weights

When shell (i) closes, its prior mass (\mu(\Delta_i)) becomes known (from (X_{i-1},X_i)).

If we have a set of (M_i) candidate points ({\theta_{i,1},\dots,\theta_{i,M_i}}) intended to represent shell (i) (e.g. dead point alone; or dead point + some shell-filtered phantoms), assign each a prior-mass share:
[
\mu_{i,m} := \frac{\mu(\Delta_i)}{M_i}.
]
Each candidate then has **unnormalized posterior weight**
[
w_{i,m} := \mu_{i,m},L(\theta_{i,m}) = \frac{\mu(\Delta_i)}{M_i},L(\theta_{i,m}).
]
Note: normalization by total evidence (Z) is not needed for reservoir selection (global scale cancels).

### 8.2 Exponential-race key

For each candidate with weight (w>0), draw (u\sim\mathrm{Unif}(0,1)) and compute:
[
t := \frac{-\log u}{w}.
]
Maintain the (R) candidates with **smallest** (t). This implements weighted sampling without replacement from the candidate stream, using only unnormalized weights.

Online update:

* if reservoir has < (R): insert (candidate, (t))
* else if (t < t_{\max}) in reservoir: replace the max-(t) item

Data structure: max-heap keyed by (t).

### 8.3 When weights are unknown until later

If a candidate point is generated before (\mu(\Delta_i)) is known, store it as “pending” with a cached (u) (or (-\log u)). Once (\mu(\Delta_i)) is finalized, compute (w) and then (t), and ingest.

### 8.4 What candidates to use per shell

Three increasing-storage options:

1. **Dead-point only** (classic): (M_i=1), (\theta_{i,1}=) discarded point at (\ell_i). Lowest memory, standard NS behavior.

2. **Dead point + a few shell phantoms**: attempt to collect (M_i-1) phantoms with (\ell_{i-1}<L\le \ell_i). If none, fall back to dead point only.

3. **Dedicated shell sampling** (expensive): if you require true shell-distributed samples even when shells are tiny, run rejection sampling from (\pi_{\ell_{i-1}}) into the shell; cost grows like (1/(1-r_i)).

Recommendation: (1) or (2). Shells with (1-r_i) extremely small contribute little prior mass and usually don’t deserve extra sampling effort.

---

## 9. Online control flow (sequential discards)

Here is the high-level event ordering for a sequential NS run:

### State kept per level (i)

* threshold (\ell_i)
* live count (K_{i+1}) before the next discard (for the next prior)
* phantom accumulator at level (i), built while sampling replacements under constraint (\ell_i):

  * indicator buffer for future exceedances: store phantom (L)-values (or keep counts once the next (\ell) is known)
  * running means (m_i(f)) for chosen functions (f) (e.g. (L), (gL))
  * ESS diagnostics for indicator sequences (batching metadata) if used

### On discard event (i) (discover (\ell_i))

1. **Finalize shrinkage (r_i)** using phantoms from level (i-1):

   * compute (I_{i,m}=\mathbf 1{L(\theta_{i-1,m})>\ell_i})
   * compute (\hat r_i), (M_{\mathrm{eff},i}), then posterior Beta((\alpha_i,\beta_i))
   * sample (r_i^{(b)}) (sampling-based) or compute moments (expectation-based)

2. **Update volumes**:

   * (X_i = X_{i-1} r_i) (per replicate) and shell mass (\mu(\Delta_i)=(1-r_i)X_{i-1})

3. **Finalize shell contribution and ingest reservoir candidates**:

   * evidence: choose either classic quadrature or phantom-based shell mean (direct hits or mixture with lag)
   * posterior reservoir: assign candidate weights (w_{i,m}) and ingest (dead point always available; shell phantoms optional)

4. **Start accumulating phantoms for level (i)** while sampling replacements under constraint (\ell_i).

This design ensures phantoms affect shrinkage and quadrature **without altering the discard schedule** (i.e. no invariance issues from schedule feedback).

---

## 10. Plateau handling (important limitation)

Phantom exceedance estimation of (r_i) uses the identity
[
r_i = \mathbb P_{\pi_{\ell_{i-1}}}(L>\ell_i).
]
This requires (\ell_i) to be a meaningful strict threshold.

If you have **plateaus** where many discarded points share identical (L), then:

* exceedance indicators for equal thresholds become uninformative (naively gives (\hat r=1)),
* but the prior mass still shrinks across plateau ranks.

Therefore:

* within plateau blocks, rely on the race/order-statistic prior (or plateau permutation marginalization) for shrinkage,
* apply phantom shrinkage updates only across **strict increases** in (\ell) (plateau boundaries),
* or inject a deterministic tie-breaking scheme that effectively assigns distinct thresholds for mass accounting while keeping true (L) for evidence (implementation-specific).

---

## 11. Summary of “what this buys”

* **Shrinkage uncertainty** becomes data-informed: Beta((K,1)) prior is updated using phantom exceedances with an ESS correction.
* **Quadrature** can be improved without changing schedule:

  * replace model-based shrinkage with phantom-estimated (\hat r_i),
  * optionally estimate shell means using phantom statistics (direct shell hits or mixture identity).
* **Posterior sample output** can be bounded-memory using exponential-race reservoir sampling, with weights that do not require knowing (Z) in advance.
