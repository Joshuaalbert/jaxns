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