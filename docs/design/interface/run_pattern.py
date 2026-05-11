import jax.random
import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jaxctx import CtxParams
from jaxctx.priors.prior import Prior

from jaxns.core import NestedSampler
from jaxns.model import Model
from jaxns.phantom_eval import EvidenceSamples
from jaxns.runtime import LoadBalancerClient
from jaxns.state import State
from jaxns.termination_condition import TerminationCondition

tfpd = tfp.distributions


def prior_model(a, b):
    x = Prior(tfpd.Uniform(0., a), name='x').realise()
    y = Prior(tfpd.Uniform(0., b), name='y').parameter()
    return x.sum() + y


model = Model(prior_model=prior_model)
model_args = (1., 2.)  # passed into prior model; must be pickleable or a pytree.
model_params: CtxParams = model.init_params(
    key=jax.random.PRNGKey(0),
    args=model_args
)  # Parameters are the values constructed with .parameter().


def goal_cond(state: State) -> bool:
    result = state.to_result()
    return result.log_Z_uncert < 0.1


# A client for LB is what enables adding node-managed workers.
# Workers are process-isolated likelihood evaluators with overlapping computation.
# LB is what enables creating a nested sampler object that can run NS over the workers.
# When address is 'local' it finds or starts a local LB, connects to it, and
# tears down only resources owned by this context on exit.
with LoadBalancerClient(address='local') as lb:
    # Add workers on the current host. This creates or reuses a local node
    # process manager, which owns one node ingress/coordinator process plus the
    # worker processes. The node ingress talks to the LB; the node coordinator
    # fans out to local workers over random ipc:// endpoints under /tmp.
    # syntax is `device_type:device_ids:num_workers_per_device`
    lb.add_workers(['cpu:*:5', 'gpu:0,1:10'])
    # This adds:
    # - to each (* means all) CPU 5 workers (most machines have a single CPU, but some have 2)
    # - to GPUs 0 and 1 10 workers each.

    ns: NestedSampler = lb.get_nested_sampler(
        model=model,
        args=model_args,
        params=model_params,
        collect_phantoms=True  # collect phantom likelihoods, discarding points
    )
    # get_nested_sampler registers the compile identity and creates an isolated
    # runner. Workers JIT/cache the likelihood on first matching likelihood-eval
    # work, then return scalar log_L results to the original runner. Many
    # clients can share a worker pool across different likelihood problems;
    # fair sharing is managed by the LB at node-advertised capacity.
    state: State = ns.run_until_goal(
        goal_cond=goal_cond,
        depth_cond=TerminationCondition(),
        allocation_target='uniform'
    )
    # Resuming is possible
    state: State = ns.resume_until_goal(
        state=state,
        goal_cond=goal_cond,
        depth_cond=TerminationCondition(),
        allocation_target='evidence_improving'
    )
    # Results contain expectation based results
    results = state.to_result()
    # Expectation based evaluation of evidence for quick understanding
    print(results.log_Z_mean, results.log_Z_uncert)
    # Inspection is possible
    results.summary()
    results.plot_diagnostics()
    results.plot_cornerplot()




    # Integrating anything over the posterior is possible
    def some_fn(model_params: CtxParams):
        return {'x2': model_params['x'] ** 2, 'y2': model_params['y'] ** 2}


    expected_post_predictive = results.integrate_fn_over_posterior(
        some_fn,
        semi_positive=True
    )
    # Model comparison science should be done with sampling shrinkage
    shrinkage_samples: EvidenceSamples = results.sample_mc_shrinkage(
        num_samples=1000
    )
    log_Z_samples = shrinkage_samples.log_Z_samples  # [num_samples]
    phantom_diag = results.phantom_conditioning_diagnostics()
    # Target diagnostics for phantom-conditioned shrinkage:
    # - participating independent-cluster count used by the Kish gate;
    # - whether phantom counts were activated for each block.
    plt.plot(
        results.log_L_blocks,
        phantom_diag.kish_participating_cluster_counts
    )
    plt.step(
        results.log_L_blocks,
        phantom_diag.phantom_gate_active.astype(float),
        where='post'
    )
    plt.show()


### This is what a worker-node script might look like for joining nodes.

try:
    with LoadBalancerClient(address='tcp://123.123.123.123:5555') as lb:
        # Add a node process manager and workers on the current host. The
        # remote LB sees one node ingress/coordinator, not every worker socket.
        # syntax is `device_type:device_ids:num_workers_per_device`
        lb.add_workers(['cpu:*:5', 'gpu:0,1:10'])
        lb.wait_until_shutdown()
except KeyboardInterrupt:
    print("Shutting down LB client and workers...")
