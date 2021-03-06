from jax import numpy as jnp, random, jit, vmap
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import UniformPrior, PriorChain, NormalPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample


def get_constraints(num_options, num_raters, tests_per_rater, rater_accuracy):
    key = random.PRNGKey(47573957)
    actual_rank = random.uniform(random.PRNGKey(324574),shape=(num_options,),minval=0., maxval=5.)

    I = []
    J = []
    R = []
    S = []
    for rater in range(num_raters):
        for test in range(tests_per_rater):
            key, sample_key1, sample_key2 = random.split(key, 3)
            i,j = random.choice(sample_key1,num_options, shape=(2,), replace=False)
            score_ij = (actual_rank[i] - actual_rank[j]) > rater_accuracy * random.normal(sample_key2)
            I.append(i)
            J.append(j)
            S.append(score_ij)
            R.append(rater)
    return actual_rank, jnp.asarray(I, dtype=jnp.int_),jnp.asarray(J, dtype=jnp.int_),jnp.asarray(S, dtype=jnp.float_), jnp.asarray(R,dtype=jnp.int_)

def main(num_options=10, num_raters=10, tests_per_rater=3, rater_accuracy=1):
    actual_rank, I, J, S, R = get_constraints(num_options, num_raters, tests_per_rater, rater_accuracy)
    def log_likelihood(rank, **kwargs):
        score_ij = rank[I] > rank[J]
        violations = jnp.sum(score_ij != S)
        return -violations

    rank = UniformPrior('rank', jnp.zeros(num_options), 5*jnp.ones(num_options))
    prior_chain = rank.prior_chain()

    print('Number of live points', prior_chain.U_ndims * 20)
    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain,
                       sampler_name='slice', num_parallel_samplers=1,
                       sampler_kwargs=dict(depth=5, num_slices=prior_chain.U_ndims),
                       num_live_points=prior_chain.U_ndims*50, max_samples=1e6, collect_samples=True,
                       collect_diagnostics=True)
    results = jit(ns)(random.PRNGKey(32564), termination_frac=0.001)
    save_results(results, 'ranking_save.npz')
    results = load_results('ranking_save.npz')

    summary(results)

    plot_diagnostics(results)
    plot_cornerplot(results, vars=['rank'])

    samples = resample(random.PRNGKey(245944),results.samples, results.log_p, S=int(results.ESS))

    posterior_rank = jnp.argsort(jnp.median(samples['rank'], axis=0))
    true_rank = jnp.argsort(actual_rank)
    for i in range(num_options):
        print("True rank = {}, posterior rank = {}".format(true_rank[i],posterior_rank[i]))


if __name__ == '__main__':
    main()