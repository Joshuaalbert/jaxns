from jax import numpy as jnp, random, jit, vmap
from jaxns.nested_sampling import NestedSampler, save_results, load_results
from jaxns.prior_transforms import UniformPrior, PriorChain, NormalPrior
from jaxns.plotting import plot_diagnostics, plot_cornerplot
from jaxns.utils import summary, resample
from itertools import combinations


def get_constraints(num_options, num_raters, tests_per_rater, rater_accuracy):
    key = random.PRNGKey(47573957)
    actual_rank = random.uniform(random.PRNGKey(324574),shape=(num_options,),minval=0., maxval=5.)

    pairs = jnp.asarray(list(combinations(range(num_options), 2)), dtype=jnp.int_)
    I = []
    J = []
    S = []
    for rater in range(num_raters):
        key, sample_key1, sample_key2 = random.split(key, 3)
        choices = random.choice(sample_key1,pairs.shape[0], shape=(tests_per_rater,), replace=False)
        I.append(pairs[choices,0])
        J.append(pairs[choices,1])
        S.append((actual_rank[I[-1]] - actual_rank[J[-1]]) >
                 rater_accuracy * random.normal(sample_key2, shape=(tests_per_rater,)))

    return actual_rank, jnp.concatenate(I),jnp.concatenate(J),jnp.concatenate(S)

def main(num_options=10, num_raters=5, tests_per_rater=10, rater_accuracy=1):
    actual_rank, I, J, S = get_constraints(num_options, num_raters, tests_per_rater, rater_accuracy)
    def log_likelihood(rank, **kwargs):
        score_ij = rank[I] > rank[J]
        violations = jnp.sum(score_ij != S)
        return -violations

    rank = UniformPrior('rank', jnp.zeros(num_options), 5*jnp.ones(num_options))
    prior_chain = rank.prior_chain()

    ns = NestedSampler(loglikelihood=log_likelihood, prior_chain=prior_chain)
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