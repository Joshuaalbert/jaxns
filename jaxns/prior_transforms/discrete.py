from jax import numpy as jnp

from jaxns.prior_transforms import DeterministicTransformPrior, prior_docstring, Gumbel, get_shape

class GumbelCategoricalPrior(DeterministicTransformPrior):
    @prior_docstring
    def __init__(self, name, logits, tracked=True):
        """
        Categorical variable, which uses the GumbelMax reparametrisation to sample a discrete variable.

        Args:
            logits: the log-probability (unnormalised) of the outcomes.
        """
        logits = self._prepare_parameter(name, 'logits', logits)
        gumbel = Gumbel('_{}_gumbel'.format(name), get_shape(logits), False)

        def _transform(gumbel, logits):
            return jnp.argmax(logits + gumbel, axis=-1)

        shape = get_shape(logits)[:-1]
        super(GumbelCategoricalPrior, self).__init__(name, _transform, shape, gumbel, logits, tracked=tracked,
                                                     dtype=jnp.int_)


class GumbelBernoulliPrior(GumbelCategoricalPrior):
    @prior_docstring
    def __init__(self, name, p, tracked=True):
        """
        Bernoulli distribution, using the GumbelMax trick.

        Y ~ B[p]
        Args:
            p: prob of the event
        """
        p = self._prepare_parameter(name, 'p', p)
        log_p = DeterministicTransformPrior('_log_{}'.format(p.name),
                                            lambda p: jnp.stack([jnp.log(1. - p), jnp.log(p)], axis=-1),
                                            get_shape(p) + (2,), p, tracked=False)
        super(GumbelBernoulliPrior, self).__init__(name, log_p, tracked)
