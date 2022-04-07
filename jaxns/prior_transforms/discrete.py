from jax import numpy as jnp
from jax.scipy.special import logsumexp
from jax.lax import while_loop

from jaxns.internals.types import float_type, int_type
from jaxns.prior_transforms import (DeterministicTransformPrior, prior_docstring, Gumbel, get_shape, UniformBase, HierarchicalPrior, UniformPrior)
from jaxns.internals.shapes import broadcast_dtypes


class PoissonPrior(HierarchicalPrior):
    @prior_docstring
    def __init__(self, name, lamda, tracked=True):
        """
        Uses a sequential transform which works for small lamda.
        Empirical mean within 1 for lamda < 100.

        Args:
            lamda: the intensity of the process.
        """
        lamda = self._prepare_parameter(name, 'lamda', lamda)
        prior_base = UniformBase(get_shape(lamda), broadcast_dtypes(float_type, lamda.dtype))

        super(PoissonPrior, self).__init__(name, [lamda], tracked=tracked, prior_base=prior_base)

    def transform_U(self, U, lamda, **kwargs):
        """
        Algorithmic Poisson generator based upon the inversion by sequential search:

            init:
                Let log_x ← -inf, log_p ← −λ, log_s ← log_p.
                Generate uniform random number u in [0,1].
            while log_u > log_s do:
                log_x ← logaddexp(log_x, 0).
                log_p ← log_p + log_λ - log_x.
                log_s ← logaddexp(log_s, log_p).
            return exp(log_x).

        Args:
            U:
            lamda:
            **kwargs:

        Returns:

        """
        U = jnp.clip(U, 1e-8, 1. - 1e-8)
        U, lamda = jnp.broadcast_arrays(U, lamda)
        log_u = jnp.log(U)
        log_x = jnp.full(log_u.shape, -jnp.inf)
        log_p = -lamda
        log_s = log_p

        def body(state):
            (_log_x, _log_p, _log_s) = state
            impute = log_u <= _log_s
            log_x = jnp.logaddexp(_log_x, 0.)
            log_p = _log_p + jnp.log(lamda) - log_x
            log_s = jnp.logaddexp(_log_s, log_p)

            log_x = jnp.where(impute, _log_x, log_x)
            log_p = jnp.where(impute, _log_p, log_p)
            log_x = jnp.where(impute, _log_x, log_x)

            return (log_x, log_p, log_s)

        (log_x, log_p, log_s) = while_loop(lambda s: jnp.any(log_u > s[2]),
                                           body,
                                           (log_x, log_p, log_s))
        return jnp.exp(log_x).astype(int_type)

class CategoricalPrior(DeterministicTransformPrior):
    @prior_docstring
    def __init__(self, name, logits, tracked=True):
        """
        Categorical variable, which uses the GumbelMax reparametrisation to sample a discrete variable.

        Args:
            logits: the log-probability (unnormalised) of the outcomes.
        """
        logits = self._prepare_parameter(name, 'logits', logits)
        shape = get_shape(logits)[:-1]
        uniform = UniformPrior('_{}_uniform'.format(name), jnp.zeros(shape), jnp.ones(shape) , False)
        def _transform(u, logits):
            # normalise logits
            logits -= logsumexp(logits, axis=-1, keepdims=True)
            # we take the index which the uniform variable falls in in the cumulative mass array
            # 1-d
            # if len(logits.shape) == 1:
            #     cumulative_mass = cumulative_logsumexp(logits)
            #     return jnp.clip(jnp.searchsorted(cumulative_mass, jnp.log(u), side='right') - 1, 0, logits.shape[-1])
            # else:
            # logits [..., N]
            log_u = jnp.log(u)
            # parallel CDF sampling
            def body(state):
                #done [...]
                #accumulant [...]
                #i []
                #output [...]
                #u [...]
                (_, accumulant, i, output) = state
                new_accumulant = jnp.logaddexp(accumulant, logits[..., i]) #[...]
                done = log_u < new_accumulant # [...]
                output = jnp.where(done, output, i+1) # [...]
                return (done, new_accumulant, i + 1, output)

            loop_vars = (
                jnp.zeros(logits.shape[:-1], dtype=jnp.bool_),
                -jnp.inf * jnp.ones(logits.shape[:-1], dtype=u.dtype),
                0,
                jnp.zeros(logits.shape[:-1], dtype=int_type)
            )
            (_, _, _, output) = while_loop(lambda state: ~jnp.all(state[0]),
                              body,
                              loop_vars)
            return output

        super(CategoricalPrior, self).__init__(name, _transform, uniform, logits, tracked=tracked)


class BernoulliPrior(CategoricalPrior):
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
                                            p, tracked=False)
        super(BernoulliPrior, self).__init__(name, log_p, tracked)

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

        super(GumbelCategoricalPrior, self).__init__(name, _transform, gumbel, logits, tracked=tracked)


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
                                            p, tracked=False)
        super(GumbelBernoulliPrior, self).__init__(name, log_p, tracked)
