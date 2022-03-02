from collections import namedtuple, OrderedDict
from jax import numpy as jnp, tree_multimap
from jax.lax import scan
from jax.flatten_util import ravel_pytree
from jaxns.utils import dict_multimap, tuple_prod
from jaxns.log_math import signed_logaddexp

LogParam = namedtuple('LogParam', ['log_value'])
SignedLogParam = namedtuple('SignedLogParam', ['log_abs_value', 'sign'])
ParamTrackingState = namedtuple("ParamTrackingState", ['f', 'f2', 'fX', 'X', 'X2', 'w', 'w2', 'L_i1', 'dw'])


class TrackedExpectation(object):
    def __init__(self,
                 *,
                 marginalised_funcs=None,
                 test_sample=None,
                 dtype=None
                 ):
        """
        Tracks marginalised functions including evidence and information gain.

        Args:
            state: ParamTrackingState or None
            *
            marginalised_funcs: dict of callable(**samples)
            test_sample: dict of jnp.ndarray which are exmaple samples
            dtype: dtype of log-likelihood or None which is jnp.float_
        """
        if dtype is None:
            self._dtype = jnp.float_
        self.State = ParamTrackingState
        self.marginalised_funcs = marginalised_funcs
        self.meta_funcs, example_meta = self.build_meta(test_sample, marginalised_funcs)

        initial_f = SignedLogParam(-jnp.inf * jnp.ones_like(example_meta),
                                   jnp.ones_like(example_meta))
        initial_f2 = initial_f
        initial_fX = initial_f
        initial_X = LogParam(jnp.zeros((), self._dtype))
        initial_X2 = initial_X
        initial_w = LogParam(-jnp.inf*jnp.ones((), self._dtype))
        initial_w2 = initial_w
        initial_L_i1 = initial_w
        initial_dw = initial_w

        self.state = self.State(f=initial_f,
                                f2=initial_f2,
                                fX=initial_fX,
                                X=initial_X,
                                X2=initial_X2,
                                w=initial_w,
                                w2=initial_w2,
                                L_i1=initial_L_i1,
                                dw=initial_dw)
        self._dirty = True

    def build_meta(self, test_sample, marginalised_funcs):
        def build_marg_func(func):
            def marg_func(posterior_sample, n_i, log_L_i):
                return func(**posterior_sample)
            return marg_func

        meta_funcs = dict()
        for key in marginalised_funcs:
            meta_funcs[key] = build_marg_func(marginalised_funcs[key])
        meta_funcs["static:Z"] = lambda posterior_sample, n_i, log_L_i: jnp.ones((), self._dtype)
        meta_funcs["static:H"] = lambda posterior_sample, n_i, log_L_i: log_L_i

        meta_values = dict((k, f(test_sample, 1, jnp.zeros((), self._dtype))) for k,f in meta_funcs.items())
        example_meta, self.unravel_func = ravel_pytree(meta_values)

        return meta_funcs, example_meta

    def lookup_meta(self, name):
        """
        Get an item from the unravelled state.

        Args:
            name: key to get

        Returns: State with state replaced by specific item
        """
        if self._dirty:
            self._meta = self.state._replace(
            f=self.unravel_func(self.state.f),
            f2=self.unravel_func(self.state.f2),
            fX=self.unravel_func(self.state.fX)
        )
            self._dirty = False
        return self._meta._replace(f=self._meta.f[name],
                                   f2=self._meta.f2[name],
                                   fX=self._meta.fX[name]
                                  )

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._dirty = True  # need to unravel later, else already unravelled
        self._state = self.State(*state)

    def effective_sample_size(self):
        """Kish's ESS = [sum weights]^2 / [sum weights^2]
        """
        return jnp.exp(2. * self.state.w.log_value - self.state.w2.log_value)

    @property
    def enclosed_prior_mass_mean(self):
        """
        The current log(E[X])
        """
        return self.state.X.log_value

    @property
    def enclosed_prior_mass_variance(self):
        """
        The current log(Var[X])
        Returns:

        """
        #jnp.log(jnp.exp(self.state.X2.log_value) - jnp.exp(self.state.X.log_value)**2)
        return signed_logaddexp(self.state.X2.log_value, 1., 2.*self.state.X.log_value, -1.)

    def evidence_mean(self):
        return self._log_mean(*self.lookup_meta('static:Z'), normalised=False)

    def evidence_variance(self):
        return self._log_variance(*self.lookup_meta('static:Z'), normalised=False)

    def information_gain_mean(self):
        return -self._linear_mean(*self.lookup_meta('static:H'), normalised=True) + self.evidence_mean()

    def information_gain_variance(self):
        return self._linear_variance(*self.lookup_meta('static:H'), normalised=True) + self.evidence_variance()

    def marg_mean(self):
        d = OrderedDict()
        for key in self.marginalised_funcs.keys():
            d[key] = self._linear_mean(self.lookup_meta(key), normalised=True)
        return d

    def marg_variance(self):
        d = OrderedDict()
        for key in self.marginalised_funcs.keys:
            d[key] = self._linear_variance(self.lookup_meta(key), normalised=True)
        return d

    def _linear_mean(self, state:ParamTrackingState, normalised: bool):
        if normalised:
            return state.f.sign * jnp.exp(state.f.log_abs_value - state.w.log_value)
        else:
            return state.f.sign * jnp.exp(state.f.log_abs_value)

    def _log_mean(self, state: ParamTrackingState, normalised: bool):
        # lambda log_f, log_f2: (2. * log_f - 0.5 * log_f2) - self.w.log_value
        if normalised:
            return (2. * state.f.log_abs_value - 0.5 * state.f2.log_abs_value) - state.w.log_value
        else:
            return 2. * state.f.log_abs_value - 0.5 * state.f2.log_abs_value

    def _linear_variance(self, state:ParamTrackingState, normalised: bool):
        if normalised:
            # lambda f, f2: (f2/ w - (f/w) ** 2)
            t1, t1_sign = signed_logaddexp(state.f2.log_abs_value - state.w.log_value,
                                           state.f2.sign,
                                           2. * (state.f.log_abs_value - state.w.log_value),
                                           -1.)
            return t1_sign * jnp.exp(t1)
        else:
            t1, t1_sign = signed_logaddexp(
                state.f2.log_abs_value,
                state.f2.sign,
                2. * state.f.log_abs_value,
                -1.)
            return t1_sign * jnp.exp(t1)

    def _log_variance(self, state: ParamTrackingState, normalised: bool):
        if normalised:
            # (log_f2 - 2. * log_f) - self.w.log_value
            return state.f2.log_abs_value - 2. * state.f.log_abs_value - state.w.log_value
        else:
            return state.f2.log_abs_value - 2. * state.f.log_abs_value

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i) -> SignedLogParam:
        """
        Computes the log(f) for each func in meta.

        Each meta func is callable(sample : dict, n : int, log_L : float) -> ndarray

        Args:
            posterior_sample: dict of (key, ndarray)
            n_i: int
            log_L_i: float

        Returns:
            SignedLogParam with the value of log(flatten(f))
        """
        # use meta data to compute
        res = dict((k, f(posterior_sample, n_i, log_L_i)) for k, f in self.meta_funcs.items())
        example_meta, _ = ravel_pytree(res)
        return SignedLogParam(jnp.log(jnp.abs(res)), jnp.sign(res))

    def update_from_live_points(self, live_points, log_L_live, is_satisfying=None, num_likelihood_evals=None):
        """
        Update tracked state from set of points.

        Args:
            live_points: dict of (key, array) where the key is RV name, and array of RV sampled at that point, first
            dimension indexing sample index.
            log_L_live: array of log-likelihood associated with each live-point.
            is_satisfying: array of whether each sample in live-points can be pulled from.
            num_likelihood_evals: array of num-likelihood evals required for each sample in live-points.

        Returns:
            List of (n, log_L_min, self.state.X.log_value, self.state.dw.log_value, n_evals, x_min)
            where each is an array of appropriate type.
        """
        if is_satisfying is None:
            is_satisfying = jnp.ones(log_L_live.shape, dtype=jnp.bool_)
        if num_likelihood_evals is None:
            num_likelihood_evals = jnp.ones(log_L_live.shape)
        ar = jnp.argsort(log_L_live)
        num_satisfying = jnp.sum(is_satisfying)
        # We offset numbering by num_satisfying

        def body(state, X):
            (self_state,found_satisfying) = state
            (i,) = X
            i_min = ar[i]
            x_min = dict_multimap(lambda x: x[i_min, ...], live_points)
            log_L_min = log_L_live[i_min]
            self.state = self_state
            n = jnp.where(is_satisfying[i_min], num_satisfying - found_satisfying, jnp.inf)
            found_satisfying = jnp.where(is_satisfying[i_min], found_satisfying+1, found_satisfying)
            self.update(x_min, n, log_L_min)
            n_evals = num_likelihood_evals[i_min]
            return (self.state, found_satisfying), (n, log_L_min, self.state.X.log_value, self.state.dw.log_value, n_evals, x_min)

        (self_state, _), results = scan(body,(self.state,jnp.asarray(0)), (jnp.arange(log_L_live.shape[0]),),unroll=1)
        # print(results)
        self.state = self_state
        return results

    def update(self, posterior_sample_i, n_i, log_L_i):
        """
        Update tracked state.

        Args:
            posterior_sample_i: dict, sample of dead point.
            n_i: number of live-points sampled from.
            log_L_i: log-likelihood of dead point
        """

        #TODO: propagate Var(X) with product of RV method to keep positive definite. And E[X]. Can reconstruct E[X^2] easily.
        L_i1 = self.state.L_i1
        f_i1 = self.state.f
        f2_i1 = self.state.f2
        fX_i1 = self.state.fX
        X_i1 = self.state.X
        X2_i1 = self.state.X2
        w_i1 = self.state.w
        w2_i1 = self.state.w2
        f_alpha_i = self.compute_log_f_alpha(posterior_sample_i, n_i, log_L_i)

        def _maybe_replace(replace, v_i, v_i1):
            if isinstance(v_i, SignedLogParam):
                return SignedLogParam(log_abs_value=jnp.where(replace, v_i.log_abs_value, v_i1.log_abs_value),
                                      sign=jnp.where(replace, v_i.sign, v_i1.sign))
            elif isinstance(v_i, LogParam):
                return LogParam(jnp.where(replace, v_i.log_value, v_i1.log_value))
            else:
                raise ValueError("Type {} invalid".format(type(v_i)))

        replace = ~jnp.isinf(n_i)

        log_triangle = jnp.logaddexp(L_i1.log_value, log_L_i) - jnp.log(2.)

        n_plus_1 = n_i + 1.
        n_plus_2 = n_i + 2.

        # w_i = w_i1 + X_i1 * triangle /(num_options+1)
        log_dw_i = X_i1.log_value + log_triangle - jnp.log(n_plus_1)
        w_i = LogParam(jnp.logaddexp(w_i1.log_value, log_dw_i))

        w2_i = LogParam(jnp.logaddexp(w2_i1.log_value, 2. * log_dw_i))
        # X_i = X_i1 * n_n_plus_1
        X_i = LogParam(X_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_1))
        # X2_i = n_n_plus_2 * X2_i1
        X2_i = LogParam(X2_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_2))

        def _log_f_i(f_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # f_i = f_i1 + X_i1 * triangle * r_n_plus_1 * f_alpha_i
            return SignedLogParam(*signed_logaddexp(f_i1.log_abs_value, f_i1.sign,
                                                    X_i1.log_value + log_triangle + f_alpha_i.log_abs_value - jnp.log(
                                                        n_plus_1),
                                                    f_alpha_i.sign))

        f_i = _log_f_i(f_i1, f_alpha_i)

        def _log_f2_i(f2_i1: SignedLogParam, fX_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # f2_i = f2_i1 + fX_i1 * two * r_n_plus_1 * triangle + X2_i1 * r_n_plus_1 * r_n_plus_2 * (triangle * f_alpha_i) ** 2
            t1 = fX_i1.log_abs_value + jnp.log(2.) - jnp.log(n_plus_1) + log_triangle
            t1_sign = fX_i1.sign
            t2 = X2_i1.log_value - jnp.log(n_plus_1) - jnp.log(n_plus_2) + 2. * (log_triangle + f_alpha_i.log_abs_value)
            t2_sign = 1.
            t3, t3_sign = signed_logaddexp(t1, t1_sign, t2, t2_sign)
            return SignedLogParam(*signed_logaddexp(f2_i1.log_abs_value, f2_i1.sign, t3, t3_sign))

        f2_i = _log_f2_i(f2_i1, fX_i1, f_alpha_i)

        def _log_fX_i(fX_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # fX_i = n_n_plus_1 * fX_i1 + n_n_plus_2_n_plus_1 * X2_i1 * (triangle * f_alpha_i)
            t1 = fX_i1.log_abs_value + jnp.log(n_i) - jnp.log(n_plus_1)
            t1_sign = fX_i1.sign
            t2 = X2_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_1) - jnp.log(n_plus_2) \
                 + log_triangle + f_alpha_i.log_abs_value
            t2_sign = f_alpha_i.sign
            return SignedLogParam(*signed_logaddexp(t1, t1_sign, t2, t2_sign))

        fX_i = _log_fX_i(fX_i1, f_alpha_i)

        self.state = self.State(f=_maybe_replace(replace, f_i, f_i1),
                                X=_maybe_replace(replace,X_i, X_i1),
                                f2=_maybe_replace(replace, f2_i, f2_i1),
                                fX=_maybe_replace(replace,fX_i,fX_i1),
                                X2=_maybe_replace(replace,X2_i,X2_i1),
                                w=_maybe_replace(replace,w_i,w_i1),
                                w2=_maybe_replace(replace,w2_i,w2_i1),
                                L_i1=_maybe_replace(replace, LogParam(log_L_i), L_i1),
                                dw=LogParam(log_dw_i))
