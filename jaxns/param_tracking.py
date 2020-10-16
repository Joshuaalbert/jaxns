from collections import namedtuple
from jax import jit, numpy as jnp
from jax.lax import while_loop
from jaxns.utils import dict_multimap, swap_dict_namedtuple, swap_namedtuple_dict, signed_logaddexp

LogParam = namedtuple('LogParam', ['log_value'])
SignedLogParam = namedtuple('SignedLogParam', ['log_abs_value', 'sign'])
ParamTrackingState = namedtuple("ParamTrackingState", ['f', 'f2', 'fX', 'X', 'X2', 'w', 'L_i1'])

class TrackedExpectation(object):
    def __init__(self, name,
                 type,
                 normalised: bool,
                 param_shape,
                 *,
                 state=None):
        self.normalised = normalised
        if type.lower() not in ['log', 'linear']:
            raise ValueError("Type {} invalid.".format(type))
        self.type = type
        self.name = name
        self.param_shape = param_shape
        self.State = ParamTrackingState
        if state is not None:
            self.state = state
        else:
            initial_f = SignedLogParam(dict_multimap(lambda s: -jnp.inf * jnp.ones(s), param_shape),
                                       dict_multimap(lambda s: jnp.ones(s), param_shape))
            initial_f2 = initial_f
            initial_fX = initial_f
            initial_X = LogParam(-jnp.inf)
            initial_X2 = initial_X
            initial_w = initial_X
            initial_L_i1 = initial_X

            self.state = self.State(f=initial_f,
                                    f2=initial_f2,
                                    fX=initial_fX,
                                    X=initial_X,
                                    X2=initial_X2,
                                    w=initial_w,
                                    L_i1=initial_L_i1)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = self.State(*state)

    def _linear_mean(self, normalised: bool):
        def _mean(f: SignedLogParam):
            if normalised:
                return f.sign * jnp.exp(f.log_abs_value - self.state.w.log_value)
            else:
                return f.sign * jnp.exp(f.log_abs_value)

        return dict_multimap(_mean, swap_dict_namedtuple(self.state.f))

    def _log_mean(self, normalised: bool):
        def _mean(f: SignedLogParam, f2: SignedLogParam):
            # lambda log_f, log_f2: (2. * log_f - 0.5 * log_f2) - self.w.log_value
            if normalised:
                return (2. * f.log_abs_value - 0.5 * f2.log_abs_value) - self.state.w.log_value
            else:
                return 2. * f.log_abs_value - 0.5 * f2.log_abs_value

        return dict_multimap(_mean, swap_dict_namedtuple(self.state.f), swap_dict_namedtuple(self.state.f2))

    @property
    def mean(self):
        if self.type.lower() == 'linear':
            return self._linear_mean(self.normalised)
        elif self.type.lower() == 'log':
            return self._log_mean(self.normalised)

    def _linear_variance(self, normalised: bool):
        def _variance(f: SignedLogParam, f2: SignedLogParam):
            if normalised:
                # lambda f, f2: (f2 - f ** 2) / self.w.value
                t1, t1_sign = signed_logaddexp(f2.log_abs_value, f2.sign, 2. * f.log_abs_value, -f.sign)
                return t1_sign * jnp.exp(t1 - self.state.w.log_value)
            else:
                t1, t1_sign = signed_logaddexp(f2.log_abs_value, f2.sign, 2. * f.log_abs_value, -f.sign)
                return t1_sign * jnp.exp(t1)

        return dict_multimap(_variance, swap_dict_namedtuple(self.state.f), swap_dict_namedtuple(self.state.f2))

    def _log_variance(self, normalised: bool):
        def _variance(f: SignedLogParam, f2: SignedLogParam):
            if normalised:
                # (log_f2 - 2. * log_f) - self.w.log_value
                return f2.log_abs_value - 2. * f.log_abs_value - self.state.w.log_value
            else:
                return f2.log_abs_value - 2. * f.log_abs_value

        return dict_multimap(_variance, swap_dict_namedtuple(self.state.f), swap_dict_namedtuple(self.state.f2))

    @property
    def variance(self):
        if self.type.lower() == 'linear':
            return self._linear_variance(self.normalised)
        elif self.type.lower() == 'log':
            return self._log_variance(self.normalised)

    def __repr__(self):
        return "{} = {} +- {}".format(
            self.name, dict_multimap(lambda x: jnp.round(x, 4), self.mean),
            dict_multimap(lambda x: jnp.round(jnp.sqrt(x), 4), self.variance))

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i) -> SignedLogParam:
        raise NotImplementedError()

    def update_from_live_points(self, live_points, log_L_live):
        ar = jnp.argsort(log_L_live)

        def body(state):
            (i, self_state) = state
            i_min = ar[i]
            self.state = self_state
            n = log_L_live.shape[0] - i
            self.update(dict_multimap(lambda x: x[i_min, ...], live_points), n, log_L_live[i_min])
            return (i + 1, self.state)

        (_, self_state) = while_loop(lambda state: state[0] < log_L_live.shape[0],
                                     body,
                                     (0, self.state))
        self.state = self_state

    def update(self, posterior_sample_i, n_i, log_L_i):
        L_i1 = self.state.L_i1
        f_i1 = self.state.f
        f2_i1 = self.state.f2
        fX_i1 = self.state.fX
        X_i1 = self.state.X
        X2_i1 = self.state.X2
        w_i1 = self.state.w
        # tuple(*dict)
        f_alpha_i = self.compute_log_f_alpha(posterior_sample_i, n_i, log_L_i)
        log_triangle = jnp.logaddexp(L_i1.log_value, log_L_i) - jnp.log(2.)

        n_plus_1 = n_i + 1.
        n_plus_2 = n_i + 2.

        # w_i = w_i1 + X_i1 * triangle /(n+1)
        log_dw_i = X_i1.log_value + log_triangle - jnp.log(n_plus_1)
        w_i = LogParam(jnp.logaddexp(w_i1.log_value, log_dw_i))
        # X_i = X_i1 * n_n_plus_1
        X_i = LogParam(X_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_1))
        # X2_i = n_n_plus_2 * X2_i1
        X2_i = LogParam(X2_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_2))

        def _log_f_i(f_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # f_i = f_i1 + X_i1 * triangle * r_n_plus_1 * f_alpha_i
            return SignedLogParam(*signed_logaddexp(f_i1.log_abs_value, f_i1.sign,
                                    X_i1.log_value + log_triangle + f_alpha_i.log_abs_value - jnp.log(n_plus_1),
                                    f_alpha_i.sign))

        f_i = swap_namedtuple_dict(
            dict_multimap(_log_f_i, swap_dict_namedtuple(f_i1), swap_dict_namedtuple(f_alpha_i)))

        def _log_f2_i(f2_i1: SignedLogParam, fX_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # f2_i = f2_i1 + fX_i1 * two * r_n_plus_1 * triangle + X2_i1 * r_n_plus_1 * r_n_plus_2 * (triangle * f_alpha_i) ** 2
            t1 = fX_i1.log_abs_value + jnp.log(2.) - jnp.log(n_plus_1) + log_triangle
            t1_sign = fX_i1.sign
            t2 = X2_i1.log_value - jnp.log(n_plus_1) - jnp.log(n_plus_2) + 2. * (log_triangle + f_alpha_i.log_abs_value)
            t2_sign = 1.
            t3, t3_sign = signed_logaddexp(t1, t1_sign, t2, t2_sign)
            return SignedLogParam(*signed_logaddexp(f2_i1.log_abs_value, f2_i1.sign, t3, t3_sign))

        f2_i = swap_namedtuple_dict(
            dict_multimap(_log_f2_i, swap_dict_namedtuple(f2_i1), swap_dict_namedtuple(fX_i1),
                          swap_dict_namedtuple(f_alpha_i)))

        def _log_fX_i(fX_i1: SignedLogParam, f_alpha_i: SignedLogParam):
            # fX_i = n_n_plus_1 * fX_i1 + n_n_plus_2_n_plus_1 * X2_i1 * (triangle * f_alpha_i)
            t1 = fX_i1.log_abs_value + jnp.log(n_i) - jnp.log(n_plus_1)
            t1_sign = fX_i1.sign
            t2 = X2_i1.log_value + jnp.log(n_i) - jnp.log(n_plus_1) - jnp.log(
                n_plus_2) + log_triangle + f_alpha_i.log_abs_value
            t2_sign = f_alpha_i.sign
            return SignedLogParam(*signed_logaddexp(t1, t1_sign, t2, t2_sign))

        fX_i = swap_namedtuple_dict(
            dict_multimap(_log_fX_i, swap_dict_namedtuple(fX_i1), swap_dict_namedtuple(f_alpha_i)))

        self.state = self.State(f=f_i,
                                X=X_i,
                                f2=f2_i,
                                fX=fX_i,
                                X2=X2_i,
                                w=w_i,
                                L_i1=LogParam(log_L_i))


class Evidence(TrackedExpectation):
    def __init__(self, *, state=None):
        super(Evidence, self).__init__('logZ', 'log', False, (),
                                       state=state)

    @property
    def effective_sample_size(self):
        """Kish's ESS = [sum weights]^2 / [sum weights^2]
        """
        return jnp.exp(2. * self.state.f.log_abs_value - self.state.f2.log_abs_value)

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        return SignedLogParam(0., 1.)


class PosteriorFirstMoment(TrackedExpectation):
    """
    Computes the mean of a parameter. If the shape_dict of the parameter is [..., M], then all dimensions except the last
    one are treated as batched dimensions.
    """

    def __init__(self, shape, *, state=None):
        super(PosteriorFirstMoment, self).__init__('m',
                                                   'linear',
                                                   True,
                                                   shape,
                                                   state=state)

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        return posterior_sample

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
        return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
                              dict_multimap(lambda f: jnp.sign(f), f))


class PosteriorSecondMoment(TrackedExpectation):
    def __init__(self, shape, *, state=None):
        shape = dict_multimap(lambda shape: shape + shape[-1:], shape)
        super(PosteriorSecondMoment, self).__init__('M',
                                                    'linear',
                                                    True,
                                                    shape,
                                                    state=state)

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        return dict_multimap(lambda posterior_sample: posterior_sample[..., None] * posterior_sample[..., None, :],
                             posterior_sample)

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
        return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
                              dict_multimap(lambda f: jnp.sign(f), f))


class InformationGain(TrackedExpectation):
    """
    H = int post(x) log(post(x)/prior(x)) dx
    = int L(x) p(x)/Z log(L(x)/Z) dx
    = int L(x) p(x)/Z log(L(x)) dx - log(Z)
    = E(log(L(x))) - log(Z)
    = sum w(x) log(L(x)) - log(Z)

    This produces -H.
    """

    def __init__(self, global_evidence: Evidence, *, state=None):
        super(InformationGain, self).__init__('H', 'linear', True, (),
                                              state=state)
        self.global_evidence = global_evidence

    @property
    def mean(self):
        mean = -super(InformationGain, self).mean + self.global_evidence.mean
        return mean

    @property
    def variance(self):
        variance = super(InformationGain, self).variance + self.global_evidence.variance
        return variance

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        """
        Args:
            posterior_sample:
            n_i:
            log_L_i:
            from_U:
        Returns:
        """
        return log_L_i

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        return SignedLogParam(jnp.log(jnp.abs(log_L_i)), jnp.sign(log_L_i))


class Marginalised(TrackedExpectation):
    def __init__(self, func_dict, shape_dict, *, state=None):
        super(Marginalised, self).__init__("marginalised", 'linear', True, shape_dict,
                                           state=state)
        self.func_dict = func_dict

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        """
        log(L(X) dX) = logL(X) + E[w]
        Args:
            posterior_sample:
            n_i:
            log_L_i:
            from_U:
        Returns:
        """
        return dict_multimap(lambda func: func(**posterior_sample), self.func_dict)

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
        return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
                              dict_multimap(lambda f: jnp.sign(f), f))
