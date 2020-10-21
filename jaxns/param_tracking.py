from collections import namedtuple, OrderedDict
from typing import List
from jax import jit, numpy as jnp
import numpy as np
from jax.lax import while_loop
from jaxns.utils import dict_multimap, swap_dict_namedtuple, swap_namedtuple_dict, signed_logaddexp, tuple_prod

LogParam = namedtuple('LogParam', ['log_value'])
SignedLogParam = namedtuple('SignedLogParam', ['log_abs_value', 'sign'])
ParamTrackingState = namedtuple("ParamTrackingState", ['f', 'f2', 'fX', 'X', 'X2', 'w', 'w2', 'L_i1'])


class TrackedExpectation(object):
    def __init__(self,
                 marginalised_funcs,
                 marginalised_shapes,
                 *,
                 state=None):
        self.State = ParamTrackingState
        total_length = self.build_meta(marginalised_funcs, marginalised_shapes)
        if state is not None:
            self.state = state
        else:
            initial_f = SignedLogParam(-jnp.inf * jnp.ones(total_length), jnp.ones(total_length))
            initial_f2 = initial_f
            initial_fX = initial_f
            initial_X = LogParam(jnp.zeros(()))
            initial_X2 = initial_X
            initial_w = LogParam(-jnp.inf)
            initial_w2 = initial_w
            initial_L_i1 = initial_w

            self.state = self.State(f=initial_f,
                                    f2=initial_f2,
                                    fX=initial_fX,
                                    X=initial_X,
                                    X2=initial_X2,
                                    w=initial_w,
                                    w2=initial_w2,
                                    L_i1=initial_L_i1)

    def build_meta(self, marginalised_funcs, marginalised_shapes):
        # build initial flat vector and the method of storing them
        self.meta = dict(funcs=[], start_idx=[], stop_idx=[], shape=[], names=[])
        _funcs = []
        _shapes = []
        _names = []
        _funcs.append(lambda posterior_sample, n_i, log_L_i: jnp.asarray(1.))  # evidence
        _shapes.append(())
        _names.append("static:Z")
        _funcs.append(lambda posterior_sample, n_i, log_L_i: log_L_i)  # H
        _shapes.append(())
        _names.append("static:H")
        if marginalised_shapes is not None:
            def build_marg_func(func):
                def marg_func(posterior_sample, n_i, log_L_i):
                    return func(**posterior_sample)

                return marg_func

            ###
            # marginalised functions
            marg_keys = sorted(marginalised_funcs.keys())
            assert marg_keys == sorted(marginalised_shapes.keys())
            for key in marg_keys:
                _funcs.append(build_marg_func(marginalised_funcs[key]))  # marginalised func
                _shapes.append(marginalised_shapes[key])
                _names.append("marg:{}".format(key))
        # build meta
        idx = 0
        for func, shape, name in zip(_funcs, _shapes, _names):
            self.meta['funcs'].append(func)
            self.meta['start_idx'].append(idx)
            self.meta['shape'].append(shape)
            idx += tuple_prod(self.meta['shape'][-1])
            self.meta['stop_idx'].append(idx)
            self.meta['names'].append(name)
        total_length = idx
        return total_length

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = self.State(*state)

    def lookup_meta(self, prefix, suffix=None):
        if suffix is not None:
            name = f"{prefix}:{suffix}"
            if name not in self.meta['names']:
                raise ValueError("{} not in names".format(name))
            idx = self.meta['names'].index(name)
            return (self.meta['start_idx'][idx], self.meta['stop_idx'][idx], self.meta['shape'][idx])
        else:
            res = []
            for name in self.meta['names']:  # get all matching prefixes
                if name.split(":")[0] == prefix:
                    idx = self.meta['names'].index(name)
                    res.append((name.split(":")[1], self.meta['start_idx'][idx], self.meta['stop_idx'][idx],
                                self.meta['shape'][idx]))
            return res

    def effective_sample_size(self):
        """Kish's ESS = [sum weights]^2 / [sum weights^2]
        """
        return jnp.exp(2. * self.state.w.log_value - self.state.w2.log_value)
        # w = self._linear_mean(*self.lookup_meta('static', 'Z'), normalised=False)
        # w2 = self._linear_variance(*self.lookup_meta('static', 'Z'), normalised=False) + w**2
        # return w**2/w2

    def evidence_mean(self):
        return self._log_mean(*self.lookup_meta('static', 'Z'), normalised=False)

    def evidence_variance(self):
        return self._log_variance(*self.lookup_meta('static', 'Z'), normalised=False)

    def information_gain_mean(self):
        return -self._linear_mean(*self.lookup_meta('static', 'H'), normalised=True) + self.evidence_mean()

    def information_gain_variance(self):
        return self._linear_variance(*self.lookup_meta('static', 'H'), normalised=True) + self.evidence_variance()

    def marg_mean(self):
        d = OrderedDict()
        for key, start_idx, stop_idx, shape in self.lookup_meta("marg"):
            d[key] = self._linear_mean(start_idx, stop_idx, shape, normalised=True)
        return d

    def marg_variance(self):
        d = OrderedDict()
        for key, start_idx, stop_idx, shape in self.lookup_meta("marg"):
            d[key] = self._linear_variance(start_idx, stop_idx, shape, normalised=True)
        return d

    def _linear_mean(self, start_idx: int, stop_idx: int, shape, normalised: bool):
        if normalised:
            return jnp.reshape(self.state.f.sign[start_idx:stop_idx] * jnp.exp(
                self.state.f.log_abs_value[start_idx: stop_idx] - self.state.w.log_value), shape)
        else:
            return jnp.reshape(
                self.state.f.sign[start_idx:stop_idx] * jnp.exp(self.state.f.log_abs_value[start_idx:stop_idx]), shape)

    def _log_mean(self, start_idx: int, stop_idx: int, shape, normalised: bool):
        # lambda log_f, log_f2: (2. * log_f - 0.5 * log_f2) - self.w.log_value
        if normalised:
            return jnp.reshape((2. * self.state.f.log_abs_value[start_idx:stop_idx] - 0.5 * self.state.f2.log_abs_value[
                                                                                            start_idx:stop_idx]) - self.state.w.log_value,
                               shape)
        else:
            return jnp.reshape(2. * self.state.f.log_abs_value[start_idx:stop_idx] - 0.5 * self.state.f2.log_abs_value[
                                                                                           start_idx:stop_idx], shape)

    def _linear_variance(self, start_idx: int, stop_idx: int, shape, normalised: bool):
        if normalised:
            # lambda f, f2: (f2/ w - (f/w) ** 2)
            t1, t1_sign = signed_logaddexp(self.state.f2.log_abs_value[start_idx:stop_idx] - self.state.w.log_value,
                                           self.state.f2.sign[start_idx:stop_idx],
                                           2. * (self.state.f.log_abs_value[start_idx:stop_idx] - self.state.w.log_value),
                                           -1.)
            return jnp.reshape(t1_sign * jnp.exp(t1), shape)
        else:
            t1, t1_sign = signed_logaddexp(
                self.state.f2.log_abs_value[start_idx:stop_idx],
                self.state.f2.sign[start_idx:stop_idx],
                2. * self.state.f.log_abs_value[start_idx:stop_idx],
                -1.)
            return jnp.reshape(t1_sign * jnp.exp(t1), shape)

    def _log_variance(self, start_idx: int, stop_idx: int, shape, normalised: bool):
        if normalised:
            # (log_f2 - 2. * log_f) - self.w.log_value
            return jnp.reshape(self.state.f2.log_abs_value[start_idx:stop_idx] - 2. * self.state.f.log_abs_value[
                                                                                      start_idx:stop_idx] - self.state.w.log_value,
                               shape)
        else:
            return jnp.reshape(
                self.state.f2.log_abs_value[start_idx:stop_idx] - 2. * self.state.f.log_abs_value[start_idx:stop_idx],
                shape)

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i) -> SignedLogParam:
        # use meta data to compute
        res = []
        for name, func in zip(self.meta['names'], self.meta['funcs']):
            res.append(func(posterior_sample, n_i, log_L_i).flatten())
        res = jnp.concatenate(res)
        return SignedLogParam(jnp.log(jnp.abs(res)), jnp.sign(res))

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
        w2_i1 = self.state.w2
        f_alpha_i = self.compute_log_f_alpha(posterior_sample_i, n_i, log_L_i)
        log_triangle = jnp.logaddexp(L_i1.log_value, log_L_i) - jnp.log(2.)

        n_plus_1 = n_i + 1.
        n_plus_2 = n_i + 2.

        # w_i = w_i1 + X_i1 * triangle /(n+1)
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

        self.state = self.State(f=f_i,
                                X=X_i,
                                f2=f2_i,
                                fX=fX_i,
                                X2=X2_i,
                                w=w_i,
                                w2=w2_i,
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
        return SignedLogParam(jnp.asarray(0.), jnp.asarray(1.))


#
#
# class PosteriorFirstMoment(TrackedExpectation):
#     """
#     Computes the mean of a parameter. If the shape_dict of the parameter is [..., M], then all dimensions except the last
#     one are treated as batched dimensions.
#     """
#
#     def __init__(self, shape, *, state=None):
#         super(PosteriorFirstMoment, self).__init__('m',
#                                                    'linear',
#                                                    True,
#                                                    shape,
#                                                    state=state)
#
#     def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
#         return posterior_sample
#
#     def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
#         f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
#         return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
#                               dict_multimap(lambda f: jnp.sign(f), f))
#
#
# class PosteriorSecondMoment(TrackedExpectation):
#     def __init__(self, shape, *, state=None):
#         shape = dict_multimap(lambda shape: shape + shape[-1:], shape)
#         super(PosteriorSecondMoment, self).__init__('M',
#                                                     'linear',
#                                                     True,
#                                                     shape,
#                                                     state=state)
#
#     def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
#         return dict_multimap(lambda posterior_sample: posterior_sample[..., None] * posterior_sample[..., None, :],
#                              posterior_sample)
#
#     def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
#         f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
#         return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
#                               dict_multimap(lambda f: jnp.sign(f), f))
#
#
# class InformationGain(TrackedExpectation):
#     """
#     H = int post(x) log(post(x)/prior(x)) dx
#     = int L(x) p(x)/Z log(L(x)/Z) dx
#     = int L(x) p(x)/Z log(L(x)) dx - log(Z)
#     = E(log(L(x))) - log(Z)
#     = sum w(x) log(L(x)) - log(Z)
#
#     This produces -H.
#     """
#
#     def __init__(self, global_evidence: Evidence, *, state=None):
#         super(InformationGain, self).__init__('H', 'linear', True, (),
#                                               state=state)
#         self.global_evidence = global_evidence
#
#     @property
#     def mean(self):
#         mean = -super(InformationGain, self).mean + self.global_evidence.mean
#         return mean
#
#     @property
#     def variance(self):
#         variance = super(InformationGain, self).variance + self.global_evidence.variance
#         return variance
#
#     def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
#         """
#         Args:
#             posterior_sample:
#             n_i:
#             log_L_i:
#             from_U:
#         Returns:
#         """
#         return log_L_i
#
#     def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
#         return SignedLogParam(jnp.log(jnp.abs(log_L_i)), jnp.sign(log_L_i))
#
#
# class Marginalised(TrackedExpectation):
#     def __init__(self, func_dict, shape_dict, *, state=None):
#         super(Marginalised, self).__init__("marginalised", 'linear', True, shape_dict,
#                                            state=state)
#         self.func_dict = func_dict
#
#     def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
#         """
#         log(L(X) dX) = logL(X) + E[w]
#         Args:
#             posterior_sample:
#             n_i:
#             log_L_i:
#             from_U:
#         Returns:
#         """
#         return dict_multimap(lambda func: func(**posterior_sample), self.func_dict)
#
#     def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
#         f = self.compute_f_alpha(posterior_sample, n_i, log_L_i)
#         return SignedLogParam(dict_multimap(lambda f: jnp.log(jnp.abs(f)), f),
#                               dict_multimap(lambda f: jnp.sign(f), f))

def test_param_tracking():
    from jax import jit, numpy as jnp, disable_jit, make_jaxpr
    shape = {
        'a': (4,),
        'b': (4,),
        'c': (4,),
        'd': (4,),
        'e': (4,),
        'f': (4,),
    }
    sample = dict_multimap(jnp.ones, shape)
    n = jnp.array(10)
    log_L = jnp.array(0.)

    @jit
    def test_jax(sample, n, log_L):
        tracked = TrackedExpectation({k: lambda sample, n, log_L: jnp.ones(shape[k]) for k in shape.keys()}, shape)
        tracked.update(sample, n, log_L)

        return (tracked.evidence_mean(), tracked.evidence_variance(), tracked.information_gain_mean())
        # return (evidence.state, H.state, m.state, M.state)

    print()
    print(len(str(make_jaxpr(test_jax)(sample, n, log_L))))
    with disable_jit():
        print(test_jax(sample, n, log_L))
