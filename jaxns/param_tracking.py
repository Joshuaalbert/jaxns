from collections import namedtuple, OrderedDict
from jax import numpy as jnp, tree_multimap
from jax.lax import scan
from jaxns.utils import dict_multimap, signed_logaddexp, tuple_prod

LogParam = namedtuple('LogParam', ['log_value'])
SignedLogParam = namedtuple('SignedLogParam', ['log_abs_value', 'sign'])
ParamTrackingState = namedtuple("ParamTrackingState", ['f', 'f2', 'fX', 'X', 'X2', 'w', 'w2', 'L_i1', 'dw'])


class TrackedExpectation(object):
    def __init__(self,
                 marginalised_funcs,
                 marginalised_shapes,
                 *,
                 state=None):
        """
        Tracks marginalised functions including evidence and information gain.
        Currently, the marginalisation expects the tracked expressions to all be of float dtype.

        Args:
            marginalised_funcs: dict of callables(**priors_X)
            marginalised_shapes: dict of shapes of results of each marginalised function.
            state: internal state, if not given then initialises this.
        """
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

    def update_from_live_points(self, live_points, log_L_live, is_satisfying=None, num_likelihood_evals=None):
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


