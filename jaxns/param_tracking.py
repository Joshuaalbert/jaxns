from collections import namedtuple
from jax import numpy as jnp
from jax.lax import while_loop
from jaxns.utils import dict_multimap

class Param(object):
    value: jnp.ndarray
    log_value: jnp.ndarray

    def __init__(self, is_pytree:bool=False):
        self.is_pytree = is_pytree

    def _paramify(self, other):
        if not isinstance(other, Param):
            if other > 0.:
                other = LogParam(jnp.log(other), is_pytree=False)
            else:
                other = LinearParam(other, is_pytree=False)
        return other

    def _maybe_dict_multimap(self, single_lambda, self_value, other_value, other_is_pytree):
        if self.is_pytree and other_is_pytree:
            return dict_multimap(single_lambda, self_value, other_value), True
        if self.is_pytree and not other_is_pytree:
            return dict_multimap(lambda a: single_lambda(a, other_value), self_value), True
        if not self.is_pytree and not other_is_pytree:
            return single_lambda(self_value, other_value), False
        if not self.is_pytree and other_is_pytree:
            return dict_multimap(lambda b: single_lambda(self_value, b), other_value), True

    def __add__(self, other):
        other = self._paramify(other)
        if isinstance(self, LogParam) and isinstance(other, LogParam):
            return LogParam(*self._maybe_dict_multimap(jnp.logaddexp, self.log_value, other.log_value, other.is_pytree))
        else:
            return LinearParam(*self._maybe_dict_multimap(jnp.add, self.value, other.value, other.is_pytree))

    def __sub__(self, other):
        return self + (-1. * other)

    def __mul__(self, other):
        other = self._paramify(other)
        if isinstance(self, LogParam) and isinstance(other, LogParam):
            return LogParam(*self._maybe_dict_multimap(jnp.add, self.log_value, other.log_value, other.is_pytree))
        else:
            return LinearParam(*self._maybe_dict_multimap(jnp.multiply, self.value, other.value, other.is_pytree))

    def __truediv__(self, other):
        other = self._paramify(other)
        if isinstance(self, LogParam) and isinstance(other, LogParam):
            return LogParam(*self._maybe_dict_multimap(jnp.subtract, self.log_value, other.log_value, other.is_pytree))
        else:
            return LinearParam(*self._maybe_dict_multimap(jnp.true_divide, self.value, other.value, other.is_pytree))

    def __pow__(self, other):
        if isinstance(self, LogParam):
            if self.is_pytree:
                return LogParam(dict_multimap(lambda x: other * x, self.log_value), is_pytree=True)
            return LogParam(other * self.log_value, is_pytree=False)
        else:
            if self.is_pytree:
                return LinearParam(dict_multimap(lambda x: x**other, self.value), is_pytree=True)
            return LinearParam(self.value**other, is_pytree=False)


class LinearParam(Param):
    def __init__(self, value, is_pytree:bool):
        super(LinearParam, self).__init__(is_pytree=is_pytree)
        self.value = value

    @property
    def log_value(self):
        if self.is_pytree:
            return dict_multimap(jnp.log, self.value)
        return jnp.log(self.value)

    def __repr__(self):
        if self.is_pytree:
            return "value={}".format(dict_multimap(lambda x: jnp.round(x, 4), self.value))
        return "value={}".format(jnp.round(self.value, 4))


class LogParam(Param):
    def __init__(self, log_value, is_pytree:bool):
        super(LogParam, self).__init__(is_pytree=is_pytree)
        self.log_value = log_value

    @property
    def value(self):
        if self.is_pytree:
            return dict_multimap(jnp.exp, self.log_value)
        return jnp.exp(self.log_value)

    def __repr__(self):
        if self.is_pytree:
            return "log_value={}".format(dict_multimap(lambda x: jnp.round(x, 4), self.log_value))
        return "log_value={}".format(jnp.round(self.log_value, 4))

class TrackedExpectation(object):
    def __init__(self, name, type, normalised: bool, param_shape, *, state=None,
                 initial_f=None, initial_f2=None,
                 initial_fX=None, initial_X=None,
                 initial_X2=None,
                 initial_w=jnp.log(0.),
                 initial_w2=jnp.log(0.),
                 initial_L_i1=jnp.log(0.)):
        self.normalised = normalised
        if type.lower() not in ['log', 'linear']:
            raise ValueError("Type {} invalid.".format(type))
        self.type = type
        self.name = name
        self.param_shape = param_shape
        self.State = namedtuple("{}State".format(self.name), ['f', 'f2', 'fX', 'X', 'X2', 'w', 'w2', 'L_i1'])
        if state is not None:
            self.state = state
        else:
            def _prepare_init(x, shape):
                x = jnp.array(x)
                if x.shape != shape:
                    raise ValueError("Expected shape {}, got {}.".format(shape, x.shape))
                return x
            initial_f = dict_multimap(_prepare_init, initial_f, param_shape)
            initial_f2 = dict_multimap(_prepare_init, initial_f2, param_shape)
            initial_fX = dict_multimap(_prepare_init, initial_fX, param_shape)

            self.state = self.State(f=initial_f,
                                    f2=initial_f2,
                                    fX=initial_fX,
                                    X=jnp.array(initial_X),
                                    X2=jnp.array(initial_X2),
                                    w=jnp.array(initial_w),
                                    w2=jnp.array(initial_w2),
                                    L_i1=jnp.array(initial_L_i1))

    @property
    def effective_sample_size(self):
        """Kish's ESS = [sum weights]^2 / [sum weights^2]
        """
        return jnp.exp(2. * self.w.log_value - self.w2.log_value)

    @property
    def mean(self):
        if self.normalised:
            if self.type.lower() == 'linear':
                mean = dict_multimap(lambda f: f/self.w.value, self.f.value)
            elif self.type.lower() == 'log':
                mean = dict_multimap(lambda log_f, log_f2: (2. * log_f - 0.5 * log_f2) - self.w.log_value,
                                     self.f.log_value, self.f2.log_value)
        else:
            if self.type.lower() == 'linear':
                mean = self.f.value
            elif self.type.lower() == 'log':
                mean = dict_multimap(lambda log_f, log_f2: 2. * log_f - 0.5 * log_f2,
                                     self.f.log_value, self.f2.log_value)
        return mean

    @property
    def variance(self):
        if self.normalised:
            if self.type.lower() == 'linear':
                variance = dict_multimap(lambda f, f2: (f2 - f ** 2) / self.w.value, self.f.value, self.f2.value)
            elif self.type.lower() == 'log':
                variance = dict_multimap(lambda log_f, log_f2: (log_f2 - 2. * log_f) - self.w.log_value,
                                         self.f.log_value, self.f2.log_value)
        else:
            if self.type.lower() == 'linear':
                variance = dict_multimap(lambda f, f2: f2 - f ** 2, self.f.value, self.f2.value)
            elif self.type.lower() == 'log':
                variance = dict_multimap(lambda log_f, log_f2: log_f2 - 2. * log_f,
                                         self.f.log_value, self.f2.log_value)
        return variance

    def __repr__(self):
        return "{} = {} +- {}".format(
            # 'log' if self.type.lower() == 'log' else "",
            self.name, dict_multimap(lambda x: jnp.round(x, 4), self.mean),
            dict_multimap(lambda x:jnp.round(jnp.sqrt(x), 4), self.variance))

    @property
    def state(self):
        return tuple(self.State(
            f=self.f.log_value if isinstance(self.f, LogParam) else self.f.value,
            f2=self.f2.log_value,
            fX=self.fX.log_value if isinstance(self.fX, LogParam) else self.fX.value,
            X=self.X.log_value,
            X2=self.X2.log_value,
            w=self.w.log_value,
            w2=self.w2.log_value,
            L_i1=self.L_i1.log_value
        ))

    @state.setter
    def state(self, state):
        state = self.State(*state)
        self.f = self._maybe_log_param(state.f, self.type, True)
        self.f2 = self._maybe_log_param(state.f2, 'log', True)
        self.fX = self._maybe_log_param(state.fX, self.type, True)
        self.X = self._maybe_log_param(state.X, 'log', False)
        self.X2 = self._maybe_log_param(state.X2, 'log', False)
        self.w = self._maybe_log_param(state.w, 'log', False)
        self.w2 = self._maybe_log_param(state.w2, 'log', False)
        self.L_i1 = self._maybe_log_param(state.L_i1, 'log', False)

    def _maybe_log_param(self, v, type, is_pytree):
        #v is a pytree, all treemapping is pushed into the params
        if type.lower() == 'linear':
            return LinearParam(v, is_pytree)
        elif type.lower() == 'log':
            return LogParam(v, is_pytree)
        else:
            raise ValueError('Invalid type {}'.format(self.type))

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        raise NotImplementedError()

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        raise NotImplementedError()

    def update_from_live_points(self, live_points, log_L_live):
        ar = jnp.argsort(log_L_live)
        def body(state):
            (i, self_state) = state
            i_min = ar[i]
            self.state = self_state
            n = log_L_live.shape[0] - i
            self.update(dict_multimap(lambda x: x[i_min, ...], live_points), n, log_L_live[i_min])
            return (i+1, self.state)

        (_, self_state) = while_loop(lambda state: state[0] < log_L_live.shape[0],
                                     body,
                                     (0, self.state))
        self.state = self_state


    def update(self, posterior_sample_i, n_i, log_L_i):
        two = LogParam(jnp.log(2.), False)
        L_i = LogParam(log_L_i, False)
        L_i1 = self.L_i1
        f_i1 = self.f
        f2_i1 = self.f2
        fX_i1 = self.fX
        X_i1 = self.X
        X2_i1 = self.X2
        w_i1 = self.w
        w2_i1 = self.w2
        if self.type.lower() == 'log':
            f_alpha_i = LogParam(self.compute_log_f_alpha(posterior_sample_i, n_i, log_L_i), True)
        else:
            f_alpha_i = LinearParam(self.compute_f_alpha(posterior_sample_i, n_i, log_L_i), True)
        triangle = (L_i + L_i1) / two
        r_n_plus_1 = LogParam(-jnp.log(1. + n_i), False)
        n_n_plus_1 = LogParam(-jnp.log(1. + jnp.reciprocal(n_i)), False)  # n/(n+1) = 1/(1+1/n)
        n_n_plus_2 = LogParam(-jnp.log(1. + 2. * jnp.reciprocal(n_i)), False)  # n/(n+2) = 1/(1+2/n)
        r_n_plus_2 = LogParam(-jnp.log(2. + n_i), False)
        n_n_plus_2_n_plus_1 = n_n_plus_1 * r_n_plus_2

        w_i = w_i1 + X_i1 * triangle * r_n_plus_1 * LogParam(0., False)
        w2_i = w2_i1 + (X_i1 * triangle * r_n_plus_1 * LogParam(0., False)) ** 2
        f_i = f_i1 \
              + X_i1 * triangle * r_n_plus_1 * f_alpha_i
        X_i = X_i1 * n_n_plus_1
        f2_i = f2_i1 \
               + fX_i1 * two * r_n_plus_1 * triangle \
               + X2_i1 * r_n_plus_1 * r_n_plus_2 * (triangle * f_alpha_i) ** 2
        fX_i = n_n_plus_1 * fX_i1 \
               + n_n_plus_2_n_plus_1 * X2_i1 * (triangle * f_alpha_i)
        X2_i = n_n_plus_2 * X2_i1
        self.f = f_i
        self.X = X_i
        self.f2 = f2_i
        self.fX = fX_i
        self.X2 = X2_i
        self.w = w_i
        self.w2 = w2_i
        self.L_i1 = L_i


class Evidence(TrackedExpectation):
    def __init__(self, *, state=None):
        super(Evidence, self).__init__('logZ', 'log', False, (),
                                       state=state,
                                       initial_f=jnp.log(0.),
                                       initial_f2=jnp.log(0.),
                                       initial_X=jnp.log(1.),
                                       initial_fX=jnp.log(0.),
                                       initial_X2=jnp.log(1.)
                                       )

    def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i):
        return 0.


class PosteriorFirstMoment(TrackedExpectation):
    """
    Computes the mean of a parameter. If the shape of the parameter is [..., M], then all dimensions except the last
    one are treated as batched dimensions.
    """
    def __init__(self, shape, *, state=None):
        def _prepare_params(shape):
            return jnp.zeros(shape), jnp.log(jnp.zeros(shape)), jnp.zeros(shape)

        initial_f = dict_multimap(lambda shape: jnp.zeros(shape), shape)
        initial_f2 = dict_multimap(lambda shape: jnp.log(jnp.zeros(shape)), shape)
        initial_fX = dict_multimap(lambda shape: jnp.zeros(shape), shape)
        super(PosteriorFirstMoment, self).__init__('m',
                                                   'linear',
                                                   True,
                                                   shape,
                                                   state=state,
                                                   initial_f=initial_f,
                                                   initial_f2=initial_f2,
                                                   initial_X=jnp.log(1.),
                                                   initial_fX=initial_fX,
                                                   initial_X2=jnp.log(1.))

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        return posterior_sample

class PosteriorSecondMoment(TrackedExpectation):
    def __init__(self, shape, *, state=None):
        shape = dict_multimap(lambda shape: shape + shape[-1:], shape)
        initial_f = dict_multimap(lambda shape: jnp.zeros(shape), shape)
        initial_f2 = dict_multimap(lambda shape: jnp.log(jnp.zeros(shape)), shape)
        initial_fX = dict_multimap(lambda shape: jnp.zeros(shape), shape)
        super(PosteriorSecondMoment, self).__init__('M',
                                                    'linear',
                                                    True,
                                                    shape,
                                                    state=state,
                                                    initial_f=initial_f,
                                                    initial_f2=initial_f2,
                                                    initial_X=jnp.log(1.),
                                                    initial_fX=initial_fX,
                                                    initial_X2=jnp.log(1.))

    def compute_f_alpha(self, posterior_sample, n_i, log_L_i):
        return dict_multimap(lambda posterior_sample: posterior_sample[..., None] * posterior_sample[...,None,:],
                             posterior_sample)


# class ClusterEvidence(TrackedExpectation):
#     def __init__(self, *, state=None,
#                  num_parent=None, num_per_cluster=None,
#                  global_evidence: Evidence = None):
#         if state is None:
#             initial_f = global_evidence.f.log_value + jnp.log(num_per_cluster) - jnp.log(num_parent)
#             initial_f2 = global_evidence.f2.log_value + jnp.log(num_per_cluster) + jnp.log(
#                 num_per_cluster + 1.) - jnp.log(num_parent) - jnp.log(num_parent + 1)
#             initial_fX = global_evidence.fX.log_value + jnp.log(num_per_cluster) + jnp.log(
#                 num_per_cluster + 1.) - jnp.log(num_parent) - jnp.log(num_parent + 1)
#             initial_X = global_evidence.X.log_value + jnp.log(num_per_cluster) - jnp.log(num_parent)
#             initial_X2 = global_evidence.X2.log_value + jnp.log(num_per_cluster) + jnp.log(
#                 num_per_cluster + 1.) - jnp.log(num_parent) - jnp.log(num_parent + 1)
#             initial_L_i1 = global_evidence.L_i1.log_value
#         else:
#             initial_f = None
#             initial_f2 = None
#             initial_fX = None
#             initial_X = None
#             initial_X2 = None
#             initial_L_i1 = None
#         super(ClusterEvidence, self).__init__('logZp', 'log', False,
#                                               state=state,
#                                               initial_f=initial_f,
#                                               initial_f2=initial_f2,
#                                               initial_X=initial_X,
#                                               initial_fX=initial_fX,
#                                               initial_X2=initial_X2,
#                                               initial_L_i1=initial_L_i1
#                                               )
#
#     def compute_log_f_alpha(self, posterior_sample, n_i, log_L_i, *, from_U=True):
#         return 0.
#
#     def update(self, posterior_sample_i, n_i, log_L_i,*, from_U=True):
#         n_i = jnp.where(n_i > 0, n_i, jnp.inf)
#         super(ClusterEvidence, self).update(posterior_sample_i, n_i, log_L_i, from_U=from_U)


class InformationGain(TrackedExpectation):
    """
    logH = -1/Z int L(X) log(L(X) dX) dX + log(Z)
    """
    def __init__(self, global_evidence: Evidence, *, state=None):
        super(InformationGain, self).__init__('logH', 'linear', True, (),
                                              state=state,
                                              initial_f=0.,
                                              initial_f2=jnp.log(0.),
                                              initial_X=jnp.log(1.),
                                              initial_fX=0.,
                                              initial_X2=jnp.log(1.)
                                              )
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
        log(L(X) dX) = logL(X) + E[w]
        Args:
            posterior_sample:
            n_i:
            log_L_i:
            from_U:

        Returns:

        """
        return log_L_i - jnp.log(n_i + 1.)

def debug_tracking():
    from jax import random, vmap
    from jax.lax import dynamic_update_slice
    from jax.scipy.special import logsumexp
    from jax.lax_linalg import triangular_solve
    import pylab as plt
    from born_rime.nested_sampling.prior_transforms import MVNDiagPrior
    # import dynesty
    # import dynesty.plotting as dyplot

    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = triangular_solve(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx
    #
    ndims = 2
    prior_mu = 1. * jnp.ones(ndims)
    prior_cov = jnp.diag(3.*jnp.ones(ndims)) ** 2

    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov==0., 0.95,data_cov)

    prior_transform = MVNDiagPrior(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    Z = Evidence()
    Z_state = Z.state
    M = PosteriorSecondMoment((ndims,))
    m = PosteriorFirstMoment((ndims,))
    H = InformationGain(global_evidence=Z)

    true_logZ = log_normal(data_mu, prior_mu, prior_cov + data_cov)

    post_mu = prior_cov @ jnp.linalg.solve(prior_cov + data_cov, data_mu) + data_cov @ jnp.linalg.solve(
        prior_cov + data_cov, prior_mu)
    post_cov = prior_cov - prior_cov @ jnp.linalg.solve(prior_cov + data_cov, prior_cov)

    def log_likelihood(U):
        param = m.prior_transform(U)
        return log_normal(param, data_mu, data_cov)

    num_live_points = 30 * prior_mu.size

    # # initialize our "static" nested sampler
    # sampler = dynesty.NestedSampler(lambda x: log_normal(x, data_mu, data_cov),
    #                                 prior_transform,
    #                                 ndims,
    #                                 nlive=num_live_points)
    #
    # # sample from the distribution
    # sampler.run_nested(dlogz=0.01)
    #
    # # grab our results
    # res = sampler.results
    # dyplot.cornerplot(res, show_titles=True)
    # plt.show()

    live_points = random.uniform(random.PRNGKey(300), shape=(num_live_points, ndims))
    log_L_live = vmap(log_likelihood)(live_points)
    dead_points = []
    log_L_dead = []
    remaining_evidence = 1.
    running_evidence = 0.
    i = 0
    while remaining_evidence > 0.01 * running_evidence:
        Z = Evidence(state=Z_state)
        U = random.uniform(random.PRNGKey(i), shape=prior_mu.shape,
                           minval=jnp.clip(jnp.min(live_points, axis=0) - 0.01, 0., 1),
                           maxval=jnp.clip(jnp.max(live_points, axis=0) + 0.01, 0., 1.))
        i_min = jnp.argmin(log_L_live)
        log_L_new = log_likelihood(U)
        if log_L_new > log_L_live[i_min]:
            dead_points.append(m.prior_transform(live_points[i_min, :]))
            log_L_dead.append(log_L_live[i_min])
            args = (live_points[i_min, :], live_points.shape[0], log_L_live[i_min])
            M.update(*args)
            m.update(*args)
            Z.update(*args)
            Z_state = Z.state
            H.update(*args)
            live_points = dynamic_update_slice(live_points, U[None, :], [i_min, 0])
            log_L_live = dynamic_update_slice(log_L_live, log_L_new[None], [i_min])
        remaining_evidence = jnp.mean(jnp.exp(log_L_live)) * Z.X.value
        running_evidence = jnp.exp(Z.mean)
        print("i = {}, Z = {}, Z_live = {}, efficiency = {}".format(i, running_evidence, remaining_evidence,
                                                                    len(log_L_dead) / (i + 1)))
        i += 1

    print(Z, H, m, M)
    print(Z.effective_sample_size)
    dead_points = jnp.array(dead_points)
    log_L_dead = jnp.array(log_L_dead)
    # print(m, M, Z)
    S = 1000
    b = random.beta(random.PRNGKey(0), num_live_points,
                    1, shape=(S, log_L_dead.size))
    live_points = vmap(prior_transform)(live_points)
    b = jnp.concatenate([b] + [random.beta(random.PRNGKey(i), num_live_points - i,
                                           1, shape=(S, 1)) for i in range(num_live_points)], axis=1)
    ar = jnp.argsort(log_L_live)
    dead_points = jnp.concatenate([dead_points, live_points[ar, :]], axis=0)
    log_L_dead = jnp.concatenate([log_L_dead, log_L_live[ar]], axis=0)
    # S, N+1
    log_X_i = jnp.concatenate([jnp.zeros((S, 1)),
                               jnp.cumsum(jnp.log(b), axis=1)], axis=1)
    # log_p_i = log_dX_i + log_L_i
    log_L_dead = jnp.concatenate([jnp.log(jnp.zeros((1,))), log_L_dead])
    plt.scatter(jnp.exp(log_X_i).mean(0), jnp.exp(log_L_dead))
    plt.title("L(X)")
    plt.show()
    # dX_i = X_i-1 - X_i
    log_dX_i = jnp.log(-jnp.diff(jnp.exp(log_X_i), axis=1))
    log_p_i = jnp.logaddexp(log_dX_i + log_L_dead[1:], log_dX_i + log_L_dead[:-1]) - jnp.log(2.)
    ESS = jnp.sum(jnp.exp(log_p_i), axis=1) ** 2 / jnp.sum(jnp.exp(log_p_i) ** 2, axis=1)
    print("ESS = {} +- {}".format(jnp.mean(ESS), jnp.std(ESS)))
    logZ = logsumexp(log_p_i, axis=1)
    print("logZ = {} +- {}".format(jnp.mean(logZ), jnp.std(logZ)))
    print("True logZ = {}".format(true_logZ))
    plt.hist(logZ)
    plt.show()
    # S, N
    w_i = jnp.exp(log_p_i - logZ[:, None])
    plt.plot(jnp.mean(w_i, axis=0))
    plt.title('weights')
    plt.show()
    plt.plot(dead_points[:, 0])
    plt.title("p[0]")
    plt.show()
    # S, M
    m = jnp.sum(w_i[:, :, None] * dead_points[None, :, :], axis=1)
    print("m={} +- {}".format(jnp.mean(m, axis=0), jnp.std(m, axis=0)))
    plt.hist(dead_points[:, 0], weights=jnp.mean(w_i, axis=0), bins=20)
    plt.show()
    plt.hist(m[:, 0], bins=30)
    plt.title('m[0]')
    plt.show()
    # S, M, M
    m2 = jnp.sum(w_i[:, :, None, None] * (dead_points[:, :, None] * dead_points[:, None, :]), axis=1)
    print("M={} +- {}".format(jnp.mean(m2 - m[:, :, None] * m[:, None, :], axis=0),
                              jnp.std(m2 - m[:, :, None] * m[:, None, :], axis=0)))
    print(post_mu, post_cov)





if __name__ == '__main__':
    pass
    # def _assert_shape(x, shape):
    #     assert x.shape == shape
    # print(dict_multimap(_assert_shape, jnp.array(1.), () ))
    # debug_tracking()
