from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import while_loop, dynamic_update_slice
from jax import random, vmap
from jax.scipy.special import logsumexp
from typing import NamedTuple, Dict
from collections import namedtuple

from jaxns.prior_transforms import PriorTransform
from jaxns.param_tracking import Evidence, PosteriorFirstMoment, PosteriorSecondMoment, \
    InformationGain
from jaxns.utils import dict_multimap
from jaxns.likelihood_samplers import (expanded_box,
                                       slice_sampling, constrained_hmc, expanded_ellipsoid)


class NestedSamplerState(NamedTuple):
    """
    key: PRNG
    i: Current iteration index
    L_live: The array of likelihood at live points_U
    L_dead: The array of likelihood at dead points_U
    live_points_U: The set of live points_U
    dead_points: The set of dead points_U
    """
    key: jnp.ndarray
    done: bool
    i: int
    num_likelihood_evaluations: int  # int, number of times logL evaluated.
    live_points_U: jnp.ndarray  # [N, M] points_U in unit cube of live points_U
    live_points: Dict  # [N, M] points_U in constrained space of live points_U in dict struture
    log_L_live: jnp.ndarray  # log likelihood of live points_U
    dead_points: Dict  # [D, M] dead points_U in dci structure
    num_dead: int  # int, number of samples (dead points_U) taken so far.
    log_L_dead: jnp.ndarray  # log likelhood of dead points_U
    evidence_state: namedtuple  # state for logZ
    m_state: namedtuple  # state for parameter mean
    M_state: namedtuple  # state for parameter covariance
    information_gain_state: namedtuple  # information, H, state
    status: int  # exit status: 0=good, 1=max samples reached


class NestedSampler(object):
    _available_samplers = ['box', 'whitened_box', 'chmc', 'slice', 'ellipsoid', 'whitened_ellipsoid']

    def __init__(self, loglikelihood, prior_transform: PriorTransform, sampler_name='ellipsoid'):
        self.sampler_name = sampler_name
        if self.sampler_name not in self._available_samplers:
            raise ValueError("sampler {} should be one of {}.".format(self.sampler_name, self._available_samplers))

        def fixed_likelihood(**x):
            log_L = loglikelihood(**x)
            return jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)

        self.loglikelihood = fixed_likelihood
        self.prior_transform = prior_transform

        def loglikelihood_from_U(U):
            return fixed_likelihood(**prior_transform(U))

        self.loglikelihood_from_U = loglikelihood_from_U

    def initial_state(self, key, num_live_points, max_samples, collect_samples: bool):
        # get initial live points_U
        def single_sample(key):
            U = random.uniform(key, shape=(self.prior_transform.U_ndims,))
            constrained = self.prior_transform(U)
            log_L = self.loglikelihood(**constrained)
            return U, constrained, log_L

        key, init_key = random.split(key, 2)
        live_points_U, live_points, log_L_live = vmap(single_sample)(random.split(init_key, num_live_points))

        if collect_samples:
            dead_points = dict_multimap(lambda shape: jnp.zeros((max_samples,) + shape), self.prior_transform.to_shapes)
            log_L_dead = jnp.zeros((max_samples,))
        else:
            dead_points = None
            log_L_dead = None

        evidence = Evidence()
        m = PosteriorFirstMoment(self.prior_transform.to_shapes)
        M = PosteriorSecondMoment(self.prior_transform.to_shapes)
        information_gain = InformationGain(global_evidence=evidence)

        state = NestedSamplerState(
            key=key,
            done=jnp.array(False),
            i=jnp.array(0),
            num_likelihood_evaluations=num_live_points,
            live_points=live_points,
            live_points_U=live_points_U,
            log_L_live=log_L_live,
            dead_points=dead_points,
            log_L_dead=log_L_dead,
            num_dead=jnp.array(0),
            evidence_state=evidence.state,
            m_state=m.state,
            M_state=M.state,
            information_gain_state=information_gain.state,
            status=jnp.array(0)
        )

        return state

    def _one_step(self, state: NestedSamplerState, collect_samples: bool):
        # get next dead point
        i_min = jnp.argmin(state.log_L_live)
        dead_point = dict_multimap(lambda x: x[i_min, ...], state.live_points)
        log_L_min = state.log_L_live[i_min]
        if collect_samples:
            dead_points = dict_multimap(lambda x, y: dynamic_update_slice(x,
                                                                          y[None, ...],
                                                                          [state.num_dead] + [0] * len(y.shape)),
                                        state.dead_points, dead_point)
            log_L_dead = dynamic_update_slice(state.log_L_dead,
                                              log_L_min[None],
                                              [state.num_dead])
            state = state._replace(dead_points=dead_points,
                                   log_L_dead=log_L_dead)
        state = state._replace(
            num_dead=state.num_dead + 1)

        n = state.log_L_live.shape[0]

        # update tracking
        evidence = Evidence(state=state.evidence_state)
        evidence.update(dead_point, n, log_L_min)
        m = PosteriorFirstMoment(self.prior_transform.to_shapes, state=state.m_state)
        m.update(dead_point, n, log_L_min)
        M = PosteriorSecondMoment(self.prior_transform.to_shapes, state=state.M_state)
        M.update(dead_point, n, log_L_min)
        H = InformationGain(global_evidence=evidence, state=state.information_gain_state)
        H.update(dead_point, n, log_L_min)

        state = state._replace(evidence_state=evidence.state,
                               m_state=m.state,
                               M_state=M.state,
                               information_gain_state=H.state)

        # select cluster to spawn into
        if self.sampler_name == 'box':
            key, spawn_id_key = random.split(state.key, 2)
            spawn_point_id = random.randint(spawn_id_key, shape=(), minval=0,
                                            maxval=n)
            sampler_results = expanded_box(key,
                                           log_L_constraint=log_L_min,
                                           live_points_U=state.live_points_U,
                                           spawn_point_U=state.live_points_U[spawn_point_id, :],
                                           loglikelihood_from_constrained=self.loglikelihood,
                                           prior_transform=self.prior_transform,
                                           whiten=False)
        elif self.sampler_name == 'whitened_box':
            key, spawn_id_key = random.split(state.key, 2)
            spawn_point_id = random.randint(spawn_id_key, shape=(), minval=0,
                                            maxval=n)
            sampler_results = expanded_box(key,
                                           log_L_constraint=log_L_min,
                                           live_points_U=state.live_points_U,
                                           spawn_point_U=state.live_points_U[spawn_point_id, :],
                                           loglikelihood_from_constrained=self.loglikelihood,
                                           prior_transform=self.prior_transform,
                                           whiten=True)
        elif self.sampler_name == 'ellipsoid':
            sampler_results = expanded_ellipsoid(state.key,
                                                 log_L_constraint=log_L_min,
                                                 live_points_U=state.live_points_U,
                                                 loglikelihood_from_constrained=self.loglikelihood,
                                                 prior_transform=self.prior_transform,
                                                 whiten=False)
        elif self.sampler_name == 'whitened_ellipsoid':
            sampler_results = expanded_ellipsoid(state.key,
                                                 log_L_constraint=log_L_min,
                                                 live_points_U=state.live_points_U,
                                                 loglikelihood_from_constrained=self.loglikelihood,
                                                 prior_transform=self.prior_transform,
                                                 whiten=True)
        elif self.sampler_name == 'chmc':
            sampler_results = constrained_hmc(state.key, log_L_constraint=log_L_min,
                                              live_points_U=state.live_points_U,
                                              last_live_point=dead_point,
                                              loglikelihood_from_constrained=self.loglikelihood,
                                              prior_transform=self.prior_transform, T=2)
        elif self.sampler_name == 'slice':
            sampler_results = slice_sampling(state.key, log_L_constraint=log_L_min, live_points_U=state.live_points_U,
                                             dead_point=dead_point,
                                             num_slices=state.live_points_U.shape[1],
                                             loglikelihood_from_constrained=self.loglikelihood,
                                             prior_transform=self.prior_transform)
        print(sampler_results)
        #
        log_L_live = dynamic_update_slice(state.log_L_live, sampler_results.log_L_new[None], [i_min])
        live_points = dict_multimap(lambda x, y: dynamic_update_slice(x, y[None, ...],
                                                                      [i_min] + [0] * len(y.shape)), state.live_points,
                                    sampler_results.x_new)
        live_points_U = dynamic_update_slice(state.live_points_U, sampler_results.u_new[None, :],
                                             [i_min, 0])

        state = state._replace(key=sampler_results.key,
                               num_likelihood_evaluations=state.num_likelihood_evaluations +
                                                          sampler_results.num_likelihood_evaluations,
                               log_L_live=log_L_live,
                               live_points=live_points,
                               live_points_U=live_points_U)

        return state

    def __call__(self, key, num_live_points, max_samples=1e6,
                 collect_samples=True,
                 termination_frac=0.05,
                 stoachastic_uncertainty=True):
        max_samples = jnp.array(max_samples, dtype=jnp.int64)
        num_live_points = jnp.array(num_live_points, dtype=jnp.int64)
        state = self.initial_state(key, num_live_points,
                                   max_samples=max_samples,
                                   collect_samples=collect_samples)

        def body(state: NestedSamplerState):
            # print(list(map(lambda x: type(x), state)))
            # do one sampling step
            state = self._one_step(state, collect_samples=collect_samples)
            evidence = Evidence(state=state.evidence_state)
            # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
            logZ_live = logsumexp(state.log_L_live) - jnp.log(state.log_L_live.shape[0]) + evidence.X.log_value
            # Z_live < f * Z => logZ_live < log(f) + logZ
            done = (logZ_live < jnp.log(termination_frac) + evidence.mean) | ((state.i + 1) >= max_samples)
            state = state._replace(done=done,
                                   i=state.i + 1)
            # print(list(map(lambda x: type(x), state)))
            # exit(0)
            return state

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        results = self._finalise_results(state, collect_samples=collect_samples,
                                         stoachastic_uncertainty=stoachastic_uncertainty)
        return results

    def _finalise_results(self, state: NestedSamplerState, collect_samples: bool, stoachastic_uncertainty: bool):
        collect = ['logZ',
                   'logZerr',
                   'ESS',
                   'ESS_err',
                   'H',
                   'H_err',
                   'num_likelihood_evaluations',
                   'efficiency',
                   'param_mean',
                   'param_mean_err',
                   'param_covariance',
                   'param_covariance_err']

        if collect_samples:
            collect.append('samples')
            collect.append('log_L_samples')
            collect.append('n_per_sample')
            collect.append('log_p')
            collect.append('log_X')

        NestedSamplerResults = namedtuple('NestedSamplerResults', collect)
        evidence = Evidence(state=state.evidence_state)
        evidence.update_from_live_points(state.live_points, state.log_L_live)
        m = PosteriorFirstMoment(self.prior_transform.to_shapes, state=state.m_state)
        m.update_from_live_points(state.live_points, state.log_L_live)
        M = PosteriorSecondMoment(self.prior_transform.to_shapes, state=state.M_state)
        M.update_from_live_points(state.live_points, state.log_L_live)
        H = InformationGain(global_evidence=evidence, state=state.information_gain_state)
        H.update_from_live_points(state.live_points, state.log_L_live)

        data = dict(
            logZ=evidence.mean,
            logZerr=jnp.sqrt(evidence.variance),
            ESS=evidence.effective_sample_size,
            ESS_err=None,
            H=H.mean,
            H_err=None,
            num_likelihood_evaluations=state.num_likelihood_evaluations,
            efficiency=(state.num_dead + state.log_L_live.shape[0]) / state.num_likelihood_evaluations,
            param_mean=m.mean,
            param_mean_err=dict_multimap(lambda x: jnp.sqrt(x), m.variance),
            param_covariance=dict_multimap(lambda x, y: x - y[..., :, None] * y[..., None, :], M.mean, m.mean),
            param_covariance_err=dict_multimap(lambda x, y: jnp.sqrt(
                x + jnp.sqrt(y[..., :, None] * y[..., None, :])), M.variance, m.variance)
        )

        if collect_samples:
            def _left_broadcast_mul(x, y):
                return jnp.reshape(x, (-1,) + tuple([1] * (len(y.shape) - 1))) * y

            def _sample(key, samples, log_L_samples):
                # N
                t = random.beta(key, n_per_sample, 1)
                log_t = jnp.log(t)
                log_X = jnp.cumsum(log_t)
                log_L_samples = jnp.concatenate([jnp.array([-jnp.inf]), log_L_samples])
                log_X = jnp.concatenate([jnp.array([0.]), log_X])
                # log_dX = log(1-t_i) + log(X[i-1])
                log_dX = jnp.log(1. - t) + log_X[:-1]  # jnp.log(-jnp.diff(jnp.exp(log_X)))
                log_avg_L = jnp.logaddexp(log_L_samples[:-1], log_L_samples[1:]) - jnp.log(2.)
                log_p = log_dX + log_avg_L
                # param calculation
                logZ = logsumexp(log_p)
                log_w = log_p - logZ
                weights = jnp.exp(log_w)
                m = dict_multimap(lambda samples: jnp.sum(_left_broadcast_mul(weights, samples), axis=0), samples)
                dsamples = dict_multimap(jnp.subtract, samples, m)
                cov = dict_multimap(lambda dsamples: jnp.sum(
                    _left_broadcast_mul(weights, (dsamples[..., :, None] * dsamples[..., None, :])), axis=0), dsamples)
                # Kish's ESS = [sum weights]^2 / [sum weights^2]
                ESS = jnp.exp(2. * logsumexp(log_w) - logsumexp(2. * log_w))
                # H = sum w_i log(w_i)
                H = jnp.sum(jnp.exp(log_w) * log_w)
                return logZ, m, cov, ESS, H

            num_live_points = state.log_L_live.shape[0]
            n_per_sample = jnp.concatenate([jnp.full((state.num_dead,), num_live_points),
                                            num_live_points - jnp.arange(num_live_points)])

            log_t = jnp.log(n_per_sample) - jnp.log(n_per_sample + 1.)
            log_X = jnp.cumsum(log_t)
            ar = jnp.argsort(state.log_L_live)
            samples = dict_multimap(lambda dead_points, live_points: jnp.concatenate([dead_points[:state.num_dead, :],
                                                                                      live_points[ar, :]], axis=0),
                                    state.dead_points, state.live_points)
            log_L_samples = jnp.concatenate([state.log_L_dead[:state.num_dead],
                                             state.log_L_live[ar]])
            data['samples'] = samples
            data['log_L_samples'] = log_L_samples
            data['n_per_sample'] = n_per_sample
            data['log_X'] = log_X

            if stoachastic_uncertainty:
                S = 200
                logZ, m, cov, ESS, H = vmap(lambda key: _sample(key, samples, log_L_samples))(
                    random.split(state.key, S))
                data['logZ'] = jnp.mean(logZ, axis=0)
                data['logZerr'] = jnp.std(logZ, axis=0)
                data['param_mean'] = dict_multimap(lambda m: jnp.mean(m, axis=0), m)
                data['param_mean_err'] = dict_multimap(lambda m: jnp.std(m, axis=0), m)
                data['param_covariance'] = dict_multimap(lambda cov: jnp.mean(cov, axis=0), cov)
                data['param_covariance_err'] = dict_multimap(lambda cov: jnp.std(cov, axis=0), cov)
                data['H'] = jnp.mean(H, axis=0)
                data['H_err'] = jnp.std(H, axis=0)
                data['ESS'] = jnp.mean(ESS, axis=0)
                data['ESS_err'] = jnp.std(ESS, axis=0)

            # build mean weights
            log_L_samples = jnp.concatenate([jnp.array([-jnp.inf]), log_L_samples])
            log_X = jnp.concatenate([jnp.array([0.]), log_X])
            # log(dX_i) = log(X[i-1] - X[i]) = log((1-t_i)*X[i-1]) = log(1-t_i) + log(X[i-1])
            log_dX = - jnp.log(n_per_sample + 1.) + log_X[:-1]
            # log_dX = jnp.log(-jnp.diff(jnp.exp(log_X)))
            log_avg_L = jnp.logaddexp(log_L_samples[:-1], log_L_samples[1:]) - jnp.log(2.)
            # w_i = dX_i avg_L_i
            log_w = log_dX + log_avg_L
            log_p = log_w - logsumexp(log_w)
            data['log_p'] = log_p

        return NestedSamplerResults(**data)
