import threading
import time
from queue import Queue

import numpy as np


def run(ndims, ensemble_size, input_queue: Queue, output_queue: Queue):
    # from jax.config import config
    # config.update("jax_enable_x64", True)

    from jaxns import Prior, Model
    from jaxns import TerminationCondition
    from jaxns.samplers import UniDimSliceSampler
    from jaxns.nested_sampler import StandardStaticNestedSampler
    import jax
    from jax import random, numpy as jnp
    import numpy as np
    import tensorflow_probability.substrates.jax as tfp
    tfpd = tfp.distributions

    prior_mu = jnp.zeros(ndims)
    prior_cov = jnp.eye(ndims)

    data_mu = 15 * jnp.ones(ndims)
    data_cov = jnp.eye(ndims)
    data_cov = jnp.where(data_cov == 0., 0.99, data_cov)

    def prior_model():
        x = yield Prior(
            tfpd.MultivariateNormalTriL(
                loc=prior_mu,
                scale_tril=jnp.linalg.cholesky(prior_cov)
            )
        )
        return x

    def log_likelihood(x):
        return tfpd.MultivariateNormalTriL(
            loc=data_mu,
            scale_tril=jnp.linalg.cholesky(data_cov)
        ).log_prob(x)

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)

    true_logZ = tfpd.MultivariateNormalTriL(
        loc=prior_mu,
        scale_tril=jnp.linalg.cholesky(prior_cov + data_cov)
    ).log_prob(data_mu)

    post_mu = prior_cov @ jnp.linalg.inv(prior_cov + data_cov) @ data_mu + data_cov @ jnp.linalg.inv(
        prior_cov + data_cov) @ prior_mu

    print(f"True post mu:{post_mu}")
    print(f"True log Z: {true_logZ}")

    while True:
        input_data = input_queue.get()
        if input_data is None:  # poison pill
            break
        (s, k, c, store_indices) = input_data

        nested_sampler = StandardStaticNestedSampler(
            model=model,
            num_live_points=c,
            max_samples=150000,
            sampler=UniDimSliceSampler(
                model=model,
                num_slices=model.U_ndims * s,
                num_phantom_save=k,
                midpoint_shrink=True,
                perfect=True
            ),
            init_efficiency_threshold=0.1,
            num_parallel_workers=1
        )

        def ns_run(key):
            termination_reason, state = nested_sampler._run(key=key, term_cond=TerminationCondition())
            results = nested_sampler._to_results(termination_reason=termination_reason, state=state, trim=False)
            # summary(results)
            # plot_diagnostics(results)
            return (results.log_Z_mean, results.log_Z_uncert, results.total_num_likelihood_evaluations,
                    results.total_num_samples, results.total_phantom_samples)

        run_compiled = jax.jit(ns_run).lower(random.PRNGKey(0)).compile()
        dt = []
        results = []
        print(f"Running s={s} k={k} c={c}")
        for i in range(ensemble_size):
            t0 = time.time()
            results.append(run_compiled(random.PRNGKey(i)))
            results[-1][0].block_until_ready()
            dt.append(time.time() - t0)
            print(dt[-1])
        dt = np.asarray(dt)  # [m]
        print(f"Time taken s={s} k={k} c={c}: {sum(dt)}")
        log_Z_mean, log_Z_uncert, num_likelihood_evals, total_num_samples, total_phantom_samples = np.asarray(
            results).T  # [:, m]
        output_data = (
            (dt, log_Z_mean, log_Z_uncert, num_likelihood_evals, total_num_samples, total_phantom_samples, true_logZ),
            store_indices)
        output_queue.put(output_data)
    # Poison pill
    output_queue.put(None)


if __name__ == '__main__':
    save_file = "ablation_results.npz"
    ndims = 8
    num_workers = 2
    ensemble_size = 30
    input_queue = Queue()
    output_queue = Queue()

    s_array = np.asarray([6])
    k_array = np.asarray(list(range(ndims)) + list(range(ndims, ndims * 6, ndims)) + [6 * ndims - 1])
    c_array = np.asarray([40]) * ndims

    Ns = len(s_array)
    Nk = len(k_array)
    Nc = len(c_array)

    log_Z_mean_array = np.zeros((Ns, Nk, Nc, ensemble_size))
    log_Z_uncert_array = np.zeros((Ns, Nk, Nc, ensemble_size))
    num_likelihood_evals_array = np.zeros((Ns, Nk, Nc, ensemble_size))
    run_time_array = np.zeros((Ns, Nk, Nc, ensemble_size))
    total_num_samples_array = np.zeros((Ns, Nk, Nc, ensemble_size))
    total_num_phantom_samples_array = np.zeros((Ns, Nk, Nc, ensemble_size))

    for i, s in enumerate(s_array):
        for j, k in enumerate(k_array):
            for l, c in enumerate(c_array):
                store_indices = (i, j, l)
                input_data = (s, k, c, store_indices)
                input_queue.put(input_data)

    # Poison pills
    for _ in range(num_workers):
        input_queue.put(None)

    workers = []
    for _ in range(num_workers):
        worker = threading.Thread(target=run, args=(ndims, ensemble_size, input_queue, output_queue))
        worker.start()
        workers.append(worker)

    num_poison_pills = 0
    true_logZ = None
    while num_poison_pills < num_workers:
        output_data = output_queue.get()
        if output_data is None:
            num_poison_pills += 1
            continue
        (
            (dt, log_Z_mean, log_Z_uncert, num_likelihood_evals, total_num_samples, total_phantom_samples, true_logZ),
            store_indices
        ) = output_data
        i, j, l = store_indices
        run_time_array[i, j, l, :] = dt
        log_Z_mean_array[i, j, l, :] = log_Z_mean
        log_Z_uncert_array[i, j, l, :] = log_Z_uncert
        num_likelihood_evals_array[i, j, l, :] = num_likelihood_evals
        total_num_samples_array[i, j, l, :] = total_num_samples
        total_num_phantom_samples_array[i, j, l, :] = total_phantom_samples

    # Save the result arrays and axes into npz file

    np.savez(
        save_file,
        run_time_array=np.asarray(run_time_array),
        log_Z_mean_array=np.asarray(log_Z_mean_array),
        log_Z_uncert_array=np.asarray(log_Z_uncert_array),
        num_likelihood_evals_array=np.asarray(num_likelihood_evals_array),
        total_num_samples_array=np.asarray(total_num_samples_array),
        total_num_phantom_samples_array=np.asarray(total_num_phantom_samples_array),
        s_array=np.asarray(s_array),
        k_array=np.asarray(k_array),
        c_array=np.asarray(c_array),
        true_logZ=true_logZ
    )
