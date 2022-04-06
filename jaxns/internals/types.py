from typing import NamedTuple, Dict, Any, Union

from jax import numpy as jnp


class SignedLog(NamedTuple):
    """
    Represents a signed value in log-space
    """
    log_abs_val: jnp.ndarray
    sign: Union[jnp.ndarray, Any]


class Reservoir(NamedTuple):
    """
    Holds the reservoir of new samples before merging.
    """
    points_U: jnp.ndarray  # [N, M] init_U in unit cube of live init_U
    points_X: Dict[str, jnp.ndarray]  # [N, M] init_U in constrained space of live init_U in dict structure
    log_L_constraint: jnp.ndarray  # [N] constraint that sample was sampled within
    log_L_samples: jnp.ndarray  # [N] log likelihood of live init_U
    num_likelihood_evaluations: jnp.ndarray  # [N] how many times the likelihood was evaluated
    num_slices: jnp.ndarray  # [N] how many slices were taken


class SampleCollection(NamedTuple):
    """
    Arrays to hold samples taken from the reservoir.
    """
    # TODO: allow any pytree for points_X
    points_U: jnp.ndarray  # [max_samples, U_ndims] -- The X-valued samples
    points_X: Dict[str, jnp.ndarray]  # {RV_name:[max_samples] + RV_shape} -- The X-valued samples
    log_L_samples: jnp.ndarray  # [max_samples] -- The log-likelihood of sample.
    log_L_constraint: jnp.ndarray  # [max_samples] -- the log-likelihood constraint sampled within to acquire sample.
    num_likelihood_evaluations: jnp.ndarray  # [max_samples] -- How many likelihood evaluations were required to obtain sample
    log_dZ_mean: jnp.ndarray  # [max_samples] -- The log mean evidence difference of the sample
    log_X_mean: jnp.ndarray  # [max_samples] -- The log mean enclosed prior volume of sample
    num_live_points: jnp.ndarray  # [max_samples] -- How many live points were taken for the samples.
    num_slices: jnp.ndarray  # [max_samples] how many slices were taken


class EvidenceCalculation(NamedTuple):
    """
    Contains a running estimate of evidence and related quantities.
    """
    log_X_mean: jnp.ndarray
    log_X2_mean: jnp.ndarray
    log_Z_mean: jnp.ndarray
    log_ZX_mean: jnp.ndarray
    log_Z2_mean: jnp.ndarray
    log_dZ2_mean: jnp.ndarray


class ThreadStats(NamedTuple):
    """
    Calculates some running statistics
    """
    evidence: jnp.ndarray
    evidence_uncert: jnp.ndarray
    ess: jnp.ndarray
    log_L_max: jnp.ndarray
    num_likelihood_evaluations: jnp.ndarray


class TerminationStats(NamedTuple):
    previous_evidence_calculation: EvidenceCalculation
    current_evidence_calculation: EvidenceCalculation
    num_samples: jnp.ndarray
    num_steps: jnp.ndarray
    log_L_contour_max: jnp.ndarray
    patience_steps: jnp.ndarray


class NestedSamplerState(NamedTuple):
    key: jnp.ndarray
    done: jnp.ndarray
    sample_collection: SampleCollection  # Arrays to hold samples taken from the reservoir.
    evidence_calculation: EvidenceCalculation  # holds running calculation of evidence
    log_L_contour: jnp.ndarray  # the contour of the sampler
    step_idx: jnp.ndarray  # the step index of the algorithm, where one step is a single consumption step.
    sample_idx: jnp.ndarray  # the sample index, pointing to the next empty sample slot
    thread_stats: ThreadStats
    patience_steps: jnp.ndarray  # how many steps since goal incremental improvement
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination


class NestedSamplerResults(NamedTuple):
    log_Z_mean: jnp.ndarray  # estimate of E[log(Z)]
    log_Z_uncert: jnp.ndarray  # estimate of StdDev[log(Z)]
    ESS: jnp.ndarray  # estimate of Kish's effective sample size
    H_mean: jnp.ndarray  # estimate of E[int log(L) L dp/Z]
    samples: Dict[str, jnp.ndarray]  # Dict of arrays with leading dimension num_samples
    log_L_samples: jnp.ndarray  # log(L) of each sample
    log_dp_mean: jnp.ndarray  # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
    log_X_mean: jnp.ndarray  # log(E[X]) of each sample
    num_live_points_per_sample: jnp.ndarray  # how many live points were taken for the samples.
    num_likelihood_evaluations_per_sample: jnp.ndarray  # how many likelihood evaluations were made per sample.
    num_slices_per_sample: jnp.ndarray  # how many slices were taken
    total_num_samples: jnp.ndarray  # int, the total number of samples collected.
    total_num_likelihood_evaluations: jnp.ndarray  # how many likelihood evaluations were made in total,
    # sum of num_likelihood_evaluations_per_sample.
    log_efficiency: jnp.ndarray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination
    thread_stats: ThreadStats


class GlobalOptimiserState(NamedTuple):
    key: jnp.ndarray
    done: jnp.ndarray
    reservoir: Reservoir  # Arrays to hold samples taken from the reservoir.
    num_steps: jnp.ndarray  # the step index of the algorithm, where one step is a single consumption step.
    num_samples: jnp.ndarray  # how many samples have been drawn
    num_likelihood_evaluations: jnp.ndarray  # how many times the likelihood has been evaluated
    patience_steps: jnp.ndarray  # how many steps since goal incremental improvement
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination


class GlobalOptimiserResults(NamedTuple):
    samples: Dict[str, jnp.ndarray]  # Dict of arrays with leading dimension num_samples
    total_num_samples: jnp.ndarray  # int, the total number of samples collected.
    log_L_samples: jnp.ndarray  # log(L) of each sample
    total_num_likelihood_evaluations: jnp.ndarray  # how many likelihood evaluations were made in total,
    log_efficiency: jnp.ndarray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: jnp.ndarray  # this will be an int reflecting the reason for termination
    log_L_max: jnp.ndarray  # maximum likelihood value obtained
    sample_L_max: Dict[str, jnp.ndarray]  # sample at the log_L_max point
