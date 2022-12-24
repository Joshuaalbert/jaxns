from typing import NamedTuple, Optional

from etils.array_types import FloatArray, IntArray, ui64, PRNGKey, BoolArray
from jaxns.internals.types import int_type

from jaxns.new_code.prior import UType, XType
import jax.numpy as jnp

class Sample(NamedTuple):
    """
    Holds the reservoir of new samples before merging.
    """
    # TODO: allow any pytree for points_X
    point_U: UType  # [d] sample in U-space
    point_X: XType
    log_L_constraint: FloatArray  # constraint that sample was sampled uniformly within
    log_L: FloatArray  # log likelihood of the sample
    num_likelihood_evaluations: IntArray  # how many times the likelihood was evaluated to produce this sample
    num_slices: ui64  # the number of slices for sliced points.
    iid: BoolArray  # whether the sample is exactly iid sampled from within the likelihood constraint


class Reservoir(NamedTuple):
    """
    Holds the reservoir samples, has a leading [n] dimension.
    """
    # TODO: allow any pytree for points_X
    point_U: UType  # [d] sample in U-space
    point_X: XType
    log_L_constraint: FloatArray  # constraint that sample was sampled uniformly within
    log_L: FloatArray  # log likelihood of the sample
    num_likelihood_evaluations: ui64  # how many times the likelihood was evaluated to produce this sample
    num_slices: ui64  # the number of slices for sliced points.
    iid: BoolArray  # whether the sample is exactly iid sampled from within the likelihood constraint


class SampleStatistics(NamedTuple):
    num_live_points: IntArray  # [max_samples] How many live points were taken for the samples.
    log_dZ_mean: FloatArray  # [max_samples] The log mean evidence difference of the sample
    log_X_mean: FloatArray  # [max_samples] The log mean enclosed prior volume of sample


class SampleCollection(NamedTuple):
    """
    Arrays to hold samples taken from the reservoir.
    """

    sample_idx: ui64  # the sample index, pointing to the next empty sample slot
    reservoir: Reservoir  # reservoir of samples with leading dimension [max_samples]


class EvidenceCalculation(NamedTuple):
    """
    Contains a running estimate of evidence and related quantities.
    """
    log_X_mean: FloatArray
    log_X2_mean: FloatArray
    log_Z_mean: FloatArray
    log_ZX_mean: FloatArray
    log_Z2_mean: FloatArray
    log_dZ2_mean: FloatArray


class NestedSamplerState(NamedTuple):
    key: PRNGKey
    sample_collection: SampleCollection  # Holds all collected samples so far.


class LivePoints(NamedTuple):
    reservoir: Reservoir


class TerminationCondition(NamedTuple):
    ess: Optional[FloatArray] = jnp.inf
    evidence_uncert: Optional[FloatArray] = jnp.inf
    live_evidence_frac: Optional[FloatArray] = 1e-4
    max_samples: Optional[IntArray] = jnp.iinfo(int_type).max
    max_num_likelihood_evaluations: Optional[IntArray] = jnp.iinfo(int_type).max
    log_L_contour: Optional[FloatArray] = jnp.inf


class NestedSamplerResults(NamedTuple):
    log_Z_mean: FloatArray  # estimate of E[log(Z)]
    log_Z_uncert: FloatArray  # estimate of StdDev[log(Z)]
    ESS: FloatArray  # estimate of Kish's effective sample size
    H_mean: FloatArray  # estimate of E[int log(L) L dp/Z]
    samples: XType  # Dict of arrays with leading dimension num_samples
    log_L_samples: FloatArray  # log(L) of each sample
    log_dp_mean: FloatArray  # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
    log_X_mean: FloatArray  # log(E[U]) of each sample
    log_posterior_density: FloatArray # log(P( theta | D )) log posteriori density
    num_live_points_per_sample: IntArray  # how many live points were taken for the samples.
    num_likelihood_evaluations_per_sample: IntArray  # how many likelihood evaluations were made per sample.
    num_slices: IntArray  # how many slices were taken for slice sampled points
    total_num_samples: IntArray  # int, the total number of samples collected.
    total_num_likelihood_evaluations: IntArray  # how many likelihood evaluations were made in total
    num_likelihood_evaluations_per_slice: FloatArray  # how many likeihood evaluations required per slice (for slice samples)
    # sum of num_likelihood_evaluations_per_sample.
    log_efficiency: FloatArray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: IntArray  # this will be an int reflecting the reason for termination
