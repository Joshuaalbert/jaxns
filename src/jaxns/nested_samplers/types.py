from typing import NamedTuple, Optional, List, Union, Callable, TypeVar, Tuple, Any, Dict

import jax
import numpy as np

from jaxns.nested_samplers.pytree import PureDataclassPytree

PRNGKey = jax.Array
Array = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
]
FloatArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    float,  # valid scalars
]
IntArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    int,  # valid scalars
]
BoolArray = Union[
    jax.Array,  # JAX array type
    np.ndarray,  # NumPy array type
    np.bool_, bool,  # valid scalars
]
LikelihoodType = Callable[..., FloatArray]
RandomVariableType = TypeVar('RandomVariableType')
MeasureType = TypeVar('MeasureType')
LikelihoodInputType = Union[Tuple[Any, ...], Any]  # Likelihood conditional variables
UType = FloatArray  # Sample space type
WType = Tuple[FloatArray, ...]
XType = Dict[str, RandomVariableType]  # Prior variable type


class SignedLog(NamedTuple):
    """
    Represents a signed value in log-space
    """
    log_abs_val: jax.Array
    sign: Union[jax.Array, Any]


def isinstance_namedtuple(obj) -> bool:
    """
    Check if object is a namedtuple.

    Args:
        obj: object

    Returns:
        bool
    """
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )


Array.__doc__ = "Type annotation for JAX array-like objects, with no scalar types."

FloatArray.__doc__ = "Type annotation for JAX array-like objects, with float scalar types."

IntArray.__doc__ = "Type annotation for JAX array-like objects, with int scalar types."

BoolArray.__doc__ = "Type annotation for JAX array-like objects, with bool scalar types."


class EvidenceCalculation(NamedTuple):
    """
    Contains a running estimate of evidence and related quantities.
    """
    log_L: FloatArray
    log_X_mean: FloatArray
    log_X2_mean: FloatArray
    log_Z_mean: FloatArray
    log_ZX_mean: FloatArray
    log_Z2_mean: FloatArray
    log_dZ_mean: FloatArray
    log_dZ2_mean: FloatArray


class TerminationCondition(NamedTuple):
    """
    Contains the termination conditions for the nested sampling run.

    Args:
        ess: The effective sample size, if the ESS (Kish's estimate) is greater than this the run will terminate.
        evidence_uncert: The uncertainty in the evidence, if the uncertainty is less than this the run will terminate.
        live_evidence_frac: Depreceated use dlogZ.
        dlogZ: Terminate if log(Z_current + Z_remaining) - log(Z_current) < dlogZ. Default log(1 + 1e-2)
        max_samples: Terminate if the number of samples exceeds this.
        max_num_likelihood_evaluations: Terminate if the number of likelihood evaluations exceeds this.
        log_L_contour: Terminate if this log(L) contour is reached. A contour is reached if any dead point
            has log(L) > log_L_contour. Uncollected live points are not considered.
        efficiency_threshold: Terminate if the efficiency (num_samples / num_likelihood_evaluations) is less than this,
            for the last shrinkage iteration.
        rtol: finish when the relative value 2*|log_L_max - log_L_min|/|log_L_max + log_L_min| < rol
        atol: finish when the absolute |log_L_max - log_L_min| < atol
    """
    ess: Optional[Union[FloatArray, IntArray]] = None
    evidence_uncert: Optional[FloatArray] = None
    live_evidence_frac: Optional[FloatArray] = None  # Depreceated use dlogZ
    dlogZ: Optional[FloatArray] = None  #
    max_samples: Optional[Union[FloatArray, IntArray]] = None
    max_num_likelihood_evaluations: Optional[Union[FloatArray, IntArray]] = None
    log_L_contour: Optional[FloatArray] = None
    efficiency_threshold: Optional[FloatArray] = None
    rtol: Optional[FloatArray] = None
    atol: Optional[FloatArray] = None
    peak_XL_frac: Optional[FloatArray] = None

    def __and__(self, other):
        return TerminationConditionConjunction(conds=[self, other])

    def __or__(self, other):
        return TerminationConditionDisjunction(conds=[self, other])


class TerminationConditionConjunction(NamedTuple):
    """
    A conjunction of termination conditions, e.g. term_cond1 & term_cond2
    """
    conds: List[Union['TerminationConditionDisjunction', 'TerminationConditionConjunction', TerminationCondition]]


class TerminationConditionDisjunction(NamedTuple):
    """
    A disjunction of termination conditions, e.g. term_cond1 | term_cond2
    """
    conds: List[Union['TerminationConditionDisjunction', TerminationConditionConjunction, TerminationCondition]]


class NestedSamplerResults(PureDataclassPytree):
    """
    Results of the nested sampling run.
    """
    log_Z_mean: FloatArray  # estimate of E[log(Z)]
    log_Z_uncert: FloatArray  # estimate of StdDev[log(Z)]
    ESS: FloatArray  # estimate of Kish's effective sample size
    H_mean: FloatArray  # estimate of E[int log(L) L dp/Z]
    samples: XType  # Dict of arrays with leading dimension num_samples
    parametrised_samples: XType  # Dict of arrays with leading dimension num_samples
    U_samples: UType  # Dict of arrays with leading dimension num_samples
    log_L_samples: FloatArray  # log(L) of each sample
    log_dp_mean: FloatArray  # log(E[dZ]) of each sample, where dZ is how much it contributes to the total evidence.
    log_X_mean: FloatArray  # log(E[U]) of each sample
    log_posterior_density: FloatArray  # log(P( theta | D )) log posteriori density
    num_live_points_per_sample: IntArray  # how many live points were taken for the samples.
    num_likelihood_evaluations_per_sample: IntArray  # how many likelihood evaluations were made per sample.
    total_num_samples: IntArray  # int, the total number of samples collected.
    total_phantom_samples: IntArray  # int, the total number of phantom samples collected.
    total_num_likelihood_evaluations: IntArray  # how many likelihood evaluations were made in total
    log_efficiency: FloatArray  # log(total_num_samples / total_num_likelihood_evaluations)
    termination_reason: IntArray  # this will be an int reflecting the reason for termination


class Sample(NamedTuple):
    U_sample: UType  # [..., D] sample in U-space
    log_L_constraint: FloatArray  # [...,] log(L) constraint
    log_L: FloatArray  # [...,] log(L) of sample
    num_likelihood_evaluations: IntArray  # [...,] number of likelihood evaluations


class LivePointCollection(NamedTuple):
    sender_node_idx: IntArray  # [N] with values in [0, N], 0 means root node
    U_sample: UType  # [N, D] sample in U-space
    log_L_constraint: FloatArray  # [N,] log(L) constraint
    log_L: FloatArray  # [N] log(L) of sample
    num_likelihood_evaluations: IntArray  # [N] number of likelihood evaluations


class SampleCollection(NamedTuple):
    sender_node_idx: IntArray  # [N] with values in [0, N], 0 means root node
    log_L: MeasureType  # [N] log(L) of each sample
    U_samples: UType  # [N, D] samples in U-space
    num_likelihood_evaluations: IntArray  # [N] number of likelihood evaluations for each sample
    phantom: BoolArray  # [N] whether the sample is a phantom sample


StaticStandardSampleCollection = SampleCollection


class TerminationRegister(NamedTuple):
    num_samples_used: IntArray
    evidence_calc: EvidenceCalculation
    evidence_calc_with_remaining: EvidenceCalculation
    num_likelihood_evaluations: IntArray
    log_L_contour: FloatArray
    efficiency: FloatArray
    plateau: BoolArray
    no_seed_points: BoolArray
    relative_spread: FloatArray
    absolute_spread: FloatArray
    peak_log_XL: FloatArray


class NestedSamplerState(NamedTuple):
    key: PRNGKey
    next_sample_idx: IntArray  # the next sample insert index <==> the number of samples
    num_samples: IntArray
    sample_collection: StaticStandardSampleCollection
