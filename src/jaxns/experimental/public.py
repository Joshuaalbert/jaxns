import dataclasses
from typing import Optional

from jaxlib import xla_client

from jaxns.experimental import SimpleGlobalOptimisation, GlobalOptimisationTerminationCondition, \
    GlobalOptimisationResults
from jaxns.experimental.global_optimisation import plot_progress, go_summary
from jaxns.framework.bases import BaseAbstractModel
from jaxns.internals.types import PRNGKey
from jaxns.samplers import UniDimSliceSampler

__all__ = [
    'GlobalOptimisation',
    'DefaultGlobalOptimisation'
]


@dataclasses.dataclass(eq=False)
class GlobalOptimisation:
    """
    A global optimiser using nested sampling as the core algorithm. Can easily globally optimise complex models, with
    curving degeneracies and multimodal structure. Highly parallelisable. Recommended to use gradient information by
    setting gradient_slice=True.

    Note, the log-likelihood over the model is maximised NOT the posterior. The prior acts as the search space prior,
    by constraining the search space and giving search preference to regions of high prior probability. Thus, the
    prior should encode your prior belief about where you think the global maximum is located.

    Args:
        model: a model to perform global optimisation on over the sample space.
        num_search_chains: number of search chains to use.
        s: number of slices to use per dimension.
        k: number of phantom samples to use.
        gradient_slice: if true use gradient information to improve. Default True.
        shell_frac: fraction of the shell to discard in parallel.
        devices: devices to use for parallel sharded computation. Default all available devices.
        verbose: whether to print verbose output. Default False.
    """
    model: BaseAbstractModel
    num_search_chains: Optional[int] = None
    s: Optional[int] = None
    k: Optional[int] = None
    gradient_slice: bool = True
    shell_frac: Optional[float] = None
    devices: Optional[xla_client.Device] = None
    verbose: bool = False

    def __post_init__(self):
        if self.num_search_chains is None:
            if self.gradient_slice:
                self.num_search_chains = self.model.U_ndims * 15
            else:
                self.num_search_chains = self.model.U_ndims * 100
        if self.s is None:
            if self.gradient_slice:
                self.s = 2
            else:
                self.s = 10
        if self.shell_frac is None:
            if self.gradient_slice:
                self.shell_frac = 0.5
            else:
                self.shell_frac = 0.5
        if self.k is None:
            if self.gradient_slice:
                self.k = self.model.U_ndims * self.s - 1
            else:
                self.k = self.model.U_ndims * self.s - 1

        sampler = UniDimSliceSampler(
            model=self.model,
            num_slices=self.model.U_ndims * int(self.s),
            num_phantom_save=int(self.k),
            midpoint_shrink=True,
            perfect=True,
            gradient_slice=self.gradient_slice
        )

        self._global_optimiser = SimpleGlobalOptimisation(
            sampler=sampler,
            num_search_chains=int(self.num_search_chains),
            shell_frac=float(self.shell_frac),
            model=self.model,
            devices=self.devices,
            verbose=self.verbose
        )

        self.summary = go_summary
        self.plot_progress = plot_progress

    def __call__(self, key: PRNGKey,
                 term_cond: Optional[GlobalOptimisationTerminationCondition] = None,
                 finetune: bool = False) -> GlobalOptimisationResults:
        """
        Runs the global optimisation.

        Args:
            key: PRNGKey
            term_cond: termination condition
            finetune: whether to use gradient-based fine-tune. Default False because not all models have gradients.

        Returns:
            results of the global optimisation
        """
        if term_cond is None:
            term_cond = GlobalOptimisationTerminationCondition(
                min_efficiency=3e-2
            )
        termination_reason, state = self._global_optimiser._run(key, term_cond)
        results = self._global_optimiser._to_results(termination_reason, state)
        if finetune:
            results = self._global_optimiser._gradient_descent(results=results)
        return results


DefaultGlobalOptimisation = GlobalOptimisation
