"""Human-readable summaries for nested-sampling results."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import jax
import numpy as np
from jax import numpy as jnp
from jaxctx import CtxParams

if TYPE_CHECKING:
    from jaxns.results import NestedSamplerResults


def _bit_mask(int_mask, width=8):
    """
    Convert an integer mask into a bit-mask. I.e. convert an integer into list of left-starting bits.

    Examples:

    1 -> [1,0,0,0,0,0,0,0]
    2 -> [0,1,0,0,0,0,0,0]
    3 -> [1,1,0,0,0,0,0,0]

    Args:
        int_mask: int
        width: number of output bits

    Returns:
        List of bits from left
    """
    return list(map(int, '{:0{size}b}'.format(int_mask, size=width)))[::-1]


def _summary(results: NestedSamplerResults, f_obj: str | TextIO | None = None):
    """
    Gives a summary of the results of a nested sampling run.

    Args:
        results (NestedSamplerResults): Nested sampler result
        f_obj: file-like object to write summary to. If None, prints to stdout.
    """

    main_s = []

    def _print(s):
        if f_obj is None:
            # It goes to file instead
            print(s)
        main_s.append(s)

    def _round(v, uncert_v):
        v = float(v)
        uncert_v = float(uncert_v)
        try:
            sig_figs = -int(f"{uncert_v:e}".split('e')[1]) + 1
            return round(float(v), sig_figs)
        except (OverflowError, ValueError):
            return float(v)

    def _print_termination_reason(_termination_reason: int):
        termination_bit_mask = _bit_mask(int(_termination_reason), width=13)

        for bit, condition in zip(termination_bit_mask, [
            'Reached max samples',
            'Evidence uncertainty low enough',
            'Small remaining evidence',
            'Reached ESS',
            "Used max num likelihood evaluations",
            'Likelihood supremum reached',
            'Likelihood contour reached',
            'Sampler efficiency too low',
            'All live-points are on a single plateau (sign of possible precision error)',
            'relative spread of live points < rtol',
            'absolute spread of live points < atol',
            'no seed points left (consider decreasing shell_fraction)',
            'XL < max(XL) * peak_XL_frac'
        ]):
            if bit == 1:
                _print(condition)

    _print("--------")
    _print("Termination Conditions:")
    if np.size(results.termination_reason) > 1:  # Reasons for each parallel sampler
        print(results.termination_reason)
        for sampler_idx in range(np.size(results.termination_reason)):
            _print(f"Sampler {sampler_idx}:")
            _print_termination_reason(int(results.termination_reason[sampler_idx]))
    else:
        _print_termination_reason(int(results.termination_reason))
    _print("--------")
    _print(f"likelihood evals: {int(results.total_num_likelihood_evaluations):d}")
    _print(f"classic samples: {int(results.total_num_samples):d}")
    _print(f"phantom samples: {int(results.total_phantom_samples):d}")
    _print(
        f"likelihood evals / sample: {float(results.total_num_likelihood_evaluations / results.total_num_samples):.1f}"
    )
    _print("--------")
    _print(
        f"logZ (classic)={_round(results.log_Z_mean, results.log_Z_uncert)} +- {_round(results.log_Z_uncert, results.log_Z_uncert)}"
    )
    if results.total_phantom_samples > 0:
        log_Z_samples = results.sample_evidence(num_samples=512)
        _log_Z_samples_mean = jnp.nanmean(log_Z_samples)
        _log_Z_samples_uncert = jnp.nanstd(log_Z_samples)
        _print(
            f"logZ (with phantom)={_round(_log_Z_samples_mean, _log_Z_samples_uncert)} +- {_round(_log_Z_samples_uncert, _log_Z_samples_uncert)}"
        )
    _print(
        f"max(logL)={_round(results.log_L_supremum, results.log_Z_uncert)}"
    )
    # _print("H={} +- {}".format(
    #     _round(results.H_mean, results.H_uncert), _round(results.H_uncert, results.H_uncert)))
    _print(
        f"H={_round(results.H_mean, 0.1)}"
    )
    _print(
        f"effective sample size (classic)={results.ess:.1f}"
    )
    if results.total_phantom_samples > 0:
        ess_with_phantom = results.ess_with_phantom()
        _print(
            f"effective sample size (with phantom)={ess_with_phantom:.1f}"
        )
    _print(
        f"likelihood evals / ess(classic): {float(results.total_num_likelihood_evaluations / results.ess):.1f}"
    )
    if results.total_phantom_samples > 0:
        _print(
            f"likelihood evals / ess(with phantom): {float(results.total_num_likelihood_evaluations / ess_with_phantom):.1f}"
        )

    def moments(x):
        x2 = jax.tree.map(jnp.square, x)
        return x, x2

    x_mean, x2_mean = jax.tree.map(np.asarray, results.integrate_fn_over_posterior(moments))
    x_std: CtxParams = jax.tree.map(lambda m, m2: np.sqrt(np.maximum(0., m2 - np.square(m))), x_mean, x2_mean)

    for name, _ in x_std.iter_items():
        _x_mean = x_mean.get_dotted(name).reshape((-1,))
        _x_std = x_std.get_dotted(name).reshape((-1,))
        _x_map = results.X_map.get_dotted(name).reshape((-1,))
        _x_ml = results.X_supremum.get_dotted(name).reshape((-1,))
        ndims = _x_mean.shape[0]
        _print("--------")
        var_name = name if ndims == 1 else f"{name}[#]"
        _print(
            f"{var_name}: mean +- std.dev. | MAP est. | max(L) est."
        )
        for dim in range(ndims):
            _uncert = _x_std[dim]
            # two sig-figs based on uncert
            sig_figs = 1 - int(f"{_uncert:e}".split('e')[1])

            def _round(ar, digits=sig_figs):
                return round(float(ar), digits)

            _print("{}: {} +- {} | {} | {}".format(
                name if ndims == 1 else f"{name}[{dim}]",
                _round(_x_mean[dim]), _round(_uncert),
                _round(_x_map[dim]),
                _round(_x_ml[dim])
            ))
    _print("--------")
    if f_obj is not None:
        out = "\n".join(main_s)
        if isinstance(f_obj, (str, Path)):
            with open(f_obj, 'w') as f:
                f.write(out)
        elif isinstance(f_obj, io.TextIOBase):
            f_obj.write(out)
        else:
            raise TypeError(f"Invalid f_obj: {type(f_obj)}")

