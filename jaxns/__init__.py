"""Nested sampling with JAX."""
import logging

logger = logging.getLogger('jaxns')

logging.basicConfig(format='%(levelname)s[%(asctime)s]: %(message)s', level=logging.INFO)

from jaxns.new_code.utils import resample, marginalise_static, marginalise_dynamic, summary, evidence_posterior_samples, \
    analytic_log_evidence, analytic_posterior_samples, save_results, load_results
from jaxns.new_code.plotting import (plot_diagnostics,
                                     plot_cornerplot)
