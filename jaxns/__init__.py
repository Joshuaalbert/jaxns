"""Nested sampling with JAX."""
import logging

logger = logging.getLogger('jaxns')

logging.basicConfig(format='%(levelname)s[%(asctime)s]: %(message)s', level=logging.INFO)

from jaxns.nested_sampler import *
from jaxns.static_nested_sampler import *
from jaxns.model import *
from jaxns.types import *
from jaxns.plotting import *
from jaxns.prior import *
from jaxns.utils import *
