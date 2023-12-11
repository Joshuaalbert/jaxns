from jaxns.framework.bases import BaseAbstractModel
from jaxns.nested_sampler.abc import AbstractNestedSampler


class BaseAbstractNestedSampler(AbstractNestedSampler):
    def __init__(self, model: BaseAbstractModel, max_samples: int):
        self.model = model
        self.max_samples = max_samples
