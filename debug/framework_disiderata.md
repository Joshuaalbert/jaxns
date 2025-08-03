We'd like to be able to beter pass data into to models that currently
relying on closure.

Better is something like this:

```python
from jaxns import Model, NestedSampler


def prior_model(..., *data):
    ...


def log_likelihood(..., *data):
    ...


model = Model(
    prior_model=prior_model,
    log_likelihood=log_likelihood
)

ns = NestedSampler(model)
ns.run(args=data)

```

This passes all data to the model which is necessary for execution. The model can then be parametrized with the data it
needs to run.

We probably also need to pass in parameters to the model, rather than close over them. Thus something like:

```python
output = model(params, *data)