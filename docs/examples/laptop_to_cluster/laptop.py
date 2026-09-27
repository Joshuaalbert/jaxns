"""Run locally until the accuracy goal is met or a pause is requested."""

from pathlib import Path

import jax
from problem import model, run_settings

from jaxns.core import NestedSampler
from jaxns.state import State


def goal_cond(state: State) -> bool:
    return (
        Path("PAUSE").exists()
        or float(state.expected_log_Z_uncert) <= 0.05
    )


if __name__ == "__main__":
    runner = NestedSampler(model=model, **run_settings)
    state = runner.run_until_goal(
        goal_cond=goal_cond,
        key=jax.random.PRNGKey(42),
        checkpoint_dir="checkpoints/laptop",
        checkpoint_cadence=60.0,
    )
    # Results support analysis, but continuation needs the complete State.
    state.save("laptop-state.pkl")
    state.to_result().summary()
