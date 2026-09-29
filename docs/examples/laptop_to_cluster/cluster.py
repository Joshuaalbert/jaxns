"""Resume on the coordinator host, even before workers have joined."""

from pathlib import Path

from problem import run_settings

from jaxns.distributed_core import DistributedNestedSampler, DistributedState
from jaxns.state import State


def goal_cond(state: State) -> bool:
    return (
        Path("PAUSE").exists()
        or float(state.expected_log_Z_uncert) <= 0.02
    )


if __name__ == "__main__":
    local_state = State.load("laptop-state.pkl")
    handoff = DistributedState.from_state(local_state)
    runner = DistributedNestedSampler(
        model=local_state.model,
        coordinator_port=5555,
        **run_settings,
    )
    # On subsequent invocations, the automatic distributed checkpoint takes
    # precedence over this original laptop handoff. Keep the two directories
    # separate because they store different continuation objects.
    completed = runner.resume_until_goal(
        handoff,
        goal_cond=goal_cond,
        checkpoint_dir="checkpoints/cluster",
        checkpoint_cadence=60.0,
    )
    completed.to_result().summary()
