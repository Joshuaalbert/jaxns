"""Resumable orchestration for the phantom-conditioning paper experiment.

This is the single user entrypoint around the importable scientific cases and
the existing stateful runner, exact phantom-prefix reducer, summaries, and
problem figure. It deliberately contains no model definitions or evidence
arithmetic of its own.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = REPO_ROOT / "benchmarks/paper_phantom_conditioning"
DEFAULT_OUTPUT_ROOT = (
    BENCHMARK_ROOT
    / "raw/phantom_percentage_sweep_10d_state_mc2048_prior1"
)
DEFAULT_STATE_ROOT = Path(
    "/data/jaxns-paper-phantom-conditioning/states-prior1"
)
PHASE_ORDER = (
    "core",
    "analysis",
    "verify",
    "summary",
    "figures",
)


@dataclass(frozen=True, slots=True)
class Protocol:
    """Names and defaults owned by the existing paper runner."""

    cases: tuple[str, ...]
    directions: tuple[str, ...]
    seed_count: int


@dataclass(frozen=True, slots=True)
class Command:
    """One labelled child process in the bounded execution pool."""

    label: str
    argv: tuple[str, ...]


def _protocol() -> Protocol:
    """Load the paper's cases and fixed isotropic direction arm lazily."""
    # Load this checkout's benchmark definitions for orchestration. Child
    # runners still honour an explicit implementation PYTHONPATH, allowing
    # experimental sampler branches to share the same benchmark definitions.
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.paper_phantom_conditioning.cases import PAPER_CASES
    from benchmarks.paper_phantom_conditioning.summarise_directions import (
        DIRECTIONS,
    )

    return Protocol(
        cases=tuple(PAPER_CASES),
        directions=tuple(DIRECTIONS),
        # Canonical seed identity is part of the published experimental
        # design, while each child runner remains the validating authority.
        seed_count=30,
    )


def _phases(values: list[str] | None) -> tuple[str, ...]:
    """Expand ``all`` while rejecting ambiguous mixed selections."""
    selected = ["all"] if values is None else values
    if "all" in selected:
        if len(selected) != 1:
            raise ValueError("phase 'all' cannot be combined with other phases.")
        return PHASE_ORDER
    unknown = sorted(set(selected) - set(PHASE_ORDER))
    if unknown:
        raise ValueError(f"Unknown phases: {unknown}.")
    return tuple(phase for phase in PHASE_ORDER if phase in selected)


def _seeds(value: str, protocol: Protocol) -> list[int]:
    """Return one sorted canonical seed subset shared by every arm."""
    if value == "canonical":
        seeds = list(range(protocol.seed_count))
    elif ":" in value:
        start, stop = value.split(":", maxsplit=1)
        seeds = list(range(int(start), int(stop)))
    else:
        seeds = [int(seed) for seed in value.split(",")]
    if not seeds:
        raise ValueError("At least one paper seed is required.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Paper seed selection contains duplicates.")
    if any(seed < 0 or seed >= protocol.seed_count for seed in seeds):
        raise ValueError(
            f"Paper seeds must be in 0--{protocol.seed_count - 1}."
        )
    return sorted(seeds)


def _output_path(output_root: Path, direction: str, case: str) -> Path:
    """Use one append-only manifest per independent experiment arm."""
    return output_root / f"{direction}_{case}.jsonl"


def _cell_commands(
        *,
        phase: str,
        protocol: Protocol,
        seeds: list[int],
        state_root: Path,
        output_root: Path,
        mc_draws: int | None,
        mc_workers: int,
        phantom_seeding: str,
) -> list[Command]:
    """Build runner commands at the finest phase-safe concurrency."""
    runner = BENCHMARK_ROOT / "run.py"
    # Core checkpoints occupy disjoint seed directories, so expose each one
    # to the bounded process pool and let large hosts run the independent NS
    # jobs concurrently. Analysis appends to one manifest per arm; retain one
    # writer for each arm there so two processes can never replace the same
    # append-only file from stale snapshots.
    seed_groups = (
        tuple((seed,) for seed in seeds)
        if phase == "core"
        else (tuple(seeds),)
    )
    commands = []
    for direction in protocol.directions:
        for case in protocol.cases:
            for seed_group in seed_groups:
                seed_text = ",".join(str(seed) for seed in seed_group)
                argv = [
                    sys.executable,
                    str(runner),
                    "--case",
                    case,
                    "--direction",
                    direction,
                    "--phantom-seeding",
                    phantom_seeding,
                    "--seeds",
                    seed_text,
                    "--mc-batch-size",
                    "1",
                    "--mc-workers",
                    str(mc_workers),
                    "--phase",
                    phase,
                    "--output",
                    str(_output_path(output_root, direction, case)),
                    "--state-root",
                    str(state_root),
                ]
                # Omitting this option delegates the scientific default to
                # the existing runner rather than copying it here.
                if mc_draws is not None:
                    argv.extend(("--mc-draws", str(mc_draws)))
                label = f"{phase}:{direction}:{case}"
                if phase == "core":
                    label = f"{label}:seed-{seed_group[0]:02d}"
                commands.append(Command(
                    label=label,
                    argv=tuple(argv),
                ))
    return commands


def _print_command(command: Command) -> None:
    """Render the exact command in a reusable shell form."""
    print(f"[{command.label}] {shlex.join(command.argv)}", flush=True)


def _stop_processes(active: dict[int, tuple[subprocess.Popen, Command]]) -> None:
    """Stop only children launched by this orchestration process."""
    for process, _ in active.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if all(process.poll() is not None for process, _ in active.values()):
            return
        time.sleep(0.1)
    for process, _ in active.values():
        if process.poll() is None:
            process.kill()
    for process, _ in active.values():
        process.wait()


def _run_parallel(
        commands: list[Command],
        process_limit: int,
        dry_run: bool,
) -> None:
    """Run cells concurrently while retaining ownership of every child."""
    if dry_run:
        for command in commands:
            _print_command(command)
        return

    pending = deque(commands)
    active: dict[int, tuple[subprocess.Popen, Command]] = {}
    try:
        while pending or active:
            while pending and len(active) < process_limit:
                command = pending.popleft()
                _print_command(command)
                process = subprocess.Popen(
                    command.argv,
                    cwd=REPO_ROOT,
                )
                active[process.pid] = (process, command)

            completed = []
            for process_id, (process, command) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                completed.append(process_id)
                if return_code != 0:
                    _stop_processes(active)
                    raise RuntimeError(
                        f"{command.label} failed with exit status "
                        f"{return_code}."
                    )
                print(f"[{command.label}] complete", flush=True)
            for process_id in completed:
                active.pop(process_id, None)
            if active and not completed:
                time.sleep(0.2)
    except BaseException:
        _stop_processes(active)
        raise


def _run_one(
        command: Command,
        dry_run: bool,
        working_directory: Path = REPO_ROOT,
) -> None:
    """Run one post-processing command with inherited diagnostic output."""
    _print_command(command)
    if dry_run:
        return
    result = subprocess.run(
        command.argv,
        cwd=working_directory,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{command.label} failed with exit status {result.returncode}."
        )


def _publish_text(path: Path, value: str) -> None:
    """Atomically publish a small derived text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _records(protocol: Protocol, output_root: Path) -> list[Path]:
    """Return all arm manifests in one deterministic order."""
    return [
        _output_path(output_root, direction, case)
        for case in protocol.cases
        for direction in protocol.directions
    ]


def _require_records(paths: list[Path]) -> None:
    """Fail before post-processing if any experiment arm is absent."""
    missing = [path for path in paths if not path.is_file()]
    if missing:
        listing = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing experiment records:\n{listing}\n"
            "Run the analysis phase first."
        )


def _verify(
        protocol: Protocol,
        state_root: Path,
        output_root: Path,
        dry_run: bool,
) -> None:
    """Independently deserialize and checksum every recorded State."""
    paths = _records(protocol, output_root)
    if not dry_run:
        _require_records(paths)
    _run_one(Command(
        label="verify",
        argv=(
            sys.executable,
            str(BENCHMARK_ROOT / "verify_states.py"),
            *(str(path) for path in paths),
            "--state-root",
            str(state_root),
        ),
    ), dry_run)


def _summarise(
        protocol: Protocol,
        output_root: Path,
        dry_run: bool,
) -> None:
    """Write the paired phantom-prefix summary and complete LaTeX rows."""
    paths = _records(protocol, output_root)
    if not dry_run:
        _require_records(paths)
    summary = output_root / "summary.json"
    tables = output_root / "tables.tex"
    temporary = output_root / f".summary.{uuid4().hex}.json"
    command = Command(
        label="summary",
        argv=(
            sys.executable,
            str(BENCHMARK_ROOT / "summarise_directions.py"),
            *(str(path) for path in paths),
            "--json-output",
            str(temporary if not dry_run else summary),
        ),
    )
    _print_command(command)
    if dry_run:
        print(f"[summary] write LaTeX rows to {tables}", flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            command.argv,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"summary failed with exit status {result.returncode}:\n"
                f"{detail}"
            )
        if not temporary.is_file():
            raise RuntimeError("Summary command did not create its JSON output.")
        os.replace(temporary, summary)
        _publish_text(tables, result.stdout)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(f"[summary] wrote {summary} and {tables}", flush=True)


def _figures(dry_run: bool) -> None:
    """Run the current paper problem-image generator unchanged."""
    _run_one(Command(
        label="figures",
        argv=(
            sys.executable,
            str(BENCHMARK_ROOT / "plot_problems.py"),
        ),
    ), dry_run)


def _compile_pdf(dry_run: bool) -> None:
    """Compile paper.tex, running BibTeX only when its aux requests it."""
    executable = "pdflatex" if dry_run else shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required by --compile-pdf.")
    pdf_command = Command(
        label="pdf",
        argv=(
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "paper.tex",
        ),
    )
    _run_one(pdf_command, dry_run, PAPER_ROOT)
    if dry_run:
        print("[pdf] run bibtex if paper.aux requests a bibliography", flush=True)
        _run_one(pdf_command, True, PAPER_ROOT)
        _run_one(pdf_command, True, PAPER_ROOT)
        return

    auxiliary = PAPER_ROOT / "paper.aux"
    if auxiliary.is_file() and "\\bibdata" in auxiliary.read_text(
            encoding="utf-8",
            errors="replace",
    ):
        bibtex = shutil.which("bibtex")
        if bibtex is None:
            raise RuntimeError("bibtex is required by paper.aux.")
        _run_one(Command(
            label="bibtex",
            argv=(bibtex, "paper"),
        ), False, PAPER_ROOT)
    _run_one(pdf_command, False, PAPER_ROOT)
    _run_one(pdf_command, False, PAPER_ROOT)


def main() -> None:
    """Parse the orchestration request and execute phases in order."""
    parser = argparse.ArgumentParser(
        description=(
            "Run, analyse, verify, summarise, and plot the full "
            "phantom-conditioning paper suite with resumable experiment "
            "phases."
        ),
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", *PHASE_ORDER),
        help=(
            "Phase to run; repeat for multiple phases. The default 'all' "
            "runs core, analysis, verification, summary, and figures."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="canonical",
        help="'canonical', start:stop, or one comma-separated canonical subset.",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=DEFAULT_STATE_ROOT,
    )
    parser.add_argument(
        "--phantom-seeding",
        choices=("off", "on"),
        default="off",
        help="Whether retained phantoms may seed later constrained chains.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Maximum simultaneously running JAXNS child processes.",
    )
    parser.add_argument(
        "--mc-workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help=(
            "Host threads per analysis process for scalar exact MC draws; "
            "total threads can reach processes * mc-workers."
        ),
    )
    parser.add_argument(
        "--mc-draws",
        type=int,
        help="Exact prefix MC draws per State; defaults to the runner protocol.",
    )
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="Compile paper.tex after the selected phases succeed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print every command without writing or running it.",
    )
    args = parser.parse_args()

    protocol = _protocol()
    phases = _phases(args.phase)
    seeds = _seeds(args.seeds, protocol)
    if args.processes < 1:
        raise ValueError("processes must be positive.")
    if args.mc_workers < 1:
        raise ValueError("mc-workers must be positive.")
    if args.mc_draws is not None and args.mc_draws < 2:
        raise ValueError("mc-draws must provide at least two draws per State.")
    if "summary" in phases and seeds != list(range(protocol.seed_count)):
        raise ValueError(
            "The paired paper summary requires canonical seeds 0--29; "
            "select core/analysis phases explicitly for a partial run."
        )

    if not args.dry_run:
        args.state_root.mkdir(parents=True, exist_ok=True)
        args.output_root.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        if phase in ("core", "analysis"):
            _run_parallel(
                _cell_commands(
                    phase=phase,
                    protocol=protocol,
                    seeds=seeds,
                    state_root=args.state_root,
                    output_root=args.output_root,
                    mc_draws=args.mc_draws,
                    mc_workers=args.mc_workers,
                    phantom_seeding=args.phantom_seeding,
                ),
                args.processes,
                args.dry_run,
            )
        elif phase == "verify":
            _verify(
                protocol,
                args.state_root,
                args.output_root,
                args.dry_run,
            )
        elif phase == "summary":
            _summarise(protocol, args.output_root, args.dry_run)
        elif phase == "figures":
            _figures(args.dry_run)

    if args.compile_pdf:
        _compile_pdf(args.dry_run)


if __name__ == "__main__":
    main()
