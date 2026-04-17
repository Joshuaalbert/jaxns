from __future__ import annotations

import argparse

from jaxns.fabric.node import FabricAddresses, build_worker_process_manager


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch JAXNS fabric likelihood workers.")
    parser.add_argument("--ctl-pub-addr", required=True)
    parser.add_argument("--ack-rep-addr", required=True)
    parser.add_argument("--backend-addr", required=True)
    parser.add_argument("--service-factory", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--start-method", default="forkserver")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    addresses = FabricAddresses(
        ctl_pub_addr=args.ctl_pub_addr,
        ack_rep_addr=args.ack_rep_addr,
        frontend_addr="",
        backend_addr=args.backend_addr,
    )
    manager = build_worker_process_manager(
        addresses=addresses,
        service_factory=args.service_factory,
        num_workers=args.num_workers,
        profile=args.profile,
        start_method=args.start_method,
    )
    manager.start_all()
    try:
        manager.wait_all()
    finally:
        manager.stop_all()
        manager.print_tracebacks()
    return 0
