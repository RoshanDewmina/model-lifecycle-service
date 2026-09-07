from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from .lifecycle import (
    GateRejected,
    drift_diagnostic,
    load_active,
    promote,
    rollback,
    train_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modelctl")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train and evaluate an immutable candidate")
    train.add_argument("--version", required=True)
    train.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    train.add_argument("--seed", type=int, default=20260907)
    train.add_argument("--model", choices=("logistic", "dummy"), default="logistic")
    train.add_argument("--receipt", type=Path)

    for name in ("promote", "rollback"):
        item = sub.add_parser(name)
        item.add_argument("--version", required=True)
        item.add_argument("--registry", type=Path, default=Path("artifacts/registry"))

    drift = sub.add_parser("drift")
    drift.add_argument("--registry", type=Path, default=Path("artifacts/registry"))
    drift.add_argument("--simulate", action="store_true", required=True)

    ensure = sub.add_parser("ensure-demo")
    ensure.add_argument("--registry", type=Path, default=Path("artifacts/registry"))

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8115)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "train":
            result = train_candidate(
                args.registry,
                args.version,
                seed=args.seed,
                model_kind=args.model,
                receipt_path=args.receipt,
                command=" ".join(["uv run modelctl", *(__import__("sys").argv[1:])]),
            )
        elif args.command == "promote":
            result = promote(args.registry, args.version)
        elif args.command == "rollback":
            result = rollback(args.registry, args.version)
        elif args.command == "drift":
            result = drift_diagnostic(args.registry, simulate=args.simulate)
        elif args.command == "ensure-demo":
            try:
                result = load_active(args.registry).metadata
            except Exception:
                result = train_candidate(args.registry, "wine-logreg-v1")
                promote(args.registry, "wine-logreg-v1")
        elif args.command == "serve":
            uvicorn.run("model_lifecycle.api:app", host=args.host, port=args.port)
            return
        else:
            raise AssertionError("unreachable")
    except GateRejected as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

