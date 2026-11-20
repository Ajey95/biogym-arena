"""Collect hosted Sarvam trajectories from authored training cases.

Example: python -m biogym.collect --cases seq-01 struct-01 --episodes 4
Requires SARVAM_API_KEY and consumes Sarvam API credits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .cases import BY_ID
from .curation import curate_successes
from .llm import SarvamActionProvider, run_trajectory
from .store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["seq-01"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes > 1000:
        parser.error("episodes must be between 1 and 1000")
    for case_id in args.cases:
        if case_id not in BY_ID or BY_ID[case_id].split != "train":
            parser.error(f"{case_id} is not an authored training case")
    data_dir = Path(os.getenv("BIOGYM_DATA_DIR", Path(__file__).resolve().parents[1] / ".data"))
    store = Store(data_dir / "biogym.db")
    provider = SarvamActionProvider()
    results = [
        run_trajectory(store, provider, args.cases[index % len(args.cases)],
                       args.seed + index, data_dir / "trajectories")
        for index in range(args.episodes)
    ]
    corpus = curate_successes(data_dir / "trajectories", data_dir / "curated" / "sarvam-train.jsonl")
    print(json.dumps({"runs": results, "curated": corpus}, indent=2))


if __name__ == "__main__":
    main()
