"""Run an autonomous bot-vs-bot simulation. Dumps every action to JSONL.

Usage:
    uv run python scripts/simulate.py [num_hands] [output_file]

Default: 1000 hands, 6-handed table with mixed archetypes, dump to ./hands.jsonl
"""

import sys
import time
from pathlib import Path

from poker.agents import make_lag, make_maniac, make_nit, make_station, make_tag
from poker.simulator import run_simulation


def main() -> None:
    num_hands = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("hands.jsonl")

    # 6-handed mixed table
    agents = [
        make_nit("Nit_1"),
        make_tag("TAG_1"),
        make_tag("TAG_2"),
        make_lag("LAG_1"),
        make_maniac("Maniac_1"),
        make_station("Station_1"),
    ]
    print(f"Simulating {num_hands:,} hands at a 6-handed table...")
    print(f"Players: {', '.join(a.name for a in agents)}")
    print(f"Output: {output_path}\n")

    t0 = time.perf_counter()
    stats = run_simulation(
        agents=agents, num_hands=num_hands, starting_stack=100.0,
        blinds=(0.5, 1.0), persistent_stacks=False,
        output_path=output_path, seed=42,
        progress_every=max(1, num_hands // 10),
    )
    elapsed = time.perf_counter() - t0
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed * 1000 / num_hands:.0f}ms/hand)")
    print()
    print(stats)


if __name__ == "__main__":
    main()
