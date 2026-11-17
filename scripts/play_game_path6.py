"""Path 6 entrypoint — play one ARC-AGI-3 game with the falsifiable agent.

This is the new entrypoint for Path 6 work (guide ground rule 4). The legacy
``play_game_llm.py`` stays untouched in behavior.

Usage:
    python scripts/play_game_path6.py --game ls20 --max-actions 200
    python scripts/play_game_path6.py --game tr87 --profile legacy
    python scripts/play_game_path6.py --game ls20 --set gated_expansion=false
"""

import argparse
import sys

sys.path.insert(0, "src")

from charith.config import config_from_cli
from charith.llm_agent.agent import LLMAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Path 6 falsifiable agent runner.")
    parser.add_argument("--game", default="ls20")
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--profile", choices=["legacy", "path6"], default="path6")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="key=value", help="Override a config flag (repeatable).")
    parser.add_argument("--record", default=None,
                        help="Path to write an episode fixture (guide §2.6).")
    args = parser.parse_args()

    config = config_from_cli(args.profile, args.overrides)

    print(f"[Path6] Game: {args.game}, Max: {args.max_actions}, "
          f"Model: {args.model}, Profile: {args.profile}")
    agent = LLMAgent(model=args.model, config=config)
    result = agent.play_game(args.game, max_actions=args.max_actions,
                            record_path=args.record)
    print(f"\nResult: {result}")

    # S3.5 Stage-A acceptance diagnostics.
    led = agent.ledger
    resolved = [r for r in led.records if r.outcome in ("CONFIRMED", "REFUTED", "PARTIAL")]
    fallbacks = [r for r in agent.incidents.records if r["kind"] == "DECISION_FALLBACK"]
    saydo = [r for r in agent.incidents.records if "E-ACTION-MISMATCH" in str(r)]
    print("\n--- S3.5 Stage-A diagnostics ---")
    print(f"ticks={agent._tick} ledger_records={len(led.records)} "
          f"resolved(C/R/P)={len(resolved)} rolling_acc={led.rolling_accuracy():.2f}")
    print(f"decision_fallbacks={len(fallbacks)} saydo_incidents={len(saydo)}")
    print(f"tracked_oids_final={sorted(agent._last_tracked_oids)}")
    print(f"c1_expansions={[c.name for c in agent.c1c2.c1_expansions]}")
    print(f"incidents_total={len(agent.incidents.records)}")


if __name__ == "__main__":
    main()
