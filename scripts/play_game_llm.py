"""Play one ARC-AGI-3 game with the LLM agent.

Backward-compatible positional usage (Path 5):
    python scripts/play_game_llm.py [game_id] [max_actions] [model]

Path 6 additions (optional, do not change default behavior):
    --profile {legacy,path6}   select a config profile (default: path6)
    --set key=value            override individual flags (repeatable)
"""

import argparse
import sys

sys.path.insert(0, "src")

from charith.config import config_from_cli
from charith.llm_agent.agent import LLMAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an ARC-AGI-3 game with the LLM agent.")
    parser.add_argument("game_id", nargs="?", default="ls20")
    parser.add_argument("max_actions", nargs="?", type=int, default=50)
    parser.add_argument("model", nargs="?", default="gemma4:4b")
    parser.add_argument("--profile", choices=["legacy", "path6"], default="path6")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="key=value", help="Override a config flag (repeatable).")
    args = parser.parse_args()

    config = config_from_cli(args.profile, args.overrides)

    print(f"Game: {args.game_id}, Max actions: {args.max_actions}, "
          f"Model: {args.model}, Profile: {args.profile}")
    agent = LLMAgent(model=args.model, config=config)
    result = agent.play_game(args.game_id, max_actions=args.max_actions)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
