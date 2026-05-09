"""Run the WeekendFlow A-stage demo flow from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.graph import get_graph


DEFAULT_INPUT = (
    "今天下午想和老婆孩子出去玩几个小时，别离家太远，孩子5岁，老婆最近在减肥。"
)


def main() -> None:
    user_input = " ".join(sys.argv[1:]).strip() or DEFAULT_INPUT
    app = get_graph()
    result = app.invoke({"user_input": user_input, "user_id": "u001"})
    print(
        json.dumps(
            {
                "intent": result.get("intent"),
                "memory": result.get("memory"),
                "constraints": result.get("constraints"),
                "short_term_memory": result.get("short_term_memory"),
                "execution_log": result.get("execution_log"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
