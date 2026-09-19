#!/usr/bin/env python3
"""Create a small, deterministic ShareGPT file with exactly N valid turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = "Aeala/ShareGPT_Vicuna_unfiltered"
DEFAULT_REVISION = "8b0048ad6ae8c22f46a78c15559dec98feef5539"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    return parser.parse_args()


def role_and_text(message: Any) -> tuple[str, str]:
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            return "", ""
    if not isinstance(message, dict):
        return "", ""
    role = str(message.get("from") or message.get("role") or "")
    text = str(message.get("value") or message.get("content") or "").strip()
    if role == "user":
        role = "human"
    elif role == "assistant":
        role = "gpt"
    return role, text


def valid_pairs(messages: list[Any]) -> list[dict[str, str]]:
    """Normalize a conversation to consecutive human/GPT pairs."""
    pairs: list[dict[str, str]] = []
    pending_user: str | None = None

    for message in messages:
        role, text = role_and_text(message)
        if not text:
            continue
        if role == "human":
            pending_user = text
        elif role == "gpt" and pending_user is not None:
            pairs.extend(
                (
                    {"from": "human", "value": pending_user},
                    {"from": "gpt", "value": text},
                )
            )
            pending_user = None

    return pairs


def main() -> None:
    args = parse_args()
    if args.conversations < 1 or args.turns < 2:
        raise SystemExit("--conversations must be >= 1 and --turns must be >= 2")

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        raise SystemExit(
            "the 'datasets' package is required; run this with .venv/bin/python"
        ) from error

    dataset = load_dataset(
        args.source,
        split="train",
        revision=args.revision,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=4096)

    selected: list[dict[str, Any]] = []
    required_messages = args.turns * 2
    for row in dataset:
        messages = row.get("conversations")
        if not isinstance(messages, list):
            continue
        normalized = valid_pairs(messages)
        if len(normalized) < required_messages:
            continue
        selected.append(
            {
                "id": str(row.get("id") or f"conversation-{len(selected)}"),
                "conversations": normalized[:required_messages],
            }
        )
        if len(selected) == args.conversations:
            break

    if len(selected) != args.conversations:
        raise SystemExit(
            f"found only {len(selected)} conversations with {args.turns} valid turns"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(selected, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(
        f"wrote {len(selected)} conversations x {args.turns} turns to {args.output}"
    )


if __name__ == "__main__":
    main()
