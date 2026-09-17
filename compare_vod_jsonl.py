"""Compare two validate_vod JSONL logs (EasyOCR vs Paddle)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _players(payload: dict) -> list[dict]:
    return payload.get("players") or []


def _filled(val) -> bool:
    return val not in (None, "", " ")


def _in_game_stats(rows: list[dict]) -> dict:
    in_game = [r for r in rows if (r.get("payload") or {}).get("state") == "in_game"]
    n = len(in_game) or 1
    stage = sum(1 for r in in_game if _filled((r.get("payload") or {}).get("stage")))
    chars = 0
    names = 0
    stocks = 0
    damage = 0
    for r in in_game:
        for p in _players(r.get("payload") or {}):
            chars += int(_filled(p.get("character")))
            names += int(_filled(p.get("name")))
            stocks += int(p.get("stocks") is not None)
            damage += int(_filled(p.get("damage")))
    slots = max(1, len(in_game) * 2)
    states = [(r.get("payload") or {}).get("state") for r in rows]
    return {
        "events": len(rows),
        "in_game_events": len(in_game),
        "versus_both_chars": any(
            (r.get("payload") or {}).get("state") in ("in_game", "character_select", None)
            and all(_filled(p.get("character")) for p in _players(r.get("payload") or {}))
            and len(_players(r.get("payload") or {})) >= 2
            for r in rows
        ),
        "game_end_events": sum(1 for s in states if s == "game_end"),
        "fill": {
            "stage": round(stage / n, 3),
            "character": round(chars / slots, 3),
            "name": round(names / slots, 3),
            "stocks": round(stocks / slots, 3),
            "damage": round(damage / slots, 3),
        },
    }


def _stock_changes(rows: list[dict]) -> list[list]:
    out = []
    prev = None
    for r in rows:
        payload = r.get("payload") or {}
        if payload.get("state") != "in_game":
            continue
        pair = [p.get("stocks") for p in _players(payload)]
        if pair != prev:
            out.append({"t": r.get("t"), "stocks": pair})
            prev = pair
    return out


def _flip_flops(rows: list[dict], field: str) -> int:
    last = {}
    flips = 0
    for r in rows:
        payload = r.get("payload") or {}
        if payload.get("state") != "in_game":
            continue
        if field == "stage":
            cur = payload.get("stage")
            if last.get("stage") and cur and cur != last["stage"]:
                flips += 1
            if cur:
                last["stage"] = cur
            continue
        for i, p in enumerate(_players(payload)):
            key = (field, i)
            cur = p.get(field)
            if last.get(key) and cur and cur != last[key]:
                flips += 1
            if cur:
                last[key] = cur
    return flips


def _by_t(rows: list[dict]) -> dict[float, dict]:
    return {float(r["t"]): r.get("payload") or {} for r in rows if "t" in r}


def _agree(a: dict, b: dict, key_path) -> bool | None:
    va, vb = a, b
    for k in key_path:
        if isinstance(va, list) and isinstance(k, int):
            if k >= len(va) or k >= len(vb):
                return None
            va, vb = va[k], vb[k]
        else:
            va = (va or {}).get(k) if isinstance(va, dict) else None
            vb = (vb or {}).get(k) if isinstance(vb, dict) else None
    if not _filled(va) or not _filled(vb):
        return None
    return va == vb


def compare(easy_path: str, paddle_path: str) -> dict:
    easy = _load(easy_path)
    paddle = _load(paddle_path)
    easy_t = _by_t(easy)
    paddle_t = _by_t(paddle)
    shared = sorted(set(easy_t) & set(paddle_t))

    def _rate(path) -> dict:
        hits = [ _agree(easy_t[t], paddle_t[t], path) for t in shared ]
        known = [x for x in hits if x is not None]
        return {
            "compared": len(known),
            "agree": round(sum(known) / len(known), 3) if known else None,
        }

    report = {
        "easyocr": _in_game_stats(easy),
        "paddle": _in_game_stats(paddle),
        "easyocr_stock_changes": _stock_changes(easy),
        "paddle_stock_changes": _stock_changes(paddle),
        "flip_flops": {
            "easyocr": {
                "character": _flip_flops(easy, "character"),
                "stage": _flip_flops(easy, "stage"),
            },
            "paddle": {
                "character": _flip_flops(paddle, "character"),
                "stage": _flip_flops(paddle, "stage"),
            },
        },
        "agreement_on_shared_t": {
            "stage": _rate(["stage"]),
            "p0_character": _rate(["players", 0, "character"]),
            "p1_character": _rate(["players", 1, "character"]),
            "p0_stocks": _rate(["players", 0, "stocks"]),
            "p1_stocks": _rate(["players", 1, "stocks"]),
        },
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two VOD OCR JSONL logs.")
    p.add_argument("easyocr_jsonl")
    p.add_argument("paddle_jsonl")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    report = compare(args.easyocr_jsonl, args.paddle_jsonl)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
