#!/usr/bin/env python3
"""View posture history analytics from posture_history.csv."""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="View posture history analytics")
    parser.add_argument(
        "file",
        nargs="?",
        default="posture_history.csv",
        help="Path to posture history CSV (default: posture_history.csv)",
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Show stats for today only",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        print("Run live_prediction_model.py with --log to create posture_history.csv")
        return

    rows: list[tuple[datetime, str, float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                posture = row["posture"]
                confidence = float(row.get("confidence", 0))
                rows.append((ts, posture, confidence))
            except (KeyError, ValueError):
                continue

    if not rows:
        print("No data in file.")
        return

    if args.today:
        today = rows[0][0].date()
        rows = [(ts, p, c) for ts, p, c in rows if ts.date() == today]
        if not rows:
            print("No data for today.")
            return
        print(f"Today ({today}) stats:\n")
    else:
        print("All-time stats:\n")

    total = len(rows)
    by_posture: dict[str, int] = defaultdict(int)
    for _, posture, _ in rows:
        by_posture[posture] += 1

    print(f"Total logged samples: {total}")
    print("Distribution:")
    for posture in ("upright", "slouched", "leaning"):
        count = by_posture.get(posture, 0)
        pct = 100 * count / total if total else 0
        print(f"  {posture}: {count} ({pct:.0f}%)")

    # Estimate time (assuming ~2s between samples)
    est_seconds = total * 2
    est_mins = est_seconds // 60
    print(f"\nEstimated session duration: ~{est_mins} min")


if __name__ == "__main__":
    main()
