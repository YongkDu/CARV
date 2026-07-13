#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute accuracy for each JSON/JSONL file under a folder, "
            "using only non-null correctness values from correctness_analysis.correctness."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="output_simple_new/mix/difference",
        help="Folder containing files to evaluate (default: output_simple_new/mix/difference)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include files in subfolders.",
    )
    return parser.parse_args()


def extract_correctness(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none", ""}:
            return None
    return None


def iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError:
                    continue
    elif path.suffix.lower() == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            elif isinstance(data, dict):
                yield data
        except (json.JSONDecodeError, OSError):
            return


def evaluate_file(path: Path):
    total = 0
    valid = 0
    true_count = 0
    false_count = 0
    null_count = 0

    for record in iter_records(path):
        total += 1
        correctness = None

        if isinstance(record, dict):
            analysis = record.get("correctness_analysis")
            if isinstance(analysis, dict):
                correctness = extract_correctness(analysis.get("correctness"))

            if correctness is None and "correctness" in record:
                correctness = extract_correctness(record.get("correctness"))

        if correctness is None:
            null_count += 1
            continue

        valid += 1
        if correctness:
            true_count += 1
        else:
            false_count += 1

    accuracy = (true_count / valid) if valid else 0.0
    return {
        "file": path,
        "total": total,
        "valid": valid,
        "true": true_count,
        "false": false_count,
        "null": null_count,
        "accuracy": accuracy,
    }


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found or not a directory: {folder}")

    pattern = "**/*" if args.recursive else "*"
    files = sorted(
        p
        for p in folder.glob(pattern)
        if p.is_file() and p.suffix.lower() in {".jsonl", ".json"}
    )

    if not files:
        raise SystemExit(f"No .jsonl or .json files found in: {folder}")

    print("file\taccuracy\tvalid\ttrue\tfalse\tnull\ttotal")

    overall_true = 0
    overall_valid = 0
    overall_false = 0
    overall_null = 0
    overall_total = 0

    for file_path in files:
        stats = evaluate_file(file_path)
        rel_name = file_path.relative_to(folder)
        print(
            f"{rel_name}\t{stats['accuracy']:.4f}\t{stats['valid']}\t"
            f"{stats['true']}\t{stats['false']}\t{stats['null']}\t{stats['total']}"
        )

        overall_true += stats["true"]
        overall_valid += stats["valid"]
        overall_false += stats["false"]
        overall_null += stats["null"]
        overall_total += stats["total"]

    overall_accuracy = (overall_true / overall_valid) if overall_valid else 0.0
    print(
        f"OVERALL\t{overall_accuracy:.4f}\t{overall_valid}\t{overall_true}\t"
        f"{overall_false}\t{overall_null}\t{overall_total}"
    )


if __name__ == "__main__":
    main()
