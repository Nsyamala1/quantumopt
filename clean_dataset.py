#!/usr/bin/env python3
"""
clean_dataset.py — Remove outlier circuits with original_depth > 10,000.

Reads dataset.json, filters out abnormally large circuits (grover, qwalk, etc.),
and saves the cleaned version to dataset_clean.json.

Usage:
    python clean_dataset.py
"""

import json
from collections import Counter

DEPTH_THRESHOLD = 10_000
INPUT_FILE = "dataset.json"
OUTPUT_FILE = "dataset_clean.json"

with open(INPUT_FILE) as f:
    dataset = json.load(f)

print(f"📂 Loaded {len(dataset)} circuits from {INPUT_FILE}")

# Partition into keep vs remove
removed = [r for r in dataset if r["original_depth"] > DEPTH_THRESHOLD]
cleaned = [r for r in dataset if r["original_depth"] <= DEPTH_THRESHOLD]

print(f"\n🗑  Removed {len(removed)} circuits with original_depth > {DEPTH_THRESHOLD:,}:")
for algo, count in sorted(Counter(r["algorithm"] for r in removed).items(), key=lambda x: -x[1]):
    print(f"   • {algo}: {count}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"\n✅ Saved {len(cleaned)} clean circuits to {OUTPUT_FILE}")
