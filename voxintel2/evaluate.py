"""
evaluate.py — measures VoxIntel accuracy across all signals
Run: python evaluate.py
"""

import json
import time
from agent import analyze
import os
import os, re, json, logging
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
# ── Load test data ────────────────────────────────────────────────────────────
with open("test_emails.json") as f:
    test_cases = json.load(f)

FIELDS = ["intent", "emotion", "sentiment", "priority", "department"]

results     = []
correct     = {f: 0 for f in FIELDS}
total       = len(test_cases)
latencies   = []
confidences = []

print(f"\nRunning evaluation on {total} test emails...\n")
print("=" * 70)

for i, tc in enumerate(test_cases):
    start  = time.perf_counter()
    output = analyze(subject="", body=tc["email"])
    elapsed = round((time.perf_counter() - start) * 1000, 1)
    latencies.append(elapsed)

    row = {"email": tc["email"][:60] + "...", "latency_ms": elapsed}

    all_correct = True
    for field in FIELDS:
        expected = tc[f"expected_{field}"].lower()
        got      = str(output.get(field, "")).lower()
        match    = expected == got
        if match:
            correct[field] += 1
        else:
            all_correct = False
        row[field] = {"expected": expected, "got": got, "match": match}

    row["all_correct"] = all_correct
    results.append(row)
    confidences.append(output.get("confidence", 0.0))

    # Print per-email result
    status = "✅" if all_correct else "❌"
    print(f"{status} Email {i+1}: {tc['email'][:55]}...")
    for field in FIELDS:
        r = row[field]
        icon = "  ✓" if r["match"] else "  ✗"
        print(f"    {icon} {field:12} expected={r['expected']:20} got={r['got']}")
    print(f"    ⏱  {elapsed}ms\n")

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("\n📊 EVALUATION SUMMARY\n")

total_correct = sum(correct.values())
total_fields  = total * len(FIELDS)
overall_acc   = round(total_correct / total_fields * 100, 1)

print(f"{'Field':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print("-" * 50)
for field in FIELDS:
    acc = round(correct[field] / total * 100, 1)
    print(f"{field:<20} {correct[field]:>8} {total:>8} {acc:>9}%")

print("-" * 50)
print(f"{'OVERALL':<20} {total_correct:>8} {total_fields:>8} {overall_acc:>9}%")

# ── Precision / Recall per intent ─────────────────────────────────────────────
print("\n📈 INTENT PRECISION & RECALL\n")

intents = list({tc["expected_intent"] for tc in test_cases})
print(f"{'Intent':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8}")
print("-" * 60)

for intent in sorted(intents):
    tp = sum(1 for r in results
             if r["intent"]["expected"] == intent and r["intent"]["got"] == intent)
    fp = sum(1 for r in results
             if r["intent"]["got"] == intent and r["intent"]["expected"] != intent)
    fn = sum(1 for r in results
             if r["intent"]["expected"] == intent and r["intent"]["got"] != intent)

    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0

    print(f"{intent:<25} {tp:>4} {fp:>4} {fn:>4} {precision:>9}% {recall:>7}%")

# ── Latency stats ─────────────────────────────────────────────────────────────
print(f"\n⏱  LATENCY\n")
print(f"  Average : {round(sum(latencies)/len(latencies), 1)} ms")
print(f"  Fastest : {min(latencies)} ms")
print(f"  Slowest : {max(latencies)} ms")

# ── Confidence stats ──────────────────────────────────────────────────────────
print(f"\n🎯 CONFIDENCE\n")
print(f"  Average : {round(sum(confidences)/len(confidences)*100, 1)}%")
print(f"  Lowest  : {round(min(confidences)*100, 1)}%")
print(f"  Highest : {round(max(confidences)*100, 1)}%")

# ── Save full results ──────────────────────────────────────────────────────────
with open("evaluation_results.json", "w") as f:
    json.dump({
        "overall_accuracy": overall_acc,
        "per_field":        {f: round(correct[f]/total*100, 1) for f in FIELDS},
        "avg_latency_ms":   round(sum(latencies)/len(latencies), 1),
        "avg_confidence":   round(sum(confidences)/len(confidences)*100, 1),
        "results":          results,
    }, f, indent=2)

print(f"\n✅ Full results saved to evaluation_results.json\n")