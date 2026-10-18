import json
from pathlib import Path
from collections import defaultdict
import statistics

PERF_LOG = Path(__file__).parent.parent / ".lumi_cache" / "db_perf_log.jsonl"

def analyze_db_log():
    if not PERF_LOG.exists():
        print(f"No performance log found at {PERF_LOG}")
        return

    records = []
    with open(PERF_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        print("Performance log is empty.")
        return

    print("================================================================================")
    print(f"📊 DATABASE PERFORMANCE PROFILE REPORT ({len(records)} operations logged)")
    print("================================================================================")

    # Group by operation
    by_op = defaultdict(list)
    by_thread = defaultdict(list)
    errors = []

    for r in records:
        op = r.get("op", "unknown")
        dur = r.get("duration_ms", 0.0)
        by_op[op].append(dur)
        thread = "MainThread (UI Blocking)" if r.get("is_main_thread") else "ThreadPool (Async Background)"
        by_thread[thread].append(dur)
        if r.get("error"):
            errors.append(r)

    print(f"\n{'Operation':<22} | {'Calls':<6} | {'Avg (ms)':<9} | {'Min (ms)':<9} | {'Max (ms)':<9} | {'p50 (ms)':<9} | {'p95 (ms)':<9} | {'Total (s)':<9}")
    print("-" * 92)

    for op, durations in sorted(by_op.items(), key=lambda x: sum(x[1]), reverse=True):
        count = len(durations)
        total_s = sum(durations) / 1000.0
        avg_ms = statistics.mean(durations)
        min_ms = min(durations)
        max_ms = max(durations)
        p50 = statistics.median(durations)
        p95 = statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max_ms
        print(f"{op:<22} | {count:<6} | {avg_ms:<9.2f} | {min_ms:<9.2f} | {max_ms:<9.2f} | {p50:<9.2f} | {p95:<9.2f} | {total_s:<9.3f}")

    print("\n--------------------------------------------------------------------------------")
    print("🧵 THREAD EXECUTION BREAKDOWN (Blocking vs. Non-Blocking):")
    print("--------------------------------------------------------------------------------")
    for t_type, durations in by_thread.items():
        pct = (len(durations) / len(records)) * 100
        tot_time_s = sum(durations) / 1000.0
        avg_ms = statistics.mean(durations)
        print(f"  • {t_type:<32}: {len(durations):<4} calls ({pct:5.1f}%) | Total time: {tot_time_s:6.3f}s | Avg: {avg_ms:6.2f}ms")

    if errors:
        print("\n--------------------------------------------------------------------------------")
        print(f"⚠️  DATABASE ERRORS / FALLBACKS ENCOUNTERED ({len(errors)} events):")
        print("--------------------------------------------------------------------------------")
        for e in errors[:5]:
            print(f"  [{e.get('op')}] Error: {e.get('error')[:100]}... (Backend: {e.get('details', {}).get('backend')})")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more error instances.")
    else:
        print("\n✅ Zero database errors encountered.")

    print("================================================================================\n")

if __name__ == "__main__":
    analyze_db_log()
