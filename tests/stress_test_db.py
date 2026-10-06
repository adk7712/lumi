import time
import json
import statistics
import concurrent.futures
import threading
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import persistence

def run_stress_test():
    print("================================================================================")
    print("🚀 LUMI DATABASE STRESS TEST & LATENCY BENCHMARK")
    print("================================================================================")
    
    # Check connection backend
    conn = persistence.get_db_connection()
    backend = "PostgreSQL" if "psycopg2" in type(conn).__module__ else "SQLite"
    persistence.release_db_connection(conn)
    print(f"Target Backend: {backend}")
    print(f"PostgreSQL Pool Config: min=1, max=10")
    print("================================================================================\n")

    # Helper for percentiles
    def stats(latencies):
        if not latencies:
            return "N/A"
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted)
        p90 = latencies_sorted[int(len(latencies_sorted) * 0.90)]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[min(int(len(latencies_sorted) * 0.99), len(latencies_sorted) - 1)]
        avg = statistics.mean(latencies_sorted)
        return f"Avg: {avg:.2f}ms | p50: {p50:.2f}ms | p90: {p90:.2f}ms | p95: {p95:.2f}ms | p99: {p99:.2f}ms | Min: {min(latencies_sorted):.2f}ms | Max: {max(latencies_sorted):.2f}ms"

    # -------------------------------------------------------------------------
    # TEST 1: Concurrency Scaling on `save_session`
    # -------------------------------------------------------------------------
    concurrency_levels = [1, 5, 10, 15, 25, 50]
    sample_recipe = [{"action": "impute_missing", "column": f"col_{i}", "strategy": "mean"} for i in range(15)]
    sample_rules = [{"type": "Range Check", "col": f"col_{i}", "min": 0, "max": 100} for i in range(10)]
    scanned_cols = [f"col_{i}" for i in range(50)]

    print("--- TEST 1: Concurrency Scaling on `save_session` ---")
    for workers in concurrency_levels:
        total_ops = 50
        errors = 0
        latencies = []
        fallbacks = 0

        def worker_task(idx):
            nonlocal errors, fallbacks
            sess_id = f"stress_sess_c{workers}_{idx}"
            t0 = time.perf_counter()
            try:
                persistence.save_session(
                    session_id=sess_id,
                    filename=f"stress_{idx}.csv",
                    recipe=sample_recipe,
                    rules=sample_rules,
                    scanned_columns=scanned_cols,
                    user_id=f"stress_user_{idx % 5}@lumi.ai",
                    project_name=f"Stress Project {idx}"
                )
                dur = (time.perf_counter() - t0) * 1000
                return dur
            except Exception as e:
                errors += 1
                return None

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(total_ops)]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res is not None:
                    latencies.append(res)
        total_wall_time = time.perf_counter() - t_start
        throughput = total_ops / total_wall_time

        print(f"Concurrency: {workers:2d} threads | Total Ops: {total_ops} | Wall Time: {total_wall_time:.2f}s | Throughput: {throughput:.1f} ops/s | Errors: {errors}")
        print(f"   Latency: {stats(latencies)}")
        print()

    # -------------------------------------------------------------------------
    # TEST 2: Concurrency Scaling on `load_session`
    # -------------------------------------------------------------------------
    print("--- TEST 2: Concurrency Scaling on `load_session` ---")
    for workers in concurrency_levels:
        total_ops = 50
        errors = 0
        latencies = []

        def load_task(idx):
            nonlocal errors
            sess_id = f"stress_sess_c1_{idx % 10}" # Load from previously saved sessions
            t0 = time.perf_counter()
            try:
                res = persistence.load_session(sess_id, f"stress_user_{idx % 5}@lumi.ai")
                dur = (time.perf_counter() - t0) * 1000
                if res is None:
                    errors += 1
                return dur
            except Exception as e:
                errors += 1
                return None

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(load_task, i) for i in range(total_ops)]
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res is not None:
                    latencies.append(res)
        total_wall_time = time.perf_counter() - t_start
        throughput = total_ops / total_wall_time

        print(f"Concurrency: {workers:2d} threads | Total Ops: {total_ops} | Wall Time: {total_wall_time:.2f}s | Throughput: {throughput:.1f} ops/s | Errors: {errors}")
        print(f"   Latency: {stats(latencies)}")
        print()

    # -------------------------------------------------------------------------
    # TEST 3: High-Stress Mixed Workload (70% Read, 20% Write, 10% Mutations)
    # -------------------------------------------------------------------------
    print("--- TEST 3: High-Stress Mixed Workload (50 concurrent threads, 100 mixed ops) ---")
    import random
    mixed_latencies = {"read": [], "write": [], "mutation": []}
    mixed_errors = {"read": 0, "write": 0, "mutation": 0}

    def mixed_op(idx):
        roll = random.random()
        user = f"stress_user_{idx % 5}@lumi.ai"
        sess_id = f"stress_sess_c1_{idx % 10}"
        t0 = time.perf_counter()
        if roll < 0.70: # Read
            try:
                res = persistence.load_session(sess_id, user)
                dur = (time.perf_counter() - t0) * 1000
                mixed_latencies["read"].append(dur)
            except Exception:
                mixed_errors["read"] += 1
        elif roll < 0.90: # Write
            try:
                persistence.save_session(
                    session_id=f"mixed_sess_{idx}",
                    filename="mixed.csv",
                    recipe=sample_recipe,
                    rules=sample_rules,
                    scanned_columns=scanned_cols,
                    user_id=user,
                    project_name=f"Mixed {idx}"
                )
                dur = (time.perf_counter() - t0) * 1000
                mixed_latencies["write"].append(dur)
            except Exception:
                mixed_errors["write"] += 1
        else: # Mutation (Rename or Pin or Delete)
            try:
                persistence.toggle_pin_session(sess_id, user)
                dur = (time.perf_counter() - t0) * 1000
                mixed_latencies["mutation"].append(dur)
            except Exception:
                mixed_errors["mutation"] += 1

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(mixed_op, i) for i in range(100)]
        concurrent.futures.wait(futures)
    total_wall_time = time.perf_counter() - t_start

    print(f"Wall Time: {total_wall_time:.2f}s | Mixed Throughput: {100 / total_wall_time:.1f} ops/s")
    print(f"  Reads:     {stats(mixed_latencies['read'])} (Errors: {mixed_errors['read']})")
    print(f"  Writes:    {stats(mixed_latencies['write'])} (Errors: {mixed_errors['write']})")
    print(f"  Mutations: {stats(mixed_latencies['mutation'])} (Errors: {mixed_errors['mutation']})")
    print()

    # -------------------------------------------------------------------------
    # TEST 4: Cleanup stress test sessions
    # -------------------------------------------------------------------------
    print("--- TEST 4: Cleanup & Teardown ---")
    conn = persistence.get_db_connection()
    try:
        cur = conn.cursor()
        persistence._execute(conn, cur, "DELETE FROM sessions WHERE session_id LIKE 'stress_%' OR session_id LIKE 'mixed_%'")
        conn.commit()
        print("Cleaned up temporary stress test sessions.")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        persistence.release_db_connection(conn)

    print("\n================================================================================")
    print("✅ STRESS TEST COMPLETED")
    print("================================================================================")

if __name__ == "__main__":
    run_stress_test()
