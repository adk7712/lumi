import time
import json
import statistics
from pathlib import Path
import persistence

def benchmark_db():
    print("================================================================================")
    print("🔬 COMPREHENSIVE DATABASE OPERATIONS LATENCY BENCHMARK")
    print("================================================================================")

    # 1. Connection acquisition
    conn_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        conn = persistence.get_db_connection()
        conn_times.append((time.perf_counter() - t0) * 1000)
        persistence.release_db_connection(conn)

    print(f"1. get_db_connection (50 iterations):")
    print(f"   Avg: {statistics.mean(conn_times):.3f}ms | p50: {statistics.median(conn_times):.3f}ms | Min: {min(conn_times):.3f}ms | Max: {max(conn_times):.3f}ms")

    # 2. save_session benchmark (varying recipe sizes)
    test_cases = [
        ("Small (1 step, 1 rule)", [{"action": "strip_whitespace", "column": "All"}], [{"type": "Null Check", "column": "Id"}]),
        ("Medium (10 steps, 10 rules)", [{"action": f"step_{i}", "col": f"c_{i}"} for i in range(10)], [{"type": "Range Check", "col": f"c_{i}"} for i in range(10)]),
        ("Large (50 steps, 30 rules)", [{"action": f"step_{i}", "col": f"c_{i}"} for i in range(50)], [{"type": "Range Check", "col": f"c_{i}"} for i in range(30)])
    ]

    for label, recipe, rules in test_cases:
        durations = []
        for i in range(20):
            t0 = time.perf_counter()
            persistence.save_session(
                session_id=f"bench_sess_{i}",
                filename="bench_train.csv",
                recipe=recipe,
                rules=rules,
                scanned_columns=[f"col_{j}" for j in range(81)],
                user_id="bench_user@lumi.ai",
                project_name=f"Benchmark Project {i}"
            )
            durations.append((time.perf_counter() - t0) * 1000)
        print(f"2. save_session - {label}:")
        print(f"   Avg: {statistics.mean(durations):.3f}ms | p50: {statistics.median(durations):.3f}ms | Min: {min(durations):.3f}ms | Max: {max(durations):.3f}ms")

    # 3. load_session benchmark
    load_durations = []
    for i in range(20):
        t0 = time.perf_counter()
        res = persistence.load_session(f"bench_sess_{i}", "bench_user@lumi.ai")
        load_durations.append((time.perf_counter() - t0) * 1000)
    print(f"3. load_session (20 iterations):")
    print(f"   Avg: {statistics.mean(load_durations):.3f}ms | p50: {statistics.median(load_durations):.3f}ms | Min: {min(load_durations):.3f}ms | Max: {max(load_durations):.3f}ms")

    # 4. get_user_projects benchmark (simulating sidebar load on every rerun)
    proj_durations = []
    for _ in range(30):
        t0 = time.perf_counter()
        projs = persistence.get_user_projects("bench_user@lumi.ai")
        proj_durations.append((time.perf_counter() - t0) * 1000)
    print(f"4. get_user_projects (30 iterations, 20 projects in DB):")
    print(f"   Avg: {statistics.mean(proj_durations):.3f}ms | p50: {statistics.median(proj_durations):.3f}ms | Min: {min(proj_durations):.3f}ms | Max: {max(proj_durations):.3f}ms")

    # 5. Mutations: rename, toggle_pin, delete
    rename_durations = []
    pin_durations = []
    delete_durations = []
    for i in range(20):
        t0 = time.perf_counter()
        persistence.rename_session(f"bench_sess_{i}", f"Renamed Project {i}", "bench_user@lumi.ai")
        rename_durations.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        persistence.toggle_pin_session(f"bench_sess_{i}", "bench_user@lumi.ai")
        pin_durations.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        persistence.delete_session(f"bench_sess_{i}", "bench_user@lumi.ai")
        delete_durations.append((time.perf_counter() - t0) * 1000)

    print(f"5. rename_session:")
    print(f"   Avg: {statistics.mean(rename_durations):.3f}ms | p50: {statistics.median(rename_durations):.3f}ms | Max: {max(rename_durations):.3f}ms")
    print(f"6. toggle_pin_session:")
    print(f"   Avg: {statistics.mean(pin_durations):.3f}ms | p50: {statistics.median(pin_durations):.3f}ms | Max: {max(pin_durations):.3f}ms")
    print(f"7. delete_session:")
    print(f"   Avg: {statistics.mean(delete_durations):.3f}ms | p50: {statistics.median(delete_durations):.3f}ms | Max: {max(delete_durations):.3f}ms")

    print("================================================================================\n")

if __name__ == "__main__":
    benchmark_db()
