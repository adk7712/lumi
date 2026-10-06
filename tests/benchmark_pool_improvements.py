import time
import json
import statistics
import concurrent.futures
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import persistence
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import PoolError

class BoundedPgPool:
    """Thread-safe PostgreSQL connection pool with semaphore queueing and autocommit."""
    def __init__(self, db_url, minconn=2, maxconn=20, timeout=10.0):
        self.db_url = db_url
        self.maxconn = maxconn
        self.timeout = timeout
        self._lock = threading.Lock()
        self._pool = []
        self._used = set()
        self._semaphore = threading.Semaphore(maxconn)
        
        # Pre-populate minconn connections
        for _ in range(minconn):
            conn = self._create_conn()
            if conn:
                self._pool.append(conn)

    def _create_conn(self):
        try:
            conn = psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)
            conn.autocommit = True
            return conn
        except Exception as e:
            print(f"Error creating PG connection: {e}")
            return None

    def getconn(self, timeout=None):
        timeout = timeout or self.timeout
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            raise PoolError(f"Connection pool timeout after {timeout}s: all {self.maxconn} connections busy")
        
        with self._lock:
            # Check for alive connection in pool
            while self._pool:
                conn = self._pool.pop()
                if getattr(conn, "closed", 0) == 0:
                    try:
                        # Quick liveness check
                        conn.poll()
                        self._used.add(conn)
                        return conn
                    except Exception:
                        try: conn.close()
                        except Exception: pass
            
            # Create new connection
            conn = self._create_conn()
            if not conn:
                self._semaphore.release()
                raise PoolError("Failed to create new connection to PostgreSQL")
            self._used.add(conn)
            return conn

    def putconn(self, conn, close=False):
        with self._lock:
            if conn in self._used:
                self._used.remove(conn)
            
            if close or getattr(conn, "closed", 0) != 0:
                try: conn.close()
                except Exception: pass
            else:
                try:
                    # Clean any open transaction
                    if not conn.autocommit and conn.get_transaction_status() != 0:
                        conn.rollback()
                    self._pool.append(conn)
                except Exception:
                    try: conn.close()
                    except Exception: pass
        self._semaphore.release()

def test_comparison():
    db_url = persistence._get_db_url()
    if not db_url:
        print("No DB_URL found. Skipping.")
        return

    print("================================================================================")
    print("🔬 BENCHMARK: ORIGINAL POOL vs BOUNDED OPTIMIZED POOL")
    print("================================================================================")

    # 1. Test original pool under 15 threads (30 total ops)
    orig_pool = persistence._get_pg_pool(db_url)
    orig_latencies = []
    orig_errors = 0
    orig_fallbacks = 0

    def run_orig(idx):
        nonlocal orig_errors, orig_fallbacks
        t0 = time.perf_counter()
        try:
            conn = orig_pool.getconn()
            cur = conn.cursor()
            cur.execute("SELECT 1 as val")
            res = cur.fetchone()
            orig_pool.putconn(conn)
            return (time.perf_counter() - t0) * 1000
        except PoolError:
            orig_fallbacks += 1
            return None
        except Exception:
            orig_errors += 1
            return None

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futs = [ex.submit(run_orig, i) for i in range(30)]
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r is not None:
                orig_latencies.append(r)
    orig_wall = time.perf_counter() - t_start

    print("ORIGINAL POOL (15 concurrent threads, 30 requests):")
    print(f"  Wall Time: {orig_wall:.2f}s | Throughput: {len(orig_latencies)/orig_wall:.1f} ops/s")
    print(f"  Pool Exhaustion Fallbacks: {orig_fallbacks} / 30")
    if orig_latencies:
        print(f"  Latency: Avg: {statistics.mean(orig_latencies):.1f}ms | p50: {statistics.median(orig_latencies):.1f}ms | Max: {max(orig_latencies):.1f}ms")
    print()

    # 2. Test bounded pool under 15, 25, and 50 threads
    new_pool = BoundedPgPool(db_url, minconn=2, maxconn=20, timeout=10.0)
    for workers in [15, 25, 50]:
        new_latencies = []
        new_errors = 0
        new_timeouts = 0

        def run_new(idx):
            nonlocal new_errors, new_timeouts
            t0 = time.perf_counter()
            try:
                conn = new_pool.getconn(timeout=10.0)
                cur = conn.cursor()
                cur.execute("SELECT 1 as val")
                res = cur.fetchone()
                new_pool.putconn(conn)
                return (time.perf_counter() - t0) * 1000
            except PoolError:
                new_timeouts += 1
                return None
            except Exception:
                new_errors += 1
                return None

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_new, i) for i in range(50)]
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                if r is not None:
                    new_latencies.append(r)
        new_wall = time.perf_counter() - t_start

        print(f"BOUNDED POOL ({workers} concurrent threads, 50 requests):")
        print(f"  Wall Time: {new_wall:.2f}s | Throughput: {len(new_latencies)/new_wall:.1f} ops/s")
        print(f"  Timeouts / Exhaustion: {new_timeouts} / 50 | Errors: {new_errors}")
        if new_latencies:
            print(f"  Latency: Avg: {statistics.mean(new_latencies):.1f}ms | p50: {statistics.median(new_latencies):.1f}ms | Max: {max(new_latencies):.1f}ms")
        print()

if __name__ == "__main__":
    test_comparison()
