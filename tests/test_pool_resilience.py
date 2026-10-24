"""BoundedPgPool must never leak connections/semaphore slots; pool construction failures are negatively cached."""
import sqlite3
import time

import psycopg2.pool
import pytest

import persistence
from lumi_db import pool as pool_mod

URL = "postgresql://u:p@fake:5432/db"


class FakeConn:
    def __init__(self, rollback_fails=False, autocommit_fails_in_txn=False):
        self.closed = 0
        self.autocommit = False
        self.status = 0
        self.rollback_fails = rollback_fails
        self.rollbacks = 0

    def get_transaction_status(self):
        return self.status

    def rollback(self):
        self.rollbacks += 1
        if self.rollback_fails:
            raise RuntimeError("rollback boom")
        self.status = 0

    def close(self):
        self.closed = 1

    def __setattr__(self, k, v):
        if k == "autocommit" and getattr(self, "status", 0) != 0:
            raise RuntimeError("set_session cannot be used inside a transaction")
        object.__setattr__(self, k, v)


class FakeRaw:
    """Mimics ThreadedConnectionPool.getconn/putconn(close=...)."""
    def __init__(self):
        self.idle = []
        self.created = []
        self.closed_by_pool = []

    def getconn(self):
        if self.idle:
            return self.idle.pop()
        c = FakeConn()
        self.created.append(c)
        return c

    def putconn(self, conn, close=False):
        if close or conn.closed:
            conn.close()
            self.closed_by_pool.append(conn)
        else:
            self.idle.append(conn)


def make_pool(maxconn=3):
    raw = FakeRaw()
    return pool_mod.BoundedPgPool(raw, maxconn=maxconn, timeout=0.05), raw


def free_slots(p):
    n = 0
    while p._semaphore.acquire(blocking=False):
        n += 1
    for _ in range(n):
        p._semaphore.release()
    return n


def test_rollback_failure_discards_connection():
    p, raw = make_pool()
    c = p.getconn()
    c.status = 2  # in transaction
    c.rollback_fails = True
    p.putconn(c)
    assert c.closed == 1 and c in raw.closed_by_pool and c not in raw.idle
    assert free_slots(p) == 3
    c2 = p.getconn()
    assert c2 is not c and c2.autocommit is True
    p.putconn(c2)


def test_no_exhaustion_after_many_failing_cycles():
    p, raw = make_pool(maxconn=3)
    for _ in range(20):
        c = p.getconn()
        c.status = 2
        c.rollback_fails = True
        p.putconn(c)
    assert free_slots(p) == 3
    held = [p.getconn() for _ in range(3)]
    for c in held:
        p.putconn(c)


def test_prepare_failure_discards_and_retries_without_leak():
    p, raw = make_pool()
    bad = FakeConn(rollback_fails=True)
    bad.status = 2  # would have failed autocommit; rollback also fails
    raw.idle.append(bad)
    c = p.getconn()
    assert c is not bad and bad.closed == 1
    assert free_slots(p) == 2  # only the one healthy checked-out conn holds a slot
    p.putconn(c)
    assert free_slots(p) == 3


def test_checkout_gives_up_without_leaking_slot():
    p, raw = make_pool()
    for _ in range(5):
        b = FakeConn(rollback_fails=True)
        b.status = 2
        raw.idle.append(b)
    with pytest.raises(pool_mod.PoolError):
        p.getconn()
    assert free_slots(p) == 3


def test_closed_connection_is_replaced():
    p, raw = make_pool()
    dead = FakeConn()
    dead.closed = 2
    raw.idle.append(dead)
    c = p.getconn()
    assert c is not dead and c.closed == 0
    p.putconn(c)
    assert free_slots(p) == 3


def test_closed_connection_on_return_not_reused():
    p, raw = make_pool()
    c = p.getconn()
    c.closed = 2
    p.putconn(c)
    assert c not in raw.idle
    assert free_slots(p) == 3


def test_raw_putconn_error_still_releases_slot():
    p, raw = make_pool()
    c = p.getconn()
    raw.putconn = lambda conn, close=False: (_ for _ in ()).throw(RuntimeError("x"))
    with pytest.raises(RuntimeError):
        p.putconn(c)
    assert free_slots(p) == 3
    assert c.closed == 1


# ---- negative cache for pool construction ----

@pytest.fixture
def ctor(monkeypatch):
    state = {"calls": 0, "fail": True}

    def fake_ctor(minconn, maxconn, dsn=None, **kw):
        state["calls"] += 1
        if state["fail"]:
            time.sleep(0.2)  # stands in for connect_timeout
            raise psycopg2.OperationalError("could not connect")
        return FakeRaw()

    monkeypatch.setattr(psycopg2.pool, "ThreadedConnectionPool", fake_ctor)
    pool_mod._get_pg_pool.clear()
    monkeypatch.setattr(pool_mod, "_get_db_url", lambda: URL)
    monkeypatch.setattr(pool_mod, "_log_db_perf", lambda *a, **k: None)
    monkeypatch.setattr(pool_mod, "_fallback_warned", False)
    yield state
    pool_mod._get_pg_pool.clear()


def test_constructor_failure_is_negatively_cached(ctor, monkeypatch):
    monkeypatch.setenv("LUMI_ENV", "production")
    with pytest.raises(persistence.DatabaseUnavailableError):
        persistence.get_db_connection()
    assert ctor["calls"] == 1
    t0 = time.perf_counter()
    with pytest.raises(persistence.DatabaseUnavailableError):
        persistence.get_db_connection()
    assert time.perf_counter() - t0 < 0.1
    assert ctor["calls"] == 1


def test_retries_after_ttl_and_recovers(ctor, monkeypatch):
    monkeypatch.setenv("LUMI_ENV", "production")
    with pytest.raises(persistence.DatabaseUnavailableError):
        persistence.get_db_connection()
    # age the failure past the TTL
    with pool_mod._pool_failures_lock:
        for k, (t, m) in list(pool_mod._pool_failures.items()):
            pool_mod._pool_failures[k] = (t - pool_mod.POOL_FAILURE_TTL - 1, m)
    ctor["fail"] = False
    conn = persistence.get_db_connection()
    assert ctor["calls"] == 2
    assert isinstance(conn, FakeConn)
    assert URL not in pool_mod._pool_failures
    persistence.release_db_connection(conn)


def test_development_falls_back_fast(ctor, tmp_path, monkeypatch):
    db = tmp_path / "fb.db"
    monkeypatch.setattr(pool_mod, "DB_PATH", db)
    monkeypatch.setenv("LUMI_ENV", "development")
    c1 = persistence.get_db_connection()
    assert isinstance(c1, sqlite3.Connection)
    c1.close()
    t0 = time.perf_counter()
    c2 = persistence.get_db_connection()
    assert isinstance(c2, sqlite3.Connection)
    c2.close()
    assert time.perf_counter() - t0 < 0.1
    assert ctor["calls"] == 1
