"""Fuzz: generated patterns over adversarial cells never hang (one worker, per-pattern alarm)."""
import random
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
ATOMS = ["a", "b", ".", r"\w", r"\d", r"\s", "[ab]", "[^a]", "(a|b)", "(a|ab)", "(?:a+)", "x"]
QUANTS = ["", "", "*", "+", "?", "{0,2}", "{1,3}", "{2}", "{1,}"]
# hand-picked dangerous families (some rejected by the guard, some accepted and timed out at runtime)
DANGEROUS = [r"(a+)+$", r"(a|aa)+$", r"\w+a\w+a\w+!", r"(a*)*b", r".*.*x", r"[ab]*[ab]*c",
             r"(a)+?(a){2,40}[ab]$", r"[^x]{1,30}(?:a){2,}(a|ab)$", r"[^x]{1,30}a[a-z]+\d\d{30}$"]

# One child evaluates every (pattern, cell) job; SIGALRM bounds each so a hang is reported, not stuck.
CHILD = r"""
import json, signal, sys, time
import pandas as pd
sys.path.insert(0, %r)
import query_guard as qg
import engine_ops

class Hung(BaseException):
    pass

def _alarm(*_):
    raise Hung()

signal.signal(signal.SIGALRM, _alarm)
out = []
for pat, cell in json.load(sys.stdin):
    t = time.time()
    status = "ok"
    signal.setitimer(signal.ITIMER_REAL, 3.5)
    try:
        if not qg.unsafe_regex_reason(pat):
            df = pd.DataFrame({"t": [cell] * 3})
            engine_ops._handle_replace(df, {"column": "t", "find": pat, "replace": "X", "regex": True})
    except Hung:
        status = "hung"
    except Exception as e:
        status = "error: " + repr(e)[:150]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    out.append([pat, status, round(time.time() - t, 2)])
print(json.dumps(out))
""" % ROOT


def _patterns(n=120, seed=7):
    rnd = random.Random(seed)
    return DANGEROUS + ["".join(rnd.choice(ATOMS) + rnd.choice(QUANTS) for _ in range(rnd.randint(2, 12)))
                        + rnd.choice(["", "b", "!", "$", "c"]) for _ in range(n)]


def test_fuzz_no_hangs():
    import json
    rnd = random.Random(1)
    jobs = [[pat, rnd.choice(["a" * 1000, "ab" * 500, "a" * 999 + "!", " " * 1000, "1" * 1000])]
            for pat in _patterns()]
    r = subprocess.run([sys.executable, "-c", CHILD], input=json.dumps(jobs), capture_output=True,
                       text=True, timeout=120, env={"LUMI_TESTING": "1", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-400:]
    bad = [row for row in json.loads(r.stdout.strip().splitlines()[-1]) if row[1] != "ok" or row[2] > 3.0]
    assert not bad, bad
