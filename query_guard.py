"""Safety validator for user-written ``DataFrame.query`` expressions (Custom Expression rules).

Stdlib-only on purpose: codegen embeds this file's source verbatim in the exported pipeline
script, so the app and the script share one validator.

``df.query`` can evaluate arbitrary Python (``@`` locals, attribute chains, calls such as
``__import__``).  We parse the expression with ``ast`` and only allow comparisons, boolean /
arithmetic operators, column names, literals, ``x.isnull()``-style null tests and a few
``x.str.contains/startswith/endswith(...)`` calls.  Backtick-quoted column names are supported.
"""
import ast
import re
import string
import time

try:  # optional: the third-party ``regex`` package can abort a match after a timeout
    import regex as _regex
except ImportError:  # pragma: no cover - exercised only where regex is not installed
    _regex = None

MAX_QUERY_LENGTH = 500
MAX_REGEX_LENGTH = 200
MAX_COUNTED_REPEAT = 50
# Each variable quantifier (* + ? {n,} {n,m}) multiplies the worst-case backtracking; real
# patterns (phone, date, e-mail, id) use at most 3-4, so more than 4 is rejected.  At most 3
# of them may be unbounded (n**k steps on an n-char cell); `\w+@\w+\.\w+$`, `(\d+)-(\d+)-(\d+)` fit.
MAX_VARIABLE_QUANTIFIERS = 4
MAX_UNBOUNDED_QUANTIFIERS = 3
# str.contains in a Custom Expression runs inside df.query, where no timeout can be applied:
# two open-ended repeats already take ~6 s on a 2,000-char cell, so allow one (two when the
# pattern is anchored with ^, which limits matching to a single start position).
CONTAINS_MAX_UNBOUNDED = 1
# A bounded repeat spanning more than this many counts ({1,30}) backtracks like an open-ended one.
WIDE_REPEAT_SPAN = 5
CONTAINS_MAX_UNBOUNDED_ANCHORED = 2
REGEX_TIME_BUDGET = 2.0  # seconds of regex matching allowed per column in a Replace step
_OVERLAP_ALPHABET = string.printable + "\u00e9\u00df\u4e2d"
_NULL_METHODS = {"isnull", "notnull", "isna", "notna"}
_STR_METHODS = {"contains", "startswith", "endswith"}
_STR_KWARGS = {"case", "na", "regex"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.Not, ast.USub, ast.UAdd)


class UnsafeQueryError(ValueError):
    """Raised when a Custom Expression query is outside the allowed subset."""


class RegexTimeoutError(RuntimeError):
    """Raised when a regex replace exceeds its time budget."""


def _quantifier_at(pattern, i):
    """Parse a quantifier at ``pattern[i]``: ``(lo, hi, end)`` (``hi`` None = unbounded) or None."""
    if i >= len(pattern):
        return None
    c = pattern[i]
    if c == "*":
        lo, hi, end = 0, None, i + 1
    elif c == "+":
        lo, hi, end = 1, None, i + 1
    elif c == "?":
        lo, hi, end = 0, 1, i + 1
    elif c == "{":
        m = re.compile(r"\{(\d+)(?:(,)(\d*))?\}").match(pattern, i)
        if not m:
            return None
        lo = int(m.group(1))
        hi = (int(m.group(3)) if m.group(3) else None) if m.group(2) else lo
        end = m.end()
    else:
        return None
    if end < len(pattern) and pattern[end] in "?+":  # lazy / possessive suffix
        end += 1
    return lo, hi, end


def _atoms_overlap(a, b):
    """True when some character is matched by both single-character atoms ``a`` and ``b``."""
    try:
        ca, cb = re.compile(a, re.S), re.compile(b, re.S)
    except re.error:
        return True
    return any(ca.fullmatch(ch) and cb.fullmatch(ch) for ch in _OVERLAP_ALPHABET)


def _quantifier_reason(pattern, max_unbounded=MAX_UNBOUNDED_QUANTIFIERS):
    """Reject patterns with too many variable quantifiers or adjacent overlapping repeats."""
    variable = unbounded = 0
    prev = [None]  # per group depth: (atom_source, is_variable) of the previous atom, or None
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        atom, end = None, i + 1
        if c == "\\":
            end = min(i + 2, n)
            nxt = pattern[i + 1] if i + 1 < n else ""
            atom = pattern[i:end] if nxt and not nxt.isdigit() and nxt not in "bBAZ" else None
        elif c == "[":
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            end = min(j + 1, n)
            atom = pattern[i:end]
        elif c == "(":
            prev.append(None)
            end = i + 1
            if pattern.startswith("(?", i):
                m = re.match(r"\?(?:P?<[^>]*>|P=[^)]*|[:=!]|<[=!]|[aiLmsux-]+:?)?", pattern[i + 1:])
                end = i + 1 + (m.end() if m else 1)
            prev[-1] = None
            i = end
            continue
        elif c == ")" and len(prev) > 1:
            prev.pop()
            atom = None
        elif c in "|^$":
            prev[-1] = None
            i += 1
            continue
        elif c == ".":
            atom = "."
        else:
            atom = re.escape(c)
        q = _quantifier_at(pattern, end)
        if q is None:
            prev[-1] = (atom, False) if atom else None
            i = end
            continue
        lo, hi, end = q
        is_var = lo != hi
        if is_var:
            variable += 1
            unbounded += hi is None or hi - lo > WIDE_REPEAT_SPAN
            if variable > MAX_VARIABLE_QUANTIFIERS:
                return (f"it has more than {MAX_VARIABLE_QUANTIFIERS} variable repeats "
                        "(*, +, ?, {n,}, {n,m}), which can backtrack catastrophically")
            if unbounded > max_unbounded:
                return (f"it has more than {max_unbounded} open-ended repeats "
                        "(*, +, {n,} or a wide {n,m}), which can backtrack catastrophically")
            p = prev[-1]
            if atom and p and p[1] and _atoms_overlap(p[0], atom):
                return "it has two adjacent repeats that can match the same characters (catastrophic backtracking risk)"
        prev[-1] = (atom, is_var) if atom else None
        i = end
    return None


def unsafe_regex_reason(pattern, max_unbounded=MAX_UNBOUNDED_QUANTIFIERS):
    """Return why ``pattern`` is too risky to run as a regex (ReDoS), else ``None``.

    Rejects patterns longer than ``MAX_REGEX_LENGTH`` and repeated groups that themselves
    contain a repeat, e.g. ``(a+)+``, ``(a*)*``, ``(.*a){10}``.  Escapes and ``[...]``
    classes are skipped.  Non-text patterns are not checked.
    """
    if not isinstance(pattern, str):
        return None
    if len(pattern) > MAX_REGEX_LENGTH:
        return f"the pattern is longer than {MAX_REGEX_LENGTH} characters"

    def _repeat_at(i):
        # (kind, index_after_quantifier); kind: 0 none/zero-or-one handled below,
        # "opt" for ?, "many" for repeats that can run more than once, "big" for large counts
        if i >= len(pattern):
            return None, i
        c = pattern[i]
        if c in "*+":
            return "many", i + 1
        if c == "?":
            return "opt", i + 1
        if c == "{":
            end = pattern.find("}", i)
            body = pattern[i + 1:end] if end > 0 else ""
            if end > 0 and body and all(ch.isdigit() or ch == "," for ch in body) and body[0] != ",":
                lo, _, hi = body.partition(",")
                top = int(hi) if hi else int(lo)
                if "," in body and hi == "":
                    return "many", end + 1
                if top > MAX_COUNTED_REPEAT:
                    return "big", end + 1
                if top > 1:
                    return "many", end + 1
                return "opt" if "," in body else None, end + 1
        return None, i

    # Each stack frame: [holds_any_quantifier, holds_alternation]
    stack, i, n = [[False, False]], 0, len(pattern)
    reason = ("it repeats a group that already contains a repeat or alternation "
              "(catastrophic backtracking risk)")

    def _note(kind):
        if kind:
            stack[-1][0] = True

    while i < n:
        c = pattern[i]
        if c == "\\":
            i += 2
            kind, i = _repeat_at(i)
            _note(kind)
            continue
        if c == "[":
            j = i + 1
            if j < n and pattern[j] == "^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 2 if pattern[j] == "\\" else 1
            kind, i = _repeat_at(j + 1)
            _note(kind)
            continue
        if c == "(":
            stack.append([False, False])
            i += 1
            # skip a group prefix such as (?:, (?=, (?P<name> so its '?' is not a quantifier
            if pattern.startswith("(?", i - 1):
                m = re.match(r"\?(?:P?<[^>]*>|P=[^)]*|[:=!]|<[=!]|[aiLmsux-]+:?)?", pattern[i:])
                i += m.end() if m else 1
            continue
        if c == "|":
            stack[-1][1] = True
            i += 1
            continue
        if c == ")" and len(stack) > 1:
            inner_q, inner_alt = stack.pop()
            kind, i = _repeat_at(i + 1)
            if kind == "big":
                return "it uses a very large repeat count on a group"
            if kind == "many" and (inner_q or inner_alt):
                return reason
            if kind == "opt" and inner_q and inner_alt:
                return reason
            stack[-1][0] = stack[-1][0] or inner_q or bool(kind)
            stack[-1][1] = stack[-1][1] or inner_alt
            continue
        kind, i = _repeat_at(i + 1)
        if kind == "big":
            return "it uses a very large repeat count"
        _note(kind)
    return _quantifier_reason(pattern, max_unbounded)


def regex_deadline(budget=REGEX_TIME_BUDGET):
    """Monotonic deadline to share across the columns of one Replace step."""
    return time.monotonic() + budget


def safe_regex_replace(series, pattern, repl, budget=REGEX_TIME_BUDGET, deadline=None):
    """``series.replace(pattern, repl, regex=True)`` with a time budget for the whole column.

    Pass ``deadline`` (from ``regex_deadline()``) to share one budget across several columns.

    Same semantics as pandas (text cells only, ``re.sub`` replacement templates).  With the
    ``regex`` package installed each cell is matched with a timeout and ``RegexTimeoutError``
    is raised once ``budget`` seconds are spent; without it the static guard is the only defence.
    """
    if _regex is None:
        return series.replace(pattern, repl, regex=True)
    try:
        compiled = _regex.compile(pattern)
    except _regex.error:
        return series.replace(pattern, repl, regex=True)  # let ``re`` report or accept it
    re.compile(pattern)  # same error type as pandas raises for an invalid pattern
    if deadline is None:
        deadline = time.monotonic() + budget

    def _sub(cell):
        if not isinstance(cell, str):
            return cell
        left = deadline - time.monotonic()
        if left <= 0:
            raise RegexTimeoutError(f"regular expression took longer than {budget:g}s")
        try:
            return compiled.sub(repl, cell, timeout=left)
        except TimeoutError:
            raise RegexTimeoutError(f"regular expression took longer than {budget:g}s") from None

    return series.astype(object).map(_sub)


def _check_node(node, in_call_arg=False):
    if isinstance(node, ast.BoolOp):
        for v in node.values:
            _check_node(v)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARY):
            raise UnsafeQueryError("unsupported operator")
        _check_node(node.operand)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeQueryError("unsupported arithmetic operator")
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, (str, bytes)):
                raise UnsafeQueryError("arithmetic on text literals is not allowed")
            _check_node(side)
    elif isinstance(node, ast.Compare):
        _check_node(node.left)
        for c in node.comparators:
            _check_node(c)
    elif isinstance(node, ast.Name):
        if "__" in node.id:
            raise UnsafeQueryError("names containing double underscores are not allowed")
    elif isinstance(node, ast.Constant):
        if not (node.value is None or isinstance(node.value, (str, int, float, bool))):
            raise UnsafeQueryError("unsupported literal")
    elif isinstance(node, (ast.List, ast.Tuple)):
        for e in node.elts:
            _check_node(e)
    elif isinstance(node, ast.Call):
        _check_call(node)
    else:
        raise UnsafeQueryError(f"{type(node).__name__} expressions are not allowed")


def _check_call(node):
    f = node.func
    if not isinstance(f, ast.Attribute):
        raise UnsafeQueryError("function calls are not allowed")
    # col.isnull() / col.notnull() / col.isna() / col.notna()
    if f.attr in _NULL_METHODS and isinstance(f.value, ast.Name):
        if node.args or node.keywords:
            raise UnsafeQueryError(f".{f.attr}() takes no arguments")
        _check_node(f.value)
        return
    # col.str.contains(...) / .startswith(...) / .endswith(...) with literal arguments
    if (f.attr in _STR_METHODS and isinstance(f.value, ast.Attribute) and f.value.attr == "str"
            and isinstance(f.value.value, ast.Name)):
        _check_node(f.value.value)
        for a in node.args:
            if not isinstance(a, ast.Constant):
                raise UnsafeQueryError("string method arguments must be literals")
        for k in node.keywords:
            if k.arg not in _STR_KWARGS or not isinstance(k.value, ast.Constant):
                raise UnsafeQueryError("unsupported string method argument")
        if f.attr == "contains":
            # str.contains is a regex match unless regex=False (5th positional / keyword).
            opts = {k.arg: k.value.value for k in node.keywords}
            if len(node.args) > 4:
                opts["regex"] = node.args[4].value
            if node.args and opts.get("regex", True) is not False:
                pat = node.args[0].value
                limit = (CONTAINS_MAX_UNBOUNDED_ANCHORED if isinstance(pat, str) and pat.startswith("^")
                         else CONTAINS_MAX_UNBOUNDED)
                why = unsafe_regex_reason(pat, max_unbounded=limit)
                if why:
                    raise UnsafeQueryError(f"unsafe regular expression: {why}")
        return
    raise UnsafeQueryError("only .isnull(), .notnull() and .str.contains/startswith/endswith() calls are allowed")


def _hide_backticks(expr):
    """Replace each backticked column name outside text literals with ``_bt_col``.

    The scan follows Python's string rules (quotes, backslash escapes), as pandas does, so a
    backtick inside a literal can never hide code that sits between two literals.
    """
    if "'''" in expr or '"""' in expr:
        raise UnsafeQueryError("triple-quoted text is not allowed")
    out, i, quote = [], 0, None
    while i < len(expr):
        c = expr[i]
        if quote:
            if c == "\\":
                out.append(expr[i:i + 2])
                i += 2
                continue
            if c == "`":
                raise UnsafeQueryError("backticks are not allowed inside text literals")
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c == "`":
            end = expr.find("`", i + 1)
            if end < 0:
                raise UnsafeQueryError("unbalanced backtick")
            out.append("_bt_col")
            i = end + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def validate_query_expression(expr) -> str:
    """Return ``expr`` unchanged if it is a safe query, else raise ``UnsafeQueryError``."""
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeQueryError("the expression is empty")
    if len(expr) > MAX_QUERY_LENGTH:
        raise UnsafeQueryError(f"the expression is longer than {MAX_QUERY_LENGTH} characters")
    if "@" in expr:
        raise UnsafeQueryError("'@' variable references are not allowed")
    # Backticked column names are plain column lookups; hide them from the parser.
    stripped = _hide_backticks(expr)
    if "__" in stripped:
        raise UnsafeQueryError("double underscores are not allowed")
    try:
        tree = ast.parse(stripped.strip(), mode="eval")
    except (SyntaxError, ValueError) as e:
        raise UnsafeQueryError(f"invalid syntax ({e})") from None
    _check_node(tree.body)
    return expr


def safe_query_index(df, expr):
    """Index of rows matching ``expr`` after validation (python engine: no numexpr variance)."""
    validate_query_expression(expr)
    try:
        return df.query(expr, engine="python").index
    except RecursionError:
        raise UnsafeQueryError("expression is nested too deeply") from None
