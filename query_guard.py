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

MAX_QUERY_LENGTH = 500
_NULL_METHODS = {"isnull", "notnull", "isna", "notna"}
_STR_METHODS = {"contains", "startswith", "endswith"}
_STR_KWARGS = {"case", "na", "regex"}
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.Not, ast.USub, ast.UAdd)
_BACKTICK = re.compile(r"`([^`]*)`")


class UnsafeQueryError(ValueError):
    """Raised when a Custom Expression query is outside the allowed subset."""


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
        return
    raise UnsafeQueryError("only .isnull(), .notnull() and .str.contains/startswith/endswith() calls are allowed")


def validate_query_expression(expr) -> str:
    """Return ``expr`` unchanged if it is a safe query, else raise ``UnsafeQueryError``."""
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeQueryError("the expression is empty")
    if len(expr) > MAX_QUERY_LENGTH:
        raise UnsafeQueryError(f"the expression is longer than {MAX_QUERY_LENGTH} characters")
    if "@" in expr:
        raise UnsafeQueryError("'@' variable references are not allowed")
    # Backticked column names are plain column lookups; hide them from the parser.
    stripped = _BACKTICK.sub(lambda m: "_bt_col", expr)
    if "`" in stripped:
        raise UnsafeQueryError("unbalanced backtick")
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
