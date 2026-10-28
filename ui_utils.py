import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st
from rule_utils import evaluate_rule
import os
import hmac as _hmac_mod
import hashlib as _hashlib_mod
import secrets as _secrets_mod

# Constants for UI plots and displays
PLOT_SAMPLE_SIZE = 1000
DIAGNOSTIC_CHART_HEIGHT = 220
MAX_CATEGORIES_DISPLAY = 10
VIOLATION_PREVIEW_LIMIT = 100

# ---------------------------------------------------------------------------
# Auth hardening helpers (dev-auth gate + HMAC-signed cookies)
# ---------------------------------------------------------------------------
USER_EMAIL_COOKIE = "lumi_user_email"
SESSION_COOKIE = "lumi_session"
_TRUTHY = {"1", "true", "yes", "on"}
# Per-process fallback secret: cookies signed with it stop verifying after a restart,
# which simply degrades users to a fresh guest session.
_PROCESS_COOKIE_SECRET = _secrets_mod.token_bytes(32)


def _is_truthy(value) -> bool:
    return str(value).strip().lower() in _TRUTHY


_DEV_AUTH_PROD_WARNED = False


def _is_production_env() -> bool:
    try:
        import persistence
        return persistence.get_env() == "production"
    except Exception:
        # Fail closed: if the environment cannot be determined, honour LUMI_ENV directly.
        return str(os.environ.get("LUMI_ENV", "")).strip().lower() == "production"


def is_dev_auth_enabled() -> bool:
    """Passwordless mock email login is only allowed with an explicit dev flag.

    Enabled by env ``LUMI_DEV_AUTH=1``, ``st.secrets["dev_auth"] = true``, or ``LUMI_TESTING=1``.
    Always disabled in production (``persistence.get_env() == "production"``), whatever the flags.
    """
    global _DEV_AUTH_PROD_WARNED
    flag = _is_truthy(os.environ.get("LUMI_DEV_AUTH", "")) or _is_truthy(os.environ.get("LUMI_TESTING", ""))
    if not flag:
        try:
            flag = _is_truthy(st.secrets.get("dev_auth", False))
        except Exception:
            flag = False
    if flag and _is_production_env():
        if not _DEV_AUTH_PROD_WARNED:
            _DEV_AUTH_PROD_WARNED = True
            import logging
            logging.getLogger("lumi.auth").warning(
                "A dev-auth flag (LUMI_DEV_AUTH / LUMI_TESTING / secret dev_auth) is set in production; "
                "ignoring it and keeping passwordless dev login disabled."
            )
        return False
    return flag


def _get_cookie_secret() -> bytes:
    """Secret for signing Lumi cookies: [auth].cookie_secret, env LUMI_COOKIE_SECRET, else per-process random."""
    try:
        if "auth" in st.secrets:
            sec = st.secrets["auth"].get("cookie_secret")
            if sec:
                return str(sec).encode("utf-8")
    except Exception:
        pass
    env_sec = os.environ.get("LUMI_COOKIE_SECRET")
    if env_sec:
        return env_sec.encode("utf-8")
    return _PROCESS_COOKIE_SECRET


def _cookie_signature(value: str, purpose: str) -> str:
    msg = f"lumi-cookie-v1|{purpose}|{value}".encode("utf-8")
    return _hmac_mod.new(_get_cookie_secret(), msg, _hashlib_mod.sha256).hexdigest()


def sign_cookie_value(value: str, purpose: str) -> str:
    """Return ``value.<hmac-sha256>`` bound to ``purpose`` (prevents cross-cookie reuse)."""
    return f"{value}.{_cookie_signature(str(value), purpose)}"


def verify_cookie_value(raw, purpose: str):
    """Return the original value if ``raw`` carries a valid signature, else None (legacy/forged cookies)."""
    if not raw or not isinstance(raw, str) or "." not in raw:
        return None
    value, sig = raw.rsplit(".", 1)
    if not value or not sig:
        return None
    if _hmac_mod.compare_digest(sig, _cookie_signature(value, purpose)):
        return value
    return None


# ---------------------------------------------------------------------------
# Cookie controller: exactly one CookieController per script run.
#
# streamlit_cookies_controller binds the browser's cookie jar to the widget key
# ``cookies``. The first instance in a run renders that widget; any further instance
# in the same run tries ``st.session_state["cookies"] = ...`` and Streamlit raises
# "cannot be modified after the widget with key cookies is instantiated", so the
# second caller's set()/get() silently failed (that is why lumi_session was never
# written). All Lumi code goes through get_cookie_controller() instead.
#
# A set()/remove() renders a hidden component whose iframe writes document.cookie.
# Callers usually st.rerun() right after (e.g. after an upload), which can discard
# that iframe before it runs, so writes are also queued and re-asserted at the top
# of the next few runs by sync_cookies() (idempotent).
# ---------------------------------------------------------------------------
_COOKIE_CTRL_STATE_KEY = "_lumi_cookie_ctrl"
_COOKIE_OPS_STATE_KEY = "_lumi_cookie_ops"
_COOKIE_REASSERT_RUNS = 3
# Browser lifetime for Lumi cookies (the library default is 1 day). The HMAC signature
# carries no timestamp, so this max_age is the only expiry.
COOKIE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60


def _current_run_marker():
    """An object unique to the current script run (None outside a Streamlit run).

    ``ScriptRunContext.reset()`` replaces ``widget_ids_this_run`` with a fresh set at the
    start of every run; we hold a reference to it, so identity comparison is reliable.
    """
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        return getattr(ctx, "widget_ids_this_run", None) if ctx is not None else None
    except Exception:
        return None


def get_cookie_controller():
    """Return this run's single CookieController (created on first use in the run), or None."""
    marker = _current_run_marker()
    if marker is not None:
        cached = st.session_state.get(_COOKIE_CTRL_STATE_KEY)
        if isinstance(cached, tuple) and len(cached) == 2 and cached[0] is marker:
            return cached[1]
    try:
        from streamlit_cookies_controller import CookieController
        controller = CookieController()
    except Exception as e:
        import logging
        logging.getLogger("lumi.auth").warning("Cookie controller unavailable: %s", e)
        return None
    if marker is not None:
        st.session_state[_COOKIE_CTRL_STATE_KEY] = (marker, controller)
    return controller


def _apply_cookie_op(controller, name, value, max_age=None):
    if value is None:
        try:
            controller.remove(name)
        except KeyError:
            pass  # library pops its local copy without a default; browser removal was still issued
    else:
        controller.set(name, value, max_age=max_age if max_age is not None else COOKIE_MAX_AGE_SECONDS)


def _queue_cookie_op(name, value, max_age=None):
    try:
        ops = dict(st.session_state.get(_COOKIE_OPS_STATE_KEY) or {})
        ops[name] = [value, _COOKIE_REASSERT_RUNS, max_age]
        st.session_state[_COOKIE_OPS_STATE_KEY] = ops
    except Exception:
        pass


def set_cookie(name: str, value: str, max_age=None) -> bool:
    """Write a cookie now and re-assert it on the next runs (survives an immediate st.rerun).

    ``max_age`` is in seconds; defaults to COOKIE_MAX_AGE_SECONDS (30 days).
    """
    _queue_cookie_op(name, value, max_age)
    controller = get_cookie_controller()
    if controller is None:
        return False
    try:
        _apply_cookie_op(controller, name, value, max_age)
        return True
    except Exception as e:
        import logging
        logging.getLogger("lumi.auth").warning("Error setting cookie %s: %s", name, e)
        return False


def remove_cookie(name: str) -> bool:
    """Delete a cookie now and re-assert the deletion on the next runs."""
    _queue_cookie_op(name, None)
    controller = get_cookie_controller()
    if controller is None:
        return False
    try:
        _apply_cookie_op(controller, name, None)
        return True
    except Exception:
        return False


def sync_cookies():
    """Call once near the top of app.py: creates the run's controller, replays queued writes."""
    controller = get_cookie_controller()
    ops = st.session_state.get(_COOKIE_OPS_STATE_KEY)
    if controller is None or not ops:
        return
    remaining = {}
    for name, op in list(ops.items()):
        value, runs_left = op[0], op[1]
        max_age = op[2] if len(op) > 2 else None
        try:
            _apply_cookie_op(controller, name, value, max_age)
        except Exception:
            pass
        if runs_left - 1 > 0:
            remaining[name] = [value, runs_left - 1, max_age]
    st.session_state[_COOKIE_OPS_STATE_KEY] = remaining


def read_session_cookie(controller=None):
    """Read and verify the signed ``lumi_session`` cookie; unsigned/tampered values yield None."""
    try:
        if controller is None:
            controller = get_cookie_controller()
        if controller is None:
            return None
        return verify_cookie_value(controller.get(SESSION_COOKIE), purpose="session")
    except Exception:
        return None


def is_auth_configured() -> bool:
    """Check if streamlit OIDC auth credentials are set in secrets.toml."""
    try:
        if "auth" in st.secrets:
            auth_sec = st.secrets["auth"]
            if "google" in auth_sec:
                return True
    except Exception:
        pass
    return False

def get_logged_in_user() -> str:
    """Get the active user's email.

    Resolution order (S1/S2 hardening):
      1. Real Streamlit OIDC identity (``st.user``) always wins.
      2. Server-side ``dev_user_email`` in session state: anonymous ``guest_*`` ids are
         always honoured; any other email only when dev auth is enabled.
      3. HMAC-signed ``lumi_user_email`` cookie, only when dev auth is enabled.
         Unsigned/forged/tampered cookies are ignored.
    """
    try:
        if hasattr(st, "user") and st.user and st.user.get("email"):
            return st.user.get("email")
    except Exception:
        pass

    dev_enabled = is_dev_auth_enabled()

    dev_email = st.session_state.get("dev_user_email")
    if dev_email:
        if str(dev_email).startswith("guest_") or dev_enabled:
            return dev_email
        # A non-guest mock email without the dev flag is never a valid identity.
        st.session_state.pop("dev_user_email", None)

    if dev_enabled and not st.session_state.get("_is_testing", False):
        try:
            controller = get_cookie_controller()
            raw_email = controller.get(USER_EMAIL_COOKIE) if controller is not None else None
            cookie_email = verify_cookie_value(raw_email, purpose="user_email")
            if cookie_email and "@" in cookie_email:
                st.session_state.dev_user_email = cookie_email
                return cookie_email
        except Exception:
            pass

    return None

def is_authenticated_user() -> bool:
    """Check if the user is signed in with a real account (not an anonymous guest)."""
    user_email = get_logged_in_user()
    if not user_email or user_email.startswith("guest_"):
        return False
    return True

def format_timestamp(iso_str: str) -> str:
    """Formats an ISO timestamp into a readable date/time string like 'Jul 31, 20:15'."""
    if not iso_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%b %d, %H:%M")
    except Exception:
        return str(iso_str)[:16]

def handle_signout():
    """Sign out the user, clearing local dev mock email, removing cookies, resetting state, and redirecting to landing page."""
    if "dev_user_email" in st.session_state:
        st.session_state.pop("dev_user_email", None)
    remove_cookie(USER_EMAIL_COOKIE)
    try:
        st.logout()
    except Exception:
        pass
    from state_manager import initialize_state
    initialize_state(from_reset=True)
    st.rerun()

@st.dialog("Sign Out")
def show_signout_dialog():
    st.write("Are you sure you want to sign out?")
    col1, col2 = st.columns(2)
    if col1.button("Yes, Sign Out", type="primary", use_container_width=True, key="confirm_signout_btn"):
        handle_signout()
    if col2.button("Cancel", use_container_width=True, key="cancel_signout_btn"):
        st.rerun()

@st.dialog("Lumi Authentication")
def show_auth_dialog():
    _render_auth_options()


def _render_auth_options():
    """Body of the sign-in dialog (split out so it is testable outside a dialog context)."""
    st.markdown('<p style="color: #a3a3a3; font-size: 0.95rem; text-align: center; margin-bottom: 1.5rem;">Sign in to sync and manage your cleaning workspaces.</p>', unsafe_allow_html=True)
    
    # 1. Dev / Local Email Login (passwordless mock) -- only behind an explicit dev flag
    if is_dev_auth_enabled():
        st.markdown("**Sign In with Email (dev mode)**")
        user_email_input = st.text_input("Email address", value="user@example.com", key="auth_dev_email_input")
        if st.button("Sign In", key="auth_dev_email_login_btn", use_container_width=True, type="primary"):
            if not is_dev_auth_enabled():
                st.error("Email sign-in is disabled.")
            elif user_email_input and "@" in user_email_input:
                email_clean = user_email_input.strip()
                st.session_state.dev_user_email = email_clean
                set_cookie(USER_EMAIL_COOKIE, sign_cookie_value(email_clean, purpose="user_email"))
                st.toast(f"Signed in as {email_clean}")
                st.rerun()
            else:
                st.error("Please enter a valid email address.")

        st.markdown('<div style="text-align: center; margin: 1rem 0; color: #555;">— OR —</div>', unsafe_allow_html=True)
    elif not is_auth_configured():
        st.info("Account sign-in is not configured on this deployment. You can keep using Lumi as a guest.")

    # 2. Google Login Button (if configured)
    if is_auth_configured():
        st.button(
            "Sign In with Google", 
            key="btn_login_google_auth", 
            use_container_width=True, 
            on_click=st.login, 
            kwargs={"provider": "google"}
        )
        st.markdown('<div style="text-align: center; margin: 1rem 0; color: #555;">— OR —</div>', unsafe_allow_html=True)
            
    # 3. Guest / Dev Login
    if st.button("Continue as Anonymous Guest", key="auth_dev_login_btn", use_container_width=True):
        import uuid
        guest_id = f"guest_{uuid.uuid4().hex}@lumi.ai"
        st.session_state.dev_user_email = guest_id
        st.toast("Started temporary guest session")
        st.rerun()

def downsample_for_plot(df_or_series, sample_size: int = PLOT_SAMPLE_SIZE):
    """Downsamples a DataFrame or Series if it exceeds sample_size, returning the sample and an is_sampled bool."""
    if len(df_or_series) > sample_size:
        return df_or_series.sample(sample_size, random_state=42).sort_index(), True
    return df_or_series, False

def render_diagnostic_metric(container, label: str, value: str):
    """
    Render a diagnostic metric with small label text using custom HTML.
    This bypasses Streamlit's metric styling limitations.

    Args:
        container: A streamlit column/container object
        label: The metric label (Type, Nulls, Unique, etc.)
        value: The metric value (int64, 0, 3, etc.)
    """
    container.markdown(f"""
    <div class="diagnostic-metric">
        <div class="diagnostic-metric-label">
            {label}
        </div>
        <div class="diagnostic-metric-value">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

def inject_custom_css(st_object):
    """Injects the global CSS for the Lumi workspace."""
    # Read the CSS from the external file using absolute path relative to this script
    css_path = Path(__file__).parent / "styles" / "global.css"

    # Fallback to the current working directory if file not found
    if not css_path.exists():
        css_path = Path("styles") / "global.css"

    try:
        with open(css_path, "r") as f:
            css_content = f.read()
        st_object.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st_object.error(f"CSS file not found at {css_path}. Please ensure styles/global.css is in the app directory.")

def load_style(name: str) -> str:
    """Reads a stylesheet from the styles directory and returns it as a string."""
    css_path = Path(__file__).parent / "styles" / name
    if not css_path.exists():
        css_path = Path("styles") / name
    try:
        with open(css_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def get_safe_hue(n: int) -> int:
    """Returns a high-contrast hue that avoids red and green."""
    # Select hues that avoid red/green to improve accessibility and visual distinction.
    safe_hues = [
        200, 240, 280, 310, 185, 220, 260, 300, 330,
        170, 210, 250, 290, 320, 195, 230, 270, 305, 340
    ]
    return safe_hues[n % len(safe_hues)]

def get_heatmap_styles(df_d: pd.DataFrame, rules: list) -> tuple[pd.DataFrame, list[str]]:
    """Generates the styling dataframe for the validation heatmap."""
    # Create a DataFrame to hold style information, highlighting cells with rule violations.
    sdf = pd.DataFrame('', index=df_d.index, columns=df_d.columns)
    messages = []
    for r in rules:
        if not r.get('enabled', True):
            continue
        try:
            violation_mask = evaluate_rule(df_d, r)
            style = f"background-color: {r['color']};"

            rule_type = r.get('type')
            if rule_type == "Custom Expression":
                sdf.loc[violation_mask, :] = style
            elif rule_type == "Relational Check":
                sdf.loc[violation_mask, r['col_a']] = style
                if r.get('target_type') == 'Feature':
                    sdf.loc[violation_mask, r['col_b']] = style
            else:
                if 'col' in r:
                    sdf.loc[violation_mask, r['col']] = style

        except Exception as e:
            messages.append(f"Heatmap Style Error ({r.get('desc', 'N/A')}): {str(e)}")
            continue
    return sdf, messages

def apply_lumi_layout(fig: go.Figure) -> go.Figure:
    """Applies a consistent theme and layout configuration to a Plotly figure."""
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        font_family="JetBrains Mono, Courier New, monospace",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Chart caches skip hashing the DataFrame (too slow), so ``df_state_key`` must identify its
# content: same file + same recipe always gives the same frame. A step *count* is not enough
# (undo then a different step reuses it, and other users with the same file share the cache).
CHART_CACHE_MAX_ENTRIES = 64


def chart_state_key() -> str:
    """Cache key for charts of the current working frame: file hash + recipe content hash."""
    import json
    recipe = st.session_state.get("cleaning_recipe", [])
    sig = _hashlib_mod.sha1(json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"{st.session_state.get('last_file_hash', 'none')}_{sig}"


@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, max_entries=CHART_CACHE_MAX_ENTRIES)
def plot_correlation_matrix(df: pd.DataFrame, corr_range: tuple, df_state_key: str = "default") -> go.Figure:
    """Computes features' correlation matrix and plots a filtered heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        corr_matrix_no_diag = corr_matrix.copy()
        np.fill_diagonal(corr_matrix_no_diag.values, np.nan)
        in_range_mask = (corr_matrix_no_diag >= corr_range[0]) & (corr_matrix_no_diag <= corr_range[1])
        correlated_cols = corr_matrix_no_diag.columns[in_range_mask.any()].tolist()

        if len(correlated_cols) > 1:
            filtered_corr = corr_matrix.loc[correlated_cols, correlated_cols]
            fig = px.imshow(
                filtered_corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1]
            )
            return apply_lumi_layout(fig)
    return None

@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, max_entries=CHART_CACHE_MAX_ENTRIES)
def plot_missingness_map(df: pd.DataFrame, df_state_key: str = "default") -> tuple:
    """Generates a binary missingness pattern map representation."""
    if df.size > 0:
        null_mask = df.isnull().astype(int)
        if null_mask.sum().sum() > 0:
            vis_df, is_sampled = downsample_for_plot(null_mask)

            fig = px.imshow(
                vis_df,
                aspect="auto",
                color_continuous_scale=[[0, "#2c3e50"], [0.5, "#2c3e50"], [0.5, "#e74c3c"], [1, "#e74c3c"]],
                labels=dict(x="Columns", y="Row Index", color="Status")
            )
            apply_lumi_layout(fig)
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="Status",
                    tickvals=[0.25, 0.75],
                    ticktext=["Present", "Missing"]
                ),
                yaxis_title="Row Index"
            )
            return fig, is_sampled
    return None, False


def get_loading_spinner_html(text: str = "Apply Column Order") -> str:
    """Returns HTML for a disabled button accompanied by a CSS loading spinner."""
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 12px; height: 38px; margin-bottom: 1rem;">
        <button disabled style="
            border-radius: 6px;
            border: 1px solid rgba(128, 128, 128, 0.2);
            background-color: transparent;
            color: inherit;
            opacity: 0.4;
            padding: 0px 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            height: 38px;
            cursor: not-allowed;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
        ">{text}</button>
        <div class="spinner-circle"></div>
    </div>
    """

@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, max_entries=CHART_CACHE_MAX_ENTRIES)
def plot_outlier_distribution(df: pd.DataFrame, df_state_key: str = "default") -> tuple:
    """Computes column Z-scores and plots comparative outlier box plots."""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        z_scored_df = pd.DataFrame()
        for col in numeric_df.columns:
            col_std = numeric_df[col].std()
            if col_std > 0:
                z_scored_df[col] = (numeric_df[col] - numeric_df[col].mean()) / col_std
            else:
                z_scored_df[col] = 0.0

        plot_z_df, is_sampled = downsample_for_plot(z_scored_df)

        melted_z = plot_z_df.melt(var_name="Feature", value_name="Standardized Value")

        fig = px.box(
            melted_z,
            x="Standardized Value",
            y="Feature",
            color="Feature",
            height=max(200, 50 * len(numeric_df.columns)),
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        apply_lumi_layout(fig)
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            xaxis_title="Standardized Value (Z-Score)"
        )
        return fig, is_sampled
    return None, False

MAX_BOX_COLUMNS = 12


def zscore_candidate_columns(df: pd.DataFrame) -> list:
    """Numeric columns with at least two non-null values and non-zero variance."""
    cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col]
        if s.notna().sum() > 1 and (s.std() or 0) > 0:
            cols.append(col)
    return cols


def build_zscore_boxplot(df: pd.DataFrame, columns=None, max_columns: int = MAX_BOX_COLUMNS) -> tuple:
    """Pure builder: side-by-side box plots of z-scored numeric columns.

    Constant / all-null columns are skipped. Returns (figure_or_None, is_sampled).
    """
    candidates = zscore_candidate_columns(df)
    if columns is not None:
        wanted = set(columns)
        candidates = [c for c in candidates if c in wanted]
    candidates = candidates[:max_columns]
    if not candidates:
        return None, False
    z_df = pd.DataFrame({c: (df[c] - df[c].mean()) / df[c].std() for c in candidates})
    plot_z_df, is_sampled = downsample_for_plot(z_df)
    melted = plot_z_df.melt(var_name="Feature", value_name="Z-Score").dropna()
    fig = px.box(
        melted, x="Feature", y="Z-Score", color="Feature",
        height=420, color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    apply_lumi_layout(fig)
    fig.update_layout(showlegend=False, hovermode="closest",
                      xaxis_title=None, yaxis_title="Standardized Value (Z-Score)")
    return fig, is_sampled


@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, max_entries=CHART_CACHE_MAX_ENTRIES)
def plot_zscore_boxes(df: pd.DataFrame, columns: tuple, df_state_key: str = "default") -> tuple:
    """Cached wrapper around build_zscore_boxplot (keyed by df_state_key like other charts)."""
    return build_zscore_boxplot(df, columns)


_posthog_nested_warned = False

def _find_nested_secret(sec, key):
    """Return `key` from any nested table of `sec` (TOML puts keys below a [table] header into it)."""
    for k in sec:
        val = sec[k]
        if hasattr(val, "keys"):
            if key in val:
                return val[key]
            found = _find_nested_secret(val, key)
            if found:
                return found
    return None

def inject_posthog(st_object):
    """Injects PostHog tracking script if a key is configured in st.secrets or environment variables."""
    posthog_key = None
    try:
        if "POSTHOG_API_KEY" in st_object.secrets:
            posthog_key = st_object.secrets["POSTHOG_API_KEY"]
        else:
            posthog_key = _find_nested_secret(st_object.secrets, "POSTHOG_API_KEY")
            global _posthog_nested_warned
            if posthog_key and not _posthog_nested_warned:
                _posthog_nested_warned = True
                import logging
                logging.getLogger("lumi.analytics").warning(
                    "POSTHOG_API_KEY is nested under a [table] in secrets; move it above [auth]"
                )
    except Exception:
        pass

    if not posthog_key:
        import os
        posthog_key = os.getenv("POSTHOG_API_KEY")
        
    if posthog_key:
        posthog_js = f"""
        <script>
            !function(t,e){{var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){{function g(t,e){{var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){{t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}}} (p=t.createElement("script")).type="text/javascript",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){{var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e}},u.people.toString=function(){{return u.toString(".people")}},o="capture identify alias people.set people.set_once set_config register register_once unregister opt_out_capturing has_opted_out_capturing opt_in_capturing list_properties single_pageview disable_sharing_assets_checking".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])}},e.__SV=1)}}(document,window.posthog||[]);
            posthog.init('{posthog_key}', {{
                api_host: 'https://us.i.posthog.com',
                person_profiles: 'identified_only'
            }})
        </script>
        """
        st_object.html(posthog_js)

def queue_event(event_name: str, properties: dict = None):
    """Queues a PostHog event to be captured on the next render pass."""
    if "pending_posthog_events" not in st.session_state:
        st.session_state.pending_posthog_events = []
    st.session_state.pending_posthog_events.append((event_name, properties or {}))

def flush_pending_events():
    """Renders any queued PostHog events via JS and clears the queue."""
    if "pending_posthog_events" in st.session_state and st.session_state.pending_posthog_events:
        import json
        js_lines = []
        for event_name, properties in st.session_state.pending_posthog_events:
            props_str = json.dumps(properties)
            js_lines.append(f"if(window.parent && window.parent.posthog) {{ window.parent.posthog.capture('{event_name}', {props_str}); }} else if(window.posthog) {{ window.posthog.capture('{event_name}', {props_str}); }}")
        js = "<script>\n" + "\n".join(js_lines) + "\n</script>"
        st.html(js)
        st.session_state.pending_posthog_events = []