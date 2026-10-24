"""State-management facade.

Implementation lives in the ``lumi_state`` package:
  core       - constants, downcast_dtypes, load_data, initialize_state, health/original-entry helpers
  session_io - local cache + DB session save/restore, cookies, background executor, switch_workspace
  steps      - add_rule/add_step/replay_steps/get_state_at_step, column dependency/rename sync

This module re-exports every public and private name. Assigning an attribute here
(``mock.patch("state_manager.load_data")``, ``monkeypatch.setattr(sm, "CACHE_DIR", ...)``) is
forwarded to each submodule that defines the name, so the override takes effect internally.
"""
from lumi_state import core as _core, session_io as _session_io, steps as _steps
from lumi_state.core import (  # noqa: F401
    logger, MAX_SAMPLE_ROWS, LARGE_FILE_THRESHOLD_BYTES, downcast_dtypes, CachedFileWrapper,
    _hash_file_or_buffer, load_data, read_tabular, calculate_health, build_original_entry,
    reset_intermediate_states, initialize_state,
)
from lumi_state.session_io import (  # noqa: F401
    _db_executor, _update_project_cache, CACHE_DIR, calculate_file_hash, _warn_save_failure_once,
    save_session_state, cache_uploaded_file, load_cached_file, delete_cached_file,
    process_uploaded_file, load_full_data, load_session_state, _rotated_sessions, _on_bg_done, _set_session_cookie,
    _apply_rotated_session, save_db_session, load_db_session, switch_workspace,
)
from lumi_state.steps import (  # noqa: F401
    add_rule, add_step, replay_steps, get_state_at_step, get_column_dependencies, _name_regex,
    _STR_OR_BACKTICK, _rename_in_expression, _rename_in_text, sync_column_rename,
    regenerate_proposals,
)
from facade_util import install_facade

install_facade(__name__, (_core, _session_io, _steps), live=("CACHE_DIR", "_rotated_sessions"))
