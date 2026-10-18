import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import pandas as pd
import io
import streamlit as st
import sqlite3

import persistence
persistence.DB_PATH = Path("test_lumi.db")

from persistence import (
    init_db,
    save_session,
    load_session,
    get_user_projects,
    count_user_sessions,
    delete_session,
    rename_session,
    toggle_pin_session
)

from state_manager import (
    CachedFileWrapper,
    cache_uploaded_file,
    load_cached_file,
    delete_cached_file,
    initialize_state
)

from ui_utils import is_authenticated_user

class TestMultiWorkspace(unittest.TestCase):

    def setUp(self):
        persistence._DB_INITIALIZED = False
        if persistence.DB_PATH.exists():
            try: persistence.DB_PATH.unlink()
            except Exception: pass
        init_db()
        initialize_state(from_reset=True)

    def tearDown(self):
        if persistence.DB_PATH.exists():
            try: persistence.DB_PATH.unlink()
            except Exception: pass

    def test_cached_file_wrapper(self):
        csv_bytes = b"col1,col2\n1,2\n3,4\n"
        wrapper_csv = CachedFileWrapper(csv_bytes, "test.csv")
        self.assertEqual(wrapper_csv.name, "test.csv")
        self.assertEqual(wrapper_csv.size, len(csv_bytes))
        self.assertEqual(wrapper_csv.type, "text/csv")
        self.assertEqual(wrapper_csv.read(), csv_bytes)

        xlsx_bytes = b"dummy excel content"
        wrapper_xlsx = CachedFileWrapper(xlsx_bytes, "data.xlsx")
        self.assertEqual(wrapper_xlsx.type, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    def test_authenticated_user_check(self):
        st.session_state.dev_user_email = None
        self.assertFalse(is_authenticated_user())

        st.session_state.dev_user_email = "guest_12345@lumi.ai"
        self.assertFalse(is_authenticated_user())

        st.session_state.dev_user_email = "realuser@company.com"
        self.assertTrue(is_authenticated_user())

    def test_workspace_crud_and_limits(self):
        user_id = "user@test.com"
        
        # Test initial empty
        self.assertEqual(count_user_sessions(user_id), 0)
        self.assertEqual(len(get_user_projects(user_id)), 0)

        # Create 3 workspaces
        save_session("s1", "train.csv", [{"action": "drop"}], [], ["col1"], user_id=user_id, project_name="Workspace 1")
        save_session("s2", "test.csv", [], [], ["col2"], user_id=user_id, project_name="Workspace 2")
        save_session("s3", "data.csv", [], [], [], user_id=user_id, project_name="Workspace 3")

        self.assertEqual(count_user_sessions(user_id), 3)
        projects = get_user_projects(user_id)
        self.assertEqual(len(projects), 3)

        # Pin workspace 2
        toggle_pin_session("s2", user_id)
        projects_after_pin = get_user_projects(user_id)
        self.assertEqual(projects_after_pin[0]["session_id"], "s2")
        self.assertEqual(projects_after_pin[0]["pinned"], 1)

        # Rename workspace 1
        rename_session("s1", "Renamed Workspace 1", user_id)
        s1_loaded = load_session("s1", user_id)
        self.assertEqual(s1_loaded["project_name"], "Renamed Workspace 1")

        # Delete workspace 3
        delete_session("s3", user_id)
        self.assertEqual(count_user_sessions(user_id), 2)
        self.assertIsNone(load_session("s3", user_id))

    def test_guest_user_isolation(self):
        guest_id = "guest_abc123@lumi.ai"
        save_session("g1", "guest.csv", [], [], [], user_id=guest_id)
        
        # Guests should not have listed projects or active count
        self.assertEqual(count_user_sessions(guest_id), 0)
        self.assertEqual(len(get_user_projects(guest_id)), 0)
        self.assertFalse(rename_session("g1", "New Name", guest_id))
        self.assertFalse(delete_session("g1", guest_id))
        self.assertFalse(toggle_pin_session("g1", guest_id))

if __name__ == "__main__":
    unittest.main()
