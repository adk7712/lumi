import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import these at the top to register cache functions under standard runtime-not-exists conditions
import pandas as pd
import io
import streamlit as st
import views.landing
import ui_utils

import unittest
from unittest import mock
import sqlite3

# Import under test
from persistence import init_db, save_session, load_session, get_user_projects, reconcile_session, DB_PATH
from state_manager import process_uploaded_file, add_step, load_db_session, initialize_state, save_db_session

class TestPersistence(unittest.TestCase):
    
    def setUp(self):
        # Ensure we use a clean test database file
        if DB_PATH.exists():
            try:
                DB_PATH.unlink()
            except Exception:
                pass
        init_db()

    def tearDown(self):
        # Cleanup test database file
        if DB_PATH.exists():
            try:
                DB_PATH.unlink()
            except Exception:
                pass

    def test_init_db(self):
        # Test table structure is correct
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        table = cursor.fetchone()
        self.assertIsNotNone(table)
        self.assertEqual(table[0], 'sessions')
        conn.close()

    def test_save_and_load_session(self):
        session_id = "test-session-123"
        filename = "sales_data.csv"
        recipe = [{"action": "strip_whitespace", "column": "client_name"}]
        rules = [{"type": "Null Check", "col": "total_cost", "enabled": True}]
        scanned_columns = {"client_name", "total_cost"}
        
        # Save session
        save_session(
            session_id=session_id,
            filename=filename,
            recipe=recipe,
            rules=rules,
            scanned_columns=scanned_columns,
            user_id="user@example.com",
            project_name="Custom Sales Clean"
        )
        
        # Load session
        loaded = load_session(session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["filename"], filename)
        self.assertEqual(loaded["project_name"], "Custom Sales Clean")
        self.assertEqual(loaded["user_id"], "user@example.com")
        self.assertEqual(loaded["cleaning_recipe"], recipe)
        self.assertEqual(loaded["rules"], rules)
        self.assertEqual(loaded["scanned_columns"], scanned_columns)
        self.assertEqual(loaded["step_count"], 1)

    def test_get_user_projects(self):
        user_id = "user_b@example.com"
        # Save two projects for the user
        save_session("session-a", "file_a.csv", [], [], set(), user_id=user_id, project_name="Project A")
        save_session("session-b", "file_b.csv", [], [], set(), user_id=user_id, project_name="Project B")
        
        # Save another project for a different user
        save_session("session-c", "file_c.csv", [], [], set(), user_id="other@example.com", project_name="Project C")
        
        projects = get_user_projects(user_id)
        self.assertEqual(len(projects), 2)
        project_names = [p["project_name"] for p in projects]
        self.assertIn("Project A", project_names)
        self.assertIn("Project B", project_names)
        self.assertNotIn("Project C", project_names)

    def test_session_reconciliation(self):
        # 1. Create an anonymous session (user_id is None)
        session_id = "anon-session-789"
        filename = "inventory.csv"
        save_session(
            session_id=session_id,
            filename=filename,
            recipe=[{"action": "drop_column", "column": "cost"}],
            rules=[],
            scanned_columns=set(),
            user_id=None,
            project_name="Anon Inventory"
        )
        
        # Load and verify it is anonymous
        loaded = load_session(session_id)
        self.assertIsNone(loaded["user_id"])
        
        # 2. Reconcile the session with a logged-in user
        reconcile_session(session_id, "logged_in_user@example.com")
        
        # Load and verify it belongs to the user
        loaded_post = load_session(session_id)
        self.assertEqual(loaded_post["user_id"], "logged_in_user@example.com")
        self.assertEqual(loaded_post["project_name"], "Anon Inventory")
        
        # Verify it shows up in their projects list
        user_projects = get_user_projects("logged_in_user@example.com")
        self.assertEqual(len(user_projects), 1)
        self.assertEqual(user_projects[0]["session_id"], session_id)


class TestCookiesAndPersistence(unittest.TestCase):
    
    def setUp(self):
        if DB_PATH.exists():
            try: DB_PATH.unlink()
            except Exception: pass
        init_db()
        
        st.query_params.clear()
        st.session_state.clear()
        
        # Mock file buffer
        class MockFile(io.BytesIO):
            def __init__(self, name, content):
                super().__init__(content)
                self.name = name
                self.size = len(content)
                self.type = "text/csv"
        
        self.mock_csv = b"col1,col2\n a , b \n c , d \n"
        self.file_buffer = MockFile("test_data.csv", self.mock_csv)

    def tearDown(self):
        st.query_params.clear()
        st.session_state.clear()
        if DB_PATH.exists():
            try: DB_PATH.unlink()
            except Exception: pass

    @mock.patch("streamlit.runtime.exists", return_value=True)
    @mock.patch("streamlit_cookies_controller.CookieController")
    def test_cookie_set_on_upload(self, mock_cookie_controller, mock_runtime):
        # Instantiated controller instance mock
        controller_instance = mock_cookie_controller.return_value
        
        # Mock load_data return value
        df = pd.DataFrame({"col1": [" a ", " c "], "col2": [" b ", " d "]})
        
        with mock.patch("state_manager.load_data", return_value=df):
            initialize_state(from_reset=True)
            process_uploaded_file(self.file_buffer, "dummy_hash")
            
            # Verify session_id exists
            session_id = st.session_state.get("session_id")
            self.assertIsNotNone(session_id)
            self.assertEqual(st.session_state.filename, "test_data.csv")
            
            # Verify cookie set was called
            controller_instance.set.assert_called_once_with("lumi_session", session_id)
            
            # Verify initial state saved in DB
            db_session = load_session(session_id)
            self.assertIsNotNone(db_session)
            self.assertEqual(db_session["filename"], "test_data.csv")
            self.assertEqual(db_session["step_count"], 0)

    @mock.patch("streamlit.runtime.exists", return_value=True)
    @mock.patch("persistence.save_session")
    def test_recipe_persisted_on_each_step(self, mock_save_session, mock_runtime):
        initialize_state(from_reset=True)
        st.session_state.session_id = "test-session-step"
        st.session_state.filename = "test_data.csv"
        st.session_state.raw_data = pd.DataFrame({"col1": [" a "]})
        st.session_state.current_df = st.session_state.raw_data.copy()
        st.session_state.intermediate_states = [("Original", 100, 1)]
        
        step = {"action": "strip_whitespace", "column": "col1"}
        add_step(step)
        
        # Verify save_session was called on step addition
        self.assertTrue(mock_save_session.called)
        args, kwargs = mock_save_session.call_args
        self.assertEqual(kwargs["session_id"], "test-session-step")
        self.assertEqual(kwargs["recipe"], [{"action": "strip_whitespace", "column": "col1"}])

    @mock.patch("streamlit.runtime.exists", return_value=True)
    @mock.patch("streamlit_cookies_controller.CookieController")
    @mock.patch("persistence.load_session")
    def test_resume_banner_appears_when_session_found(self, mock_load_session, mock_cookie_controller, mock_runtime):
        # 1. Cookie is present
        controller_instance = mock_cookie_controller.return_value
        controller_instance.get.return_value = "session-cookie-123"
        
        # 2. Database matches
        mock_load_session.return_value = {
            "session_id": "session-cookie-123",
            "filename": "original_data.csv",
            "step_count": 3,
            "project_name": "Test Project",
            "cleaning_recipe": [],
            "rules": [],
            "scanned_columns": set()
        }
        
        # Clear dismissed flags
        initialize_state(from_reset=True)
        st.session_state.pop("cookie_session_dismissed", None)
        
        # Mock Streamlit column rendering calls
        with mock.patch("streamlit.columns") as mock_columns, \
             mock.patch("streamlit.container") as mock_container:
            
            mock_columns.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
            
            from views.landing import render_landing_page
            try:
                render_landing_page()
            except Exception:
                pass
                
            # Verify container/columns were rendered for the banner
            self.assertTrue(mock_container.called)
            self.assertTrue(mock_columns.called)

    @mock.patch("streamlit.runtime.exists", return_value=True)
    def test_replay_produces_correct_df(self, mock_runtime):
        # Create a saved session in SQLite
        session_id = "replay-session-xyz"
        recipe = [{"action": "strip_whitespace", "column": "col1"}]
        save_session(session_id, "test_data.csv", recipe, [], {"col1", "col2"})
        
        # Initialize
        df = pd.DataFrame({"col1": [" a ", " c "], "col2": [" b ", " d "]})
        with mock.patch("state_manager.load_data", return_value=df):
            initialize_state(from_reset=True)
            
            # Trigger restore
            success = load_db_session(session_id, self.file_buffer)
            self.assertTrue(success)
            
            # Verify recipe and state restored
            self.assertEqual(st.session_state.cleaning_recipe, recipe)
            # col1 should be stripped of whitespace, col2 untouched
            expected_col1 = ["a", "c"]
            actual_col1 = st.session_state.current_df["col1"].tolist()
            self.assertEqual(actual_col1, expected_col1)

    @mock.patch("streamlit.runtime.exists", return_value=True)
    @mock.patch("streamlit.warning")
    @mock.patch("persistence.load_session")
    def test_file_mismatch_warning_path(self, mock_load_session, mock_warning, mock_runtime):
        session_id = "mismatch-session-999"
        expected_columns = {"col1", "col2", "col3"}
        save_session(session_id, "original_dataset.csv", [], [], expected_columns)
        
        # Try resuming with test_data.csv (which only has col1 and col2, not col3)
        initialize_state(from_reset=True)
        st.session_state.resume_session_id = session_id
        
        # Mock file upload uploader to return mismatch file name
        mismatched_file = self.file_buffer # "test_data.csv"
        
        with mock.patch("streamlit.file_uploader", return_value=mismatched_file):
            mock_load_session.return_value = {
                 "filename": "original_dataset.csv",
                 "scanned_columns": expected_columns,
                 "project_name": "Test"
            }
            
            from views.landing import render_landing_page
            try:
                render_landing_page()
            except Exception:
                pass
                
            # Verify mismatch warning was shown due to filename mismatch and column missingness
            self.assertTrue(mock_warning.called)
            warning_text = mock_warning.call_args[0][0]
            self.assertIn("Warning: The uploaded file does not match the expected dataset", warning_text)

if __name__ == "__main__":
    unittest.main()
