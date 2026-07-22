import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import unittest
import os
import sqlite3
from persistence import init_db, save_session, load_session, get_user_projects, reconcile_session, DB_PATH

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

if __name__ == "__main__":
    unittest.main()
