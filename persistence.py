import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "lumi.db"

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database schema if it does not exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                filename TEXT,
                cleaning_recipe TEXT,
                step_count INTEGER,
                rules TEXT,
                scanned_columns TEXT,
                user_id TEXT,
                project_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Database initialization error: {e}")
    finally:
        conn.close()

def save_session(session_id: str, filename: str, recipe: list, rules: list, scanned_columns: list, user_id: str = None, project_name: str = None):
    """Saves or updates the session details in the SQLite database."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Serialize fields to JSON
        recipe_json = json.dumps(recipe)
        rules_json = json.dumps(rules)
        scanned_columns_json = json.dumps(list(scanned_columns))
        step_count = len(recipe)
        
        # Check if session exists to preserve project_name/user_id/created_at
        cursor.execute("SELECT project_name, user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            db_project_name = row["project_name"]
            db_user_id = row["user_id"]
        else:
            db_project_name = None
            db_user_id = None
            
        final_project_name = project_name or db_project_name
        if not final_project_name:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            base_name = Path(filename).stem if filename else "untitled"
            final_project_name = f"{base_name}_{timestamp}"
            
        final_user_id = user_id or db_user_id
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO sessions (session_id, filename, cleaning_recipe, step_count, rules, scanned_columns, user_id, project_name, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                filename = excluded.filename,
                cleaning_recipe = excluded.cleaning_recipe,
                step_count = excluded.step_count,
                rules = excluded.rules,
                scanned_columns = excluded.scanned_columns,
                user_id = COALESCE(excluded.user_id, sessions.user_id),
                project_name = COALESCE(excluded.project_name, sessions.project_name),
                updated_at = excluded.updated_at
        """, (session_id, filename, recipe_json, step_count, rules_json, scanned_columns_json, final_user_id, final_project_name, now))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving session {session_id}: {e}")
    finally:
        conn.close()

def load_session(session_id: str) -> dict:
    """Loads a session's details from the SQLite database."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return {
                "session_id": row["session_id"],
                "filename": row["filename"],
                "cleaning_recipe": json.loads(row["cleaning_recipe"]),
                "step_count": row["step_count"],
                "rules": json.loads(row["rules"]) if row["rules"] else [],
                "scanned_columns": set(json.loads(row["scanned_columns"])) if row["scanned_columns"] else set(),
                "user_id": row["user_id"],
                "project_name": row["project_name"],
                "updated_at": row["updated_at"]
            }
        return None
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None
    finally:
        conn.close()

def get_user_projects(user_id: str) -> list:
    """Retrieves all sessions/projects belonging to a specific user ordered by last update."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
        rows = cursor.fetchall()
        projects = []
        for row in rows:
            projects.append({
                "session_id": row["session_id"],
                "filename": row["filename"],
                "project_name": row["project_name"],
                "updated_at": row["updated_at"],
                "step_count": row["step_count"]
            })
        return projects
    except Exception as e:
        print(f"Error getting user projects for {user_id}: {e}")
        return []
    finally:
        conn.close()

def reconcile_session(session_id: str, user_id: str):
    """Reconciles an anonymous session by assigning it to a logged-in user."""
    init_db()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE sessions 
            SET user_id = ?, updated_at = ?
            WHERE session_id = ?
        """, (user_id, now, session_id))
        conn.commit()
    except Exception as e:
        print(f"Error reconciling session {session_id} to user {user_id}: {e}")
    finally:
        conn.close()
