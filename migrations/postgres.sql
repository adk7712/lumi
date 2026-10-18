-- Lumi Postgres schema migrations (mirrors persistence.MIGRATIONS).
-- All statements are idempotent and safe on an existing production DB that already
-- has the sessions table (with pinned) and no schema_version table.
-- Application order: schema_version table, then migrations 1..3, each recorded after it runs.

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    name TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Migration 1: create_sessions_table
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    filename TEXT,
    cleaning_recipe TEXT,
    step_count INTEGER,
    rules TEXT,
    scanned_columns TEXT,
    user_id TEXT,
    project_name TEXT,
    pinned INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO schema_version (version, name) VALUES (1, 'create_sessions_table') ON CONFLICT (version) DO NOTHING;

-- Migration 2: add_sessions_pinned_column
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS pinned INTEGER DEFAULT 0;
INSERT INTO schema_version (version, name) VALUES (2, 'add_sessions_pinned_column') ON CONFLICT (version) DO NOTHING;

-- Migration 3: add_sessions_indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at);
INSERT INTO schema_version (version, name) VALUES (3, 'add_sessions_indexes') ON CONFLICT (version) DO NOTHING;
