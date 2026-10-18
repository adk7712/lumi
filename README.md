# LUMI

LUMI is a professional data cleaning and validation platform designed to transform raw datasets into high-quality, analysis-ready assets. By combining automated intelligence with intuitive visual tools, LUMI streamlines the journey from messy data to reliable insights.

## Automated Data Quality Scouting
Eliminate the manual effort of hunting for errors. LUMI’s proactive scouting engine automatically scans your dataset to identify inconsistencies, outliers, and structural issues, offering intelligent recommendations to fix them instantly.

## Interactive Data Diagnostics
Understand your data at a glance. Navigate through rich, interactive diagnostic cards that visualize feature distributions, null value density, and unique value counts, providing a comprehensive map of your data’s health and composition.

## Visual Insights Dashboard
Gain deep visual understanding of your dataset. The visual insights panel features:
* **Correlation range filtering:** A dual-handle range slider lets you isolate specific positive or negative correlation scales across your features.
* **Missingness patterns:** A binary matrix visualizer maps where missing values reside across rows and columns to identify systemic data collection gaps.
* **Standardized outlier box plots:** All variables are scaled to statistical Z-scores for a direct, stacked comparison of outliers side-by-side.

## Comprehensive Cleaning Toolkit
Take full control of your data transformations. LUMI provides a versatile suite of tools for precise data manipulation, including column renaming, interactive column reordering, whitespace stripping, smart type casting, find-and-replace operations, and sophisticated outlier management.

## Custom Rulebook & Validation
Enforce rigorous quality standards with a tailored rulebook. Define and apply custom validation constraints to ensure your data adheres to specific business logic and integrity requirements, preventing bad data from reaching your downstream systems.

## Data Lineage & Audit Log
Maintain a transparent record of every modification. Every transformation is captured in a detailed audit log, providing complete data lineage so you can track, review, and verify the evolution of your dataset from its raw state to the final export.

## One-Click Pipeline Export
Move from discovery to production seamlessly. Once you have perfected your cleaning workflow, export the entire logic as a reusable Python script. This allows you to deploy your validated data pipeline into any production environment with a single click.

## Configuration

Settings come from environment variables or Streamlit secrets (`.streamlit/secrets.toml` locally, the Secrets box on Streamlit Cloud). See `.streamlit/secrets.toml.example` for every key with placeholders.

| Key | Purpose |
|---|---|
| `env` / `LUMI_ENV` | `production` or `development` (default). In production a missing DB URL logs an error and shows a storage warning. |
| `DB_URL` / `DATABASE_URL` | Postgres URL (Supabase transaction pooler, port 6543, `sslmode=require`). Without it Lumi uses local SQLite. |
| `[auth]` `redirect_uri`, `cookie_secret` | Streamlit OIDC redirect and HMAC key for Lumi cookies. |
| `[auth.google]` `client_id`, `client_secret`, `server_metadata_url` | Google OAuth credentials. |
| `POSTHOG_API_KEY` | Optional analytics. |
| `dev_auth` / `LUMI_DEV_AUTH=1` | Email mock login. Development only, never in production. |

## Deployment

### Streamlit Community Cloud
1. Push the repo to GitHub, then create a new app at share.streamlit.io: choose the repo, branch, and main file `app.py`.
2. In Advanced settings, set Python version to 3.13.
3. Paste your secrets (based on `.streamlit/secrets.toml.example`) with `env = "production"`, `DB_URL`, `[auth]`, `[auth.google]`. Generate `cookie_secret` with `python -c "import secrets;print(secrets.token_urlsafe(48))"`. Leave `dev_auth` unset.
4. In Google Cloud console (APIs & Services > Credentials), add `https://<app>.streamlit.app/oauth2callback` as an authorized redirect URI and set the same value as `redirect_uri` in secrets.

### Supabase (Postgres)
Use the **transaction pooler** connection string (port 6543, host `*.pooler.supabase.com`) with `?sslmode=require` as `DB_URL`. Do not use the direct 5432 connection. Schema migrations run automatically on startup via `persistence.init_db`; the reference schema is in `migrations/postgres.sql`. SQLite on Streamlit Cloud is ephemeral, so production requires `DB_URL`.

### Local development
```
pip install -r requirements.txt
LUMI_DEV_AUTH=1 streamlit run app.py   # email mock login, SQLite storage
```

### Running tests
```
pip install -r requirements-dev.txt
LUMI_TESTING=1 python -m pytest -q tests/
```
