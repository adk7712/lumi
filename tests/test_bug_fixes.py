import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.express as px
from scout import generate_proposals
from engine import apply_recipe

def test_scout_string_dtype():
    # Test string dtype column detection in scout
    df = pd.DataFrame({
        'col_str': pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], dtype='string')
    })
    
    proposals = generate_proposals(df, set())
    
    # Assert that a high cardinality warning is generated for col_str
    # because it is treated as a text column and has 100% unique values.
    high_card_proposals = [p for p in proposals if p['type'] == 'High Cardinality' and p['column'] == 'col_str']
    assert len(high_card_proposals) > 0, "Should detect High Cardinality in string dtype column"
    print("Scout string dtype test passed.")

def test_intermediate_states_simulation():
    # Simulate intermediate states list as maintained in state_manager.py
    df_raw = pd.DataFrame({
        'name': ['  Alice  ', 'Bob'],
        'age': [25, 30]
    })
    
    # Setup initial state (3-tuple: desc, health, row_count — no dataframe)
    intermediate_states = []
    bh = 100
    intermediate_states.append(("Original Data", bh, len(df_raw)))
    current_df = df_raw.copy()
    
    # Apply a step
    step1 = {'action': 'strip_whitespace', 'column': 'name'}
    new_df, messages = apply_recipe(current_df, [step1])
    
    # Append new state metadata only
    intermediate_states.append(("strip_whitespace on name", bh, len(new_df)))
    current_df = new_df
    
    # Verify current_df holds the latest state
    assert all(current_df['name'] == ['Alice', 'Bob']), "current_df should have whitespace stripped"
    # Verify intermediate_states only has metadata (3-tuple)
    assert len(intermediate_states[-1]) == 3, "intermediate_states should store 3-tuples (no dataframe)"
    print("Intermediate states simulation test passed.")

def test_diagnostics_histogram_grouping():
    # Setup series with 15 unique values
    vals = [f"val_{i}" for i in range(15)]
    series = pd.Series(vals)
    counts = series.value_counts()
    num_uniques = len(counts)
    
    assert num_uniques == 15
    
    top_n = counts.head(9)
    other_sum = counts.iloc[9:].sum()
    chart_data = pd.concat([top_n, pd.Series({"Other": other_sum})])
    
    assert len(chart_data) == 10
    assert chart_data["Other"] == 6
    print("Diagnostics histogram grouping test passed.")

def test_chart_updates():
    df_numeric = pd.DataFrame({'age': [20, 25, 30]})
    fig_box = px.box(df_numeric, x='age', height=220)
    fig_box.update_layout(hovermode=False)
    assert fig_box.layout.hovermode is False, "Figure layout hovermode should be False"
    assert fig_box.data[0].orientation == 'h', "Box plot should be horizontal"
    
    chart_data = pd.Series([5, 3], index=['A', 'B'])
    fig_bar = px.bar(x=chart_data.index, y=chart_data.values, height=220)
    assert fig_bar.data[0].orientation is None or fig_bar.data[0].orientation == 'v', "Bar orientation should be vertical"
    assert all(fig_bar.data[0].x == ['A', 'B']), "Bar x should represent category labels"
    assert all(fig_bar.data[0].y == [5, 3]), "Bar y should represent counts"
    print("Chart updates test passed.")

def test_correlation_range_filtering():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [2, 4, 6, 8, 10],
        'C': [5, 1, 9, 2, 8]
    })
    corr_matrix = df.corr()
    corr_matrix_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_matrix_no_diag.values, np.nan)
    
    # 1. High positive range [0.9, 1.0] -> A and B have r=1.0, so they qualify. C does not.
    range_high = (0.9, 1.0)
    in_range_mask = (corr_matrix_no_diag >= range_high[0]) & (corr_matrix_no_diag <= range_high[1])
    correlated_cols_high = corr_matrix_no_diag.columns[in_range_mask.any()].tolist()
    assert 'A' in correlated_cols_high
    assert 'B' in correlated_cols_high
    assert 'C' not in correlated_cols_high
    
    # 2. Middle range [-0.2, 0.2] -> A and B have r=1.0, so they are filtered out.
    range_mid = (-0.2, 0.2)
    in_range_mask_mid = (corr_matrix_no_diag >= range_mid[0]) & (corr_matrix_no_diag <= range_mid[1])
    correlated_cols_mid = corr_matrix_no_diag.columns[in_range_mask_mid.any()].tolist()
    assert 'A' not in correlated_cols_mid
    assert 'B' not in correlated_cols_mid
    print("Correlation range filtering unit test passed.")

def test_uploader_large_file_capping():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"))
    at.run()
    
    # Generate an 52MB CSV dataset with 52,000 rows
    # 52,000 rows, each row has a column with 1,000 'x' characters (~52MB total)
    header = b"A,B\n"
    row = b"1.0," + b"x"*1000 + b"\n"
    large_csv_bytes = header + row * 52000
    
    
    # Upload via AppTest uploader
    uploader = at.file_uploader(key="welcome_uploader")
    assert uploader is not None
    uploader.upload("large_file.csv", large_csv_bytes, "text/csv").run()
    
    # Check that it loaded exactly MAX_SAMPLE_ROWS (10,000)
    assert len(at.session_state.raw_data) == 10000
    assert not at.exception
    print("Uploader large file performance protection test passed.")

def test_plot_data_downsampling():
    # Verify the downsampling logic handles various row counts correctly
    # If len(df) > 1000, it samples exactly 1000
    df_large = pd.DataFrame({'A': range(5000)})
    plot_df_large = df_large.sample(1000, random_state=42) if len(df_large) > 1000 else df_large
    assert len(plot_df_large) == 1000
    
    # If len(df) <= 1000, it retains all rows
    df_small = pd.DataFrame({'A': range(500)})
    plot_df_small = df_small.sample(1000, random_state=42) if len(df_small) > 1000 else df_small
    assert len(plot_df_small) == 500
    print("Plot data downsampling logic test passed.")

def test_plotly_layout_consolidation():
    from ui_utils import plot_correlation_matrix
    import streamlit as st
    st.cache_data.clear()
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [2, 4, 6]})
    fig = plot_correlation_matrix(df, (-1.0, 1.0))
    assert fig is not None
    assert fig.layout.margin.t == 10
    assert fig.layout.margin.b == 10
    assert fig.layout.margin.l == 10
    assert fig.layout.margin.r == 10
    assert fig.layout.font.family == "JetBrains Mono, Courier New, monospace"
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    print("Plotly layout consolidation test passed.")

def test_get_column_dependencies():
    """get_column_dependencies should return rule descriptions that reference the target column."""
    import streamlit as st
    from state_manager import get_column_dependencies
    # Patch session_state.rules
    rules = [
        {'col': 'age', 'desc': 'age is not null', 'type': 'Null Check'},
        {'col_a': 'age', 'col_b': 'score', 'desc': 'age > score', 'type': 'Relational'},
        {'col': 'name', 'desc': 'name length check', 'type': 'Null Check'},
        {'type': 'Custom Expression', 'query': 'age > 0', 'desc': 'custom age expr'},
    ]
    # Directly inject via session_state dict (works outside Streamlit runtime)
    st.session_state['rules'] = rules
    deps = get_column_dependencies('age')
    assert 'age is not null' in deps
    assert 'age > score' in deps
    assert 'custom age expr' in deps
    assert 'name length check' not in deps
    assert len(deps) == 3
    print("get_column_dependencies test passed.")


def test_sync_column_rename():
    """sync_column_rename should update matching rules and active_features in session state."""
    import streamlit as st
    from state_manager import sync_column_rename
    rules = [
        {'col': 'old', 'desc': 'old is not null', 'type': 'Null Check'},
        {'col_a': 'old', 'col_b': 'score', 'desc': 'old > score', 'type': 'Relational'},
        {'col_b': 'old', 'col_a': 'x', 'desc': 'x vs old', 'type': 'Relational'},
        {'col': 'other', 'desc': 'other rule', 'type': 'Null Check'},
    ]
    st.session_state['rules'] = rules
    st.session_state['active_features'] = ['old', 'score']
    sync_column_rename('old', 'new')
    # Rules should be updated
    assert st.session_state['rules'][0]['col'] == 'new'
    assert 'new' in st.session_state['rules'][0]['desc']
    assert st.session_state['rules'][1]['col_a'] == 'new'
    assert st.session_state['rules'][2]['col_b'] == 'new'
    # Untouched rule should remain
    assert st.session_state['rules'][3]['col'] == 'other'
    # active_features should be updated
    assert 'new' in st.session_state['active_features']
    assert 'old' not in st.session_state['active_features']
    assert 'score' in st.session_state['active_features']
    print("sync_column_rename test passed.")


def test_get_loading_spinner_html():
    """get_loading_spinner_html should return HTML containing the provided label and spinner div."""
    from ui_utils import get_loading_spinner_html
    html = get_loading_spinner_html("Apply Column Order")
    assert 'Apply Column Order' in html
    assert 'spinner-circle' in html
    assert '<button disabled' in html
    # Custom text
    html2 = get_loading_spinner_html("Processing...")
    assert 'Processing...' in html2
    print("get_loading_spinner_html test passed.")


def test_datetime_feature_extraction():
    """Verify that extract_datetime correctly extracts components (year, month, day, day_of_week, hour) and integrates with codegen."""
    from engine import apply_recipe
    from codegen import generate_pipeline_code
    
    # Create test DataFrame
    df = pd.DataFrame({
        'timestamp': [
            '2026-06-11 14:30:00',
            '2025-12-25 08:15:00',
            '2024-01-01 00:00:00'
        ]
    })
    
    # Year, month, day, day of week, hour extractions
    recipe = [
        {
            'action': 'extract_datetime',
            'column': 'timestamp',
            'new_column': 'timestamp_year',
            'component': 'year'
        },
        {
            'action': 'extract_datetime',
            'column': 'timestamp',
            'new_column': 'timestamp_month',
            'component': 'month'
        },
        {
            'action': 'extract_datetime',
            'column': 'timestamp',
            'new_column': 'timestamp_day',
            'component': 'day'
        },
        {
            'action': 'extract_datetime',
            'column': 'timestamp',
            'new_column': 'timestamp_day_name',
            'component': 'day_of_week'
        },
        {
            'action': 'extract_datetime',
            'column': 'timestamp',
            'new_column': 'timestamp_hour',
            'component': 'hour'
        }
    ]
    
    df_cleaned, msgs = apply_recipe(df.copy(), recipe)
    assert len(msgs) == 0
    
    # Verify outputs
    assert df_cleaned['timestamp_year'].tolist() == [2026, 2025, 2024]
    assert df_cleaned['timestamp_month'].tolist() == [6, 12, 1]
    assert df_cleaned['timestamp_day'].tolist() == [11, 25, 1]
    assert df_cleaned['timestamp_day_name'].tolist() == ['Thursday', 'Thursday', 'Monday']
    assert df_cleaned['timestamp_hour'].tolist() == [14, 8, 0]
    
    # Test codegen compilation and run
    code_str = generate_pipeline_code(recipe)
    
    # Execute the generated code
    namespace = {}
    exec(code_str, namespace)
    
    clean_func = namespace['clean_data']
    df_gen_cleaned = clean_func(df.copy())
    
    # Verify generated function has same result
    assert df_gen_cleaned['timestamp_year'].tolist() == [2026, 2025, 2024]
    assert df_gen_cleaned['timestamp_month'].tolist() == [6, 12, 1]
    assert df_gen_cleaned['timestamp_day'].tolist() == [11, 25, 1]
    assert df_gen_cleaned['timestamp_day_name'].tolist() == ['Thursday', 'Thursday', 'Monday']
    assert df_gen_cleaned['timestamp_hour'].tolist() == [14, 8, 0]
    
    print("test_datetime_feature_extraction passed.")


def test_notebook_export():
    """Verify that generate_notebook_code produces a valid JSON structure representing a Jupyter Notebook."""
    import json
    from codegen import generate_notebook_code
    
    recipe = [
        {'action': 'drop_column', 'column': 'unused'}
    ]
    rules = [
        {'col': 'age', 'desc': 'age is not null', 'type': 'Null Check', 'enabled': True}
    ]
    
    nb_str = generate_notebook_code(recipe, rules)
    assert nb_str is not None
    
    # Verify it is valid JSON
    data = json.loads(nb_str)
    
    # Check top-level keys
    assert 'cells' in data
    assert 'metadata' in data
    assert data['nbformat'] == 4
    
    cells = data['cells']
    assert len(cells) == 5
    
    # Cell 0: Markdown overview
    assert cells[0]['cell_type'] == 'markdown'
    assert any("Lumi Data Cleaning" in line for line in cells[0]['source'])
    
    # Cell 1: Imports
    assert cells[1]['cell_type'] == 'code'
    assert any("import pandas" in line for line in cells[1]['source'])
    
    # Cell 2: clean_data
    assert cells[2]['cell_type'] == 'code'
    assert any("def clean_data" in line for line in cells[2]['source'])
    assert any("drop(columns=['unused'])" in line for line in cells[2]['source'])
    
    # Cell 3: validate_data
    assert cells[3]['cell_type'] == 'code'
    assert any("def validate_data" in line for line in cells[3]['source'])
    assert any("age" in line for line in cells[3]['source'])
    
    # Cell 4: Example usage
    assert cells[4]['cell_type'] == 'code'
    assert any("Example Usage" in line for line in cells[4]['source'])
    
    print("test_notebook_export passed.")


def test_evidence_report_generation():
    """Verify that generate_evidence_report correctly identifies violations, handles exceptions, and formats the output Markdown."""
    from rule_utils import generate_evidence_report
    import streamlit as st
    import unittest.mock as mock
    
    df = pd.DataFrame({
        'age': [25, np.nan, 30, 45, 12],
        'score': [90, 80, 70, 60, np.nan]
    })
    
    rules = [
        {'col': 'age', 'desc': 'age is NOT NULL', 'type': 'Null Check', 'enabled': True},
        {'col': 'score', 'min': 0.0, 'max': 100.0, 'desc': 'score in [0.0, 100.0]', 'type': 'Range Check', 'enabled': True},
        {'col': 'nonexistent', 'desc': 'fails check', 'type': 'Null Check', 'enabled': True}, # This should trigger an ERROR status
        {'desc': 'informational check', 'type': 'Informational', 'enabled': True}
    ]
    
    # Mock Streamlit session state and runtime
    with mock.patch("streamlit.runtime.exists", return_value=True):
        st.session_state.intermediate_states = [
            ("Original Data", 80, len(df)),
            ("Final Cleaned", 100, len(df))
        ]
        st.session_state.current_df = df.copy()
        st.session_state.cleaning_recipe = [
            {"action": "drop_column", "column": "redundant"},
            {"action": "rename_column", "column": "old", "value": "new"}
        ]
        
        report_md = generate_evidence_report(
            df, 
            rules,
            cleaning_recipe=st.session_state.cleaning_recipe,
            original_df=df.copy()
        )
        
    assert report_md is not None
    
    # Assert report header and totals
    assert "# LUMI - Data Validation Evidence Report" in report_md
    assert "Total Rows Evaluated: 5" in report_md
    assert "Total Active Rules: 4" in report_md
    assert "Total Rule Violations: 2" in report_md  # age null (1) + score nan/null (1) = 2
    
    # Assert metrics block
    assert "## Data Cleaning Impact Metrics" in report_md
    assert "| Metric | Original Dataset | Cleaned Dataset | Change |" in report_md
    assert "Dimensions" in report_md
    
    # Assert lineage block
    assert "## Data Lineage & Audit Log" in report_md
    assert "Dropped column `redundant`" in report_md
    assert "Renamed column `old` to `new`" in report_md
    
    # Assert table contents
    assert "| Null Check | `age is NOT NULL` | FAILED | 1 |" in report_md
    assert "| Range Check | `score in [0.0, 100.0]` | FAILED | 1 |" in report_md
    assert "| Null Check | `fails check` | ERROR | 0 |" in report_md
    assert "| Informational | `informational check` | INFO | N/A |" in report_md
    
    # Assert violation details
    assert "### ❌ Null Check: `age is NOT NULL`" in report_md
    assert "* **Violation Count:** 1" in report_md
    assert "* **Violating Row Indices:** `[1]`" in report_md
    
    # Assert error trace is handled gracefully
    assert "### ⚠️ Null Check: `fails check`" in report_md
    assert "* **Status:** Evaluation Error" in report_md
    assert "KeyError" in report_md or "ValueError" in report_md
    
    print("test_evidence_report_generation passed.")


def test_string_dtype_operations():
    # 1. Scout whitespace detection with 'string' dtype
    df_scout = pd.DataFrame({
        'name': pd.Series(['  Alice  ', 'Bob  ', 'Charlie'], dtype='string')
    })
    proposals = generate_proposals(df_scout, set())
    assert any(p['type'] == 'Formatting Issue' and p['column'] == 'name' for p in proposals)

    # 2. Engine actions targeting 'All' with 'string' dtype
    df_engine = pd.DataFrame({
        'name': pd.Series(['  Alice  ', 'Bob  ', '  Charlie'], dtype='string'),
        'city': pd.Series(['nyc', 'london', 'tokyo'], dtype='string')
    })
    
    # Strip whitespace
    df_stripped, _ = apply_recipe(df_engine.copy(), [{'action': 'strip_whitespace', 'column': 'All'}])
    assert list(df_stripped['name']) == ['Alice', 'Bob', 'Charlie']
    assert df_stripped['name'].dtype == 'string'

    # Replace
    df_replaced, _ = apply_recipe(df_engine.copy(), [{'action': 'replace', 'column': 'All', 'find': 'nyc', 'replace': 'NYC', 'regex': False}])
    assert df_replaced.loc[0, 'city'] == 'NYC'
    assert df_replaced['city'].dtype == 'string'

    # Normalize text
    df_norm, _ = apply_recipe(df_engine.copy(), [{'action': 'normalize_text', 'column': 'All', 'value': 'uppercase'}])
    assert list(df_norm['city']) == ['NYC', 'LONDON', 'TOKYO']
    assert df_norm['city'].dtype == 'string'

    print("test_string_dtype_operations passed.")


if __name__ == "__main__":
    try:
        test_scout_string_dtype()
        test_string_dtype_operations()
        test_intermediate_states_simulation()
        test_diagnostics_histogram_grouping()
        test_chart_updates()
        test_correlation_range_filtering()
        test_uploader_large_file_capping()
        test_plot_data_downsampling()
        test_plotly_layout_consolidation()
        test_get_column_dependencies()
        test_sync_column_rename()
        test_get_loading_spinner_html()
        test_datetime_feature_extraction()
        test_notebook_export()
        test_evidence_report_generation()
        print("ALL BUG FIX TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
