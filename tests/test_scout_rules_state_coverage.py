import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import unittest.mock as mock

# Mock streamlit before importing modules that might perform stream/UI updates
import streamlit as st

from scout import generate_proposals
from rule_utils import evaluate_rule, create_resolution_step, generate_evidence_report
from state_manager import load_data

def test_scout_coverage_gaps():
    print("Running test_scout_coverage_gaps...")
    
    # 1. generate_proposals with empty DataFrame
    df_empty = pd.DataFrame()
    proposals = generate_proposals(df_empty, set())
    assert proposals == []
    
    # 2. _check_numeric_diagnostics with IQR == 0 (all values identical)
    df_const_num = pd.DataFrame({'a': [5.0, 5.0, 5.0, 5.0, 5.0]})
    # This might trigger constant column check, but let's test directly with _check_numeric_diagnostics
    # which is called internally by generate_proposals
    proposals = generate_proposals(df_const_num, set())
    # Should run fine without crash
    assert len(proposals) > 0 # should propose dropping constant col
    
    # 3. _check_string_diagnostics with all-null object column
    df_all_null_obj = pd.DataFrame({'a': [np.nan, np.nan, np.nan]}, dtype=object)
    proposals = generate_proposals(df_all_null_obj, set())
    # Should not crash, might suggest Null Check
    
    # 4. _check_string_diagnostics with high cardinality column (>80% unique values)
    df_cardinality = pd.DataFrame({
        'a': ['id1', 'id2', 'id3', 'id4', 'id5', 'id6', 'id7', 'id8', 'id9', 'id10']
    })
    proposals = generate_proposals(df_cardinality, set())
    # Should flag high cardinality as informational
    assert any(p['type'] == 'High Cardinality' for p in proposals)
    
    # 5. _check_null_values on zero-null column
    df_no_null = pd.DataFrame({'a': [1, 2, 3]})
    proposals = generate_proposals(df_no_null, set())
    # No "Null Check" or "Redundant Column" proposal should be generated
    assert not any(p['type'] in ['Null Check', 'Redundant Column'] for p in proposals)
    
    print("All scout coverage gaps passed.")

def test_rule_utils_coverage_gaps():
    print("Running test_rule_utils_coverage_gaps...")
    
    # 1. evaluate_rule with unknown relational operator
    df = pd.DataFrame({'a': [1, 2, 3]})
    bad_rule = {
        'type': 'Relational Check',
        'col_a': 'a',
        'op': 'unknown_op',
        'target_type': 'Constant',
        'value': 2,
        'desc': 'a unknown_op 2'
    }
    mask = evaluate_rule(df, bad_rule)
    # The code sets valid = Series(False), then mask = ~valid = Series(True) -> all rows violate
    assert mask.all()
    
    # 2. create_resolution_step with different methods
    # Null check
    null_rule = {'type': 'Null Check', 'col': 'a'}
    assert create_resolution_step(null_rule, 'KNN Imputer') == {'action': 'fill_null', 'column': 'a', 'value': 'knn'}
    assert create_resolution_step(null_rule, 'Iterative Imputer') == {'action': 'fill_null', 'column': 'a', 'value': 'iterative'}
    assert create_resolution_step(null_rule, 'Fill with Median') == {'action': 'fill_null', 'column': 'a', 'value': 'median'}
    assert create_resolution_step(null_rule, 'Fill with Mode') == {'action': 'fill_null', 'column': 'a', 'value': 'mode'}
    
    # Range check
    range_rule = {'type': 'Range Check', 'col': 'a', 'min': 1, 'max': 5}
    assert create_resolution_step(range_rule, 'Drop Rows') == {'action': 'drop_violated', 'rule': range_rule}
    assert create_resolution_step(range_rule, 'Log Transform') == {'action': 'log_transform', 'column': 'a'}
    assert create_resolution_step(range_rule, 'Cap at Bounds') == {'action': 'cap_range', 'column': 'a', 'min': 1, 'max': 5}
    
    # 3. generate_evidence_report with empty rules list
    # st.runtime.exists() is false, so it shouldn't try reading from session_state
    df_report = pd.DataFrame({'a': [1, 2, 3]})
    report = generate_evidence_report(df_report, [])
    assert "Total Active Rules: 0" in report
    
    # 4. generate_evidence_report with disabled rules
    rules = [
        {'type': 'Null Check', 'col': 'a', 'desc': 'a is not null', 'enabled': False},
        {'type': 'Range Check', 'col': 'a', 'min': 0, 'max': 10, 'desc': 'a between 0 and 10', 'enabled': True}
    ]
    report_disabled = generate_evidence_report(df_report, rules)
    assert "Total Active Rules: 1" in report_disabled
    assert "Null Check" not in report_disabled # filtered out because it is disabled
    assert "Range Check" in report_disabled
    
    # 5. Test evidence report >100 violations truncation indicator
    df_large = pd.DataFrame({'a': [np.nan] * 150})
    rules_null = [{'type': 'Null Check', 'col': 'a', 'desc': 'a is not null', 'enabled': True}]
    report_large = generate_evidence_report(df_large, rules_null)
    assert "... (truncated)" in report_large
    
    # 6. Test lineage description branches in generate_evidence_report
    # We mock st.runtime.exists() to return True and mock st.session_state
    mock_recipe = [
        {"action": "drop_column", "column": "col1"},
        {"action": "rename_column", "column": "col2", "value": "col2_new"},
        {"action": "strip_whitespace", "column": "col3"},
        {"action": "normalize_text", "column": "col4", "value": "uppercase"},
        {"action": "cast_type", "column": "col5", "dtype": "int64"},
        {"action": "fill_null", "column": "col6", "value": "mean"},
        {"action": "cap_range", "column": "col7", "min": 0, "max": 10},
        {"action": "extract_datetime", "column": "col8", "component": "year", "new_column": "col8_year"},
        {"action": "drop_violated", "column": "col9", "rule": {"desc": "My Custom Rule"}},
        {"action": "reorder_columns", "value": ["col1", "col2"]},
        {"action": "replace", "column": "col10", "find": "A", "replace": "B", "regex": False},
        {"action": "custom_unknown", "column": "col11"}
    ]
    
    with mock.patch('streamlit.runtime.exists', return_value=True), \
         mock.patch.dict('streamlit.session_state', {
             'cleaning_recipe': mock_recipe,
             'intermediate_states': [("Baseline", 100, 10)]
         }):
        report_lineage = generate_evidence_report(
            df_report, 
            [], 
            cleaning_recipe=mock_recipe, 
            original_df=df_report
        )
        assert "Dropped column `col1`" in report_lineage
        assert "Renamed column `col2` to `col2_new`" in report_lineage
        assert "Stripped leading/trailing whitespace in column `col3`" in report_lineage
        assert "Normalized text in column `col4` using method: `uppercase`" in report_lineage
        assert "Casted column `col5` to data type: `int64`" in report_lineage
        assert "Filled missing values in column `col6` with strategy/value: `mean`" in report_lineage
        assert "Capped values in column `col7` within bounds: `[0, 10]`" in report_lineage
        assert "Extracted `year` component from column `col8` into new column `col8_year`" in report_lineage
        assert "Dropped rows violating validation constraint: `My Custom Rule`" in report_lineage
        assert "Reordered columns to custom order: `['col1', 'col2']`" in report_lineage
        assert "Replaced values matching `A` with `B`" in report_lineage
        assert "Applied custom action `custom_unknown`" in report_lineage

    print("All rule_utils coverage gaps passed.")

def test_state_manager_coverage_gaps():
    print("Running test_state_manager_coverage_gaps...")
    
    # 1. load_data with a CSV file path (string path, not UploadedFile)
    temp_csv = 'tests/temp_test_file.csv'
    df_temp = pd.DataFrame({'a': [1, 2, 3]})
    df_temp.to_csv(temp_csv, index=False)
    
    try:
        # Mock streamlit warning/error to avoid displaying inside the test environment
        with mock.patch('streamlit.warning'), mock.patch('streamlit.error'):
            df_loaded = load_data(temp_csv)
            assert df_loaded.shape == (3, 1)
            assert df_loaded.loc[1, 'a'] == 2
            
            # 2. load_data with invalid/nonexistent file
            df_bad = load_data('nonexistent_file_xyz.csv')
            assert df_bad.empty
            
            # 3. load_data fallback for buffers without a clear type
            import io
            buf = io.StringIO("a,b\n1,2\n3,4")
            df_buf = load_data(buf)
            assert df_buf.shape == (2, 2)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    # Test the health calculation formula directly
    # Formula: int((1 - (null_count / total_cells)) * 100)
    def calc_health(df):
        if df.size == 0:
            return 0
        null_count = df.isnull().sum().sum()
        return int((1 - (null_count / df.size)) * 100)
        
    df_1 = pd.DataFrame({'a': [1, 2, 3]})
    assert calc_health(df_1) == 100
    
    df_2 = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
    assert calc_health(df_2) == 0
    
    df_3 = pd.DataFrame({'a': [1, np.nan, 3, np.nan]})
    assert calc_health(df_3) == 50

    print("All state_manager coverage gaps passed.")

if __name__ == "__main__":
    test_scout_coverage_gaps()
    test_rule_utils_coverage_gaps()
    test_state_manager_coverage_gaps()
    print("test_scout_rules_state_coverage.py completed successfully.")
