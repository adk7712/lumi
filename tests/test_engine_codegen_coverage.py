import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine import apply_recipe, generate_pipeline_code, generate_notebook_code
from engine_ops import TRANSFORM_REGISTRY

def test_engine_ops_coverage_gaps():
    print("Running test_engine_ops_coverage_gaps...")
    
    # 1. _handle_fill_null gaps: median strategy, empty mode result, missing column, fill constant
    df_nulls = pd.DataFrame({
        'a': [1.0, 2.0, np.nan, 4.0, 2.0],
        'b': [np.nan, np.nan, np.nan, np.nan, np.nan], # Mode will be empty
        'c': [1, 2, 3, 4, 5]
    })
    
    # Median
    recipe_median = [{'action': 'fill_null', 'column': 'a', 'value': 'median'}]
    df_res, msgs = apply_recipe(df_nulls, recipe_median)
    assert df_res.loc[2, 'a'] == 2.0 # median of [1, 2, 4, 2] is 2.0
    
    # Mode on empty mode result column
    recipe_mode_empty = [{'action': 'fill_null', 'column': 'b', 'value': 'mode'}]
    df_res, msgs = apply_recipe(df_nulls, recipe_mode_empty)
    # Since b is entirely null, its mode is empty. Should fall back to 0 or remain unchanged.
    assert df_res['b'].isnull().all()
    
    # Missing column fill_null
    recipe_missing_col = [{'action': 'fill_null', 'column': 'nonexistent', 'value': 'mean'}]
    df_res, msgs = apply_recipe(df_nulls, recipe_missing_col)
    assert len(msgs) > 0 and any("Warning" in m for m in msgs)
    
    # Fill null with custom constant value
    recipe_const = [{'action': 'fill_null', 'column': 'a', 'value': 'custom_val_99'}]
    df_res, msgs = apply_recipe(df_nulls, recipe_const)
    assert df_res.loc[2, 'a'] == 'custom_val_99'

    # 2. _handle_cast_type gaps: cast to string/object, cast exception branch
    df_cast = pd.DataFrame({'a': ['1', 'two', '3']})
    # Object/string
    recipe_string = [{'action': 'cast_type', 'column': 'a', 'dtype': 'string'}]
    df_res, msgs = apply_recipe(df_cast, recipe_string)
    assert pd.api.types.is_string_dtype(df_res['a'])
    
    # Cast exception
    recipe_bad_cast = [{'action': 'cast_type', 'column': 'a', 'dtype': 'float_nonexistent'}]
    df_res, msgs = apply_recipe(df_cast, recipe_bad_cast)
    assert any("Error" in m or "Failed" in m for m in msgs) # should report cast error message in engine status

    # 3. _handle_drop_violated gaps: informational rule (no-op), exception catch branch
    df_violations = pd.DataFrame({'a': [1, 2, 3]})
    info_rule = {'type': 'Informational', 'col': 'a', 'desc': 'Just info'}
    recipe_info = [{'action': 'drop_violated', 'rule': info_rule}]
    df_res, msgs = apply_recipe(df_violations, recipe_info)
    assert len(df_res) == 3 # no rows dropped
    
    # Exception branch (rule with missing keys to trigger exception)
    bad_rule = {'type': 'Range Check', 'desc': 'Bad rule'} # missing col/min/max
    recipe_bad_rule = [{'action': 'drop_violated', 'rule': bad_rule}]
    df_res, msgs = apply_recipe(df_violations, recipe_bad_rule)
    assert any("Error" in m for m in msgs)

    # 4. _handle_replace gaps: missing column, regex=True
    df_replace = pd.DataFrame({'a': ['apple', 'banana', 'apricot']})
    recipe_rep_regex = [{'action': 'replace', 'column': 'a', 'find': '^ap.*', 'replace': 'X', 'regex': True}]
    df_res, msgs = apply_recipe(df_replace, recipe_rep_regex)
    assert df_res.loc[0, 'a'] == 'X'
    assert df_res.loc[2, 'a'] == 'X'
    assert df_res.loc[1, 'a'] == 'banana'
    
    recipe_rep_missing = [{'action': 'replace', 'column': 'nonexistent', 'find': 'a', 'replace': 'b', 'regex': False}]
    df_res, msgs = apply_recipe(df_replace, recipe_rep_missing)
    assert any("Warning" in m for m in msgs)

    # 5. _handle_strip_whitespace gaps: missing column
    df_strip = pd.DataFrame({'a': [' a ']})
    recipe_strip_missing = [{'action': 'strip_whitespace', 'column': 'nonexistent'}]
    df_res, msgs = apply_recipe(df_strip, recipe_strip_missing)
    assert any("Warning" in m for m in msgs)

    # 6. _handle_normalize_text gaps: titlecase, missing column, unknown method
    df_norm = pd.DataFrame({'a': ['hello world']})
    recipe_title = [{'action': 'normalize_text', 'column': 'a', 'value': 'titlecase'}]
    df_res, msgs = apply_recipe(df_norm, recipe_title)
    assert df_res.loc[0, 'a'] == 'Hello World'
    
    recipe_norm_missing = [{'action': 'normalize_text', 'column': 'nonexistent', 'value': 'lowercase'}]
    df_res, msgs = apply_recipe(df_norm, recipe_norm_missing)
    assert any("Warning" in m for m in msgs)
    
    recipe_norm_unknown = [{'action': 'normalize_text', 'column': 'a', 'value': 'unknown_method'}]
    df_res, msgs = apply_recipe(df_norm, recipe_norm_unknown)
    assert df_res.loc[0, 'a'] == 'hello world' # unchanged

    # 7. _handle_log_transform gaps: missing column
    df_log = pd.DataFrame({'a': [1.0]})
    recipe_log_missing = [{'action': 'log_transform', 'column': 'nonexistent'}]
    df_res, msgs = apply_recipe(df_log, recipe_log_missing)
    assert any("Warning" in m for m in msgs)

    # 8. _handle_extract_datetime gaps: missing column, missing new_column, unknown component, already datetime
    df_dt = pd.DataFrame({'a': ['2023-01-01 12:00:00']})
    df_dt_parsed = df_dt.copy()
    df_dt_parsed['a'] = pd.to_datetime(df_dt_parsed['a'])
    
    # Already datetime
    recipe_dt_parsed = [{'action': 'extract_datetime', 'column': 'a', 'component': 'year', 'new_column': 'a_year'}]
    df_res, msgs = apply_recipe(df_dt_parsed, recipe_dt_parsed)
    assert df_res.loc[0, 'a_year'] == 2023
    
    # Missing column
    recipe_dt_missing = [{'action': 'extract_datetime', 'column': 'nonexistent', 'component': 'year', 'new_column': 'year'}]
    df_res, msgs = apply_recipe(df_dt, recipe_dt_missing)
    assert any("Warning" in m for m in msgs)
    
    # Missing new_column
    recipe_dt_nonew = [{'action': 'extract_datetime', 'column': 'a', 'component': 'year'}] # missing new_column
    df_res, msgs = apply_recipe(df_dt, recipe_dt_nonew)
    assert any("Error" in m for m in msgs)
    
    # Unknown component
    recipe_dt_badcomp = [{'action': 'extract_datetime', 'column': 'a', 'component': 'millisecond', 'new_column': 'ms'}]
    df_res, msgs = apply_recipe(df_dt, recipe_dt_badcomp)
    assert any("Error" in m for m in msgs)

    # 9. _handle_drop_empty_columns gaps: no empty columns message
    df_no_empty = pd.DataFrame({'a': [1, 2, 3]})
    recipe_empty_cols = [{'action': 'drop_empty_columns'}]
    df_res, msgs = apply_recipe(df_no_empty, recipe_empty_cols)
    assert any("No empty columns found" in m for m in msgs)

    # 10. _handle_normalize_column_names gaps: uppercase, lowercase, remove_spaces, unknown, empty string fallback
    df_names = pd.DataFrame({' First Name ': [1], '': [2]})
    
    # Uppercase
    recipe_names_upper = [{'action': 'normalize_column_names', 'value': 'uppercase'}]
    df_res, msgs = apply_recipe(df_names, recipe_names_upper)
    assert ' FIRST NAME ' in df_res.columns
    
    # Lowercase
    recipe_names_lower = [{'action': 'normalize_column_names', 'value': 'lowercase'}]
    df_res, msgs = apply_recipe(df_names, recipe_names_lower)
    assert ' first name ' in df_res.columns
    
    # Remove Spaces
    recipe_names_nospaces = [{'action': 'normalize_column_names', 'value': 'remove_spaces'}]
    df_res, msgs = apply_recipe(df_names, recipe_names_nospaces)
    assert 'FirstName' in df_res.columns
    
    # Unknown method fallback
    recipe_names_unknown = [{'action': 'normalize_column_names', 'value': 'unknown'}]
    df_res, msgs = apply_recipe(df_names, recipe_names_unknown)
    assert ' First Name ' in df_res.columns
    
    # Empty string column name fallback (when method is snake_case and makes name empty)
    # The empty column name '' will become 'column_'
    recipe_names_snake = [{'action': 'normalize_column_names', 'value': 'snake_case'}]
    df_res, msgs = apply_recipe(df_names, recipe_names_snake)
    assert 'first_name' in df_res.columns
    assert 'column_' in df_res.columns

    print("All engine_ops coverage gaps passed.")

def test_codegen_coverage_gaps():
    print("Running test_codegen_coverage_gaps...")
    
    # 1. Empty recipe
    code = generate_pipeline_code([])
    assert "No cleaning steps applied" in code
    
    # 2. cast_type with string/object codegen output
    recipe_cast_str = [{'action': 'cast_type', 'column': 'a', 'dtype': 'string'}]
    code = generate_pipeline_code(recipe_cast_str)
    assert "astype('string')" in code or 'astype("string")' in code
    
    # 3. drop_violated with Informational rule
    info_rule = {'type': 'Informational', 'col': 'a', 'desc': 'Info only'}
    recipe_info = [{'action': 'drop_violated', 'rule': info_rule}]
    code = generate_pipeline_code(recipe_info)
    # Should not produce query-based drops or rules check since Informational rules are skipped
    assert "query" not in code and "drop" not in code
    
    # 4. drop_violated with Custom Expression
    custom_rule = {'type': 'Custom Expression', 'query': 'a > 5', 'desc': 'a is greater than 5'}
    recipe_custom = [{'action': 'drop_violated', 'rule': custom_rule}]
    code = generate_pipeline_code(recipe_custom)
    assert "df.query(\"a > 5\")" in code or "df.query('a > 5')" in code
    
    # 5. normalize_text titlecase codegen
    recipe_title = [{'action': 'normalize_text', 'column': 'a', 'value': 'titlecase'}]
    code = generate_pipeline_code(recipe_title)
    assert "str.title()" in code
    
    # 6. generate_notebook_code with empty recipe and empty rules
    notebook_json = generate_notebook_code([], [])
    assert "clean_data" in notebook_json
    assert "validate_data" in notebook_json
    
    # 7. validate codegen with unknown rule type fallback
    bad_rule = {'type': 'Unknown Rule Type', 'col': 'a', 'desc': 'Bad'}
    code_val = generate_pipeline_code([], [bad_rule])
    assert "Unknown Rule Type" in code_val # check description/metadata in comments
    
    # Compile and verify execution
    exec_globals = {'pd': pd, 'np': np}
    exec(code_val, exec_globals)
    validate_func = exec_globals['validate_data']
    df_test = pd.DataFrame({'a': [1, 2, 3]})
    res = validate_func(df_test)
    assert isinstance(res, dict)
    assert len(res) == 0
    
    print("All codegen coverage gaps passed.")

if __name__ == "__main__":
    test_engine_ops_coverage_gaps()
    test_codegen_coverage_gaps()
    print("test_engine_codegen_coverage.py completed successfully.")
