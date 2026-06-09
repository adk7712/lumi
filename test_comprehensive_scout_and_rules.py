import pandas as pd
import numpy as np
import pytest
from scout import generate_proposals
from rule_utils import evaluate_rule
from engine import apply_recipe, generate_pipeline_code

def test_scout_heuristics():
    # Setup a mock dataframe tailored to trigger each scout heuristic
    df = pd.DataFrame({
        # 95% nulls -> triggers Redundant Column (NULL_DROP_PCT = 90.0)
        'mostly_null': [np.nan] * 19 + [1.0],
        # 20% nulls -> triggers Null Check
        'some_null': [1.0, 2.0, 3.0, 4.0, np.nan] * 4,
        # Zero variance -> triggers Constant Value
        'constant_col': [42] * 20,
        # IQR outliers -> triggers Range Check (l, u bounds)
        # Normal IQR is small, then 1000 is an outlier
        'outliers_col': [10.0, 11.0, 10.5, 10.2, 10.8, 11.1, 10.9, 10.4, 10.6, 1000.0] * 2,
        # Highly skewed -> triggers Distribution Warning (skewness > 2.0)
        'skewed_col': [1.0, 1.1, 1.2, 1.3, 1.0, 1.2, 1.1, 1.3, 1.2, 100.0] * 2,
        # 80% numeric values hidden in text -> triggers Type Cast (float64)
        'mixed_col': ['1.5', '2.6', '3.7', '4.8', 'not_a_number'] * 4,
        # Whitespaces present -> triggers Formatting Issue
        'whitespace_col': ['  apple  ', 'banana', 'orange  ', 'grape', '  melon'] * 4
    })

    proposals = generate_proposals(df, set())

    # Categorize proposals by type
    redundant_props = [p for p in proposals if p['type'] == 'Redundant Column']
    null_props = [p for p in proposals if p['type'] == 'Null Check']
    constant_props = [p for p in proposals if p['type'] == 'Constant Value']
    range_props = [p for p in proposals if p['type'] == 'Range Check']
    skew_props = [p for p in proposals if p['type'] == 'Distribution Warning']
    type_props = [p for p in proposals if p['type'] == 'Type Cast']
    format_props = [p for p in proposals if p['type'] == 'Formatting Issue']

    # Assertions
    assert len(redundant_props) == 1 and redundant_props[0]['column'] == 'mostly_null'
    assert len(null_props) == 1 and null_props[0]['column'] == 'some_null'
    assert len(constant_props) == 2 and any(p['column'] == 'constant_col' for p in constant_props)
    assert len(range_props) == 2 and any(p['column'] == 'outliers_col' for p in range_props)
    assert len(skew_props) == 2 and any(p['column'] == 'skewed_col' for p in skew_props)
    assert len(type_props) == 1 and type_props[0]['column'] == 'mixed_col'
    assert len(format_props) == 1 and format_props[0]['column'] == 'whitespace_col'

    # Test scanned_columns list exclusion
    scanned = {'mostly_null', 'constant_col'}
    filtered_proposals = generate_proposals(df, scanned)
    assert not any(p['column'] in scanned for p in filtered_proposals)
    print("Scout heuristics comprehensive tests passed.")

def test_rule_evaluator_edge_cases():
    df = pd.DataFrame({
        'A': [10, 20, 30, np.nan],
        'B': [5, 25, 15, 35],
        'C': ['apple', 'banana', 'orange', 'apple']
    })

    # Test Relational Operators
    operators = [">", "<", "==", "!=", ">=", "<="]
    for op in operators:
        rule = {'type': 'Relational Check', 'col_a': 'A', 'op': op, 'col_b': 'B', 'target_type': 'Feature'}
        mask = evaluate_rule(df, rule)
        assert len(mask) == len(df)
        assert mask.dtype == bool

    # Test Relational Check with constant comparison value
    rule_const = {'type': 'Relational Check', 'col_a': 'A', 'op': '>=', 'value': 20, 'target_type': 'Value'}
    mask_const = evaluate_rule(df, rule_const)
    # A >= 20 is [False, True, True, False] -> violators are [True, False, False, True]
    assert list(mask_const) == [True, False, False, True]

    # Test Custom Expression syntax error handling
    rule_bad_query = {'type': 'Custom Expression', 'query': "A >>> 50"}
    with pytest.raises(ValueError) as excinfo:
        evaluate_rule(df, rule_bad_query)
    assert "evaluating rule" in str(excinfo.value)

    # Test missing column exception
    rule_missing_col = {'type': 'Null Check', 'col': 'non_existent'}
    with pytest.raises(ValueError) as excinfo:
        evaluate_rule(df, rule_missing_col)
    assert "evaluating rule" in str(excinfo.value)
    
    # Test Informational rules returns neutral all-False mask
    rule_info = {'type': 'Informational', 'desc': 'Test note'}
    mask_info = evaluate_rule(df, rule_info)
    assert not mask_info.any()
    print("Rule evaluator comprehensive tests passed.")

def test_engine_warnings_and_errors():
    df = pd.DataFrame({
        'A': [1.0, 2.0, np.nan],
        'B': ['apple', 'banana', 'orange']
    })

    # 1. Unknown action warning
    df_res, msgs = apply_recipe(df, [{'action': 'unsupported_magic'}])
    assert any("Unknown action" in m for m in msgs)

    # 2. Non-existent column drop warning
    df_res, msgs = apply_recipe(df, [{'action': 'drop_column', 'column': 'non_existent'}])
    assert any("not found" in m for m in msgs)

    # 3. Non-existent column drop_nulls warning
    df_res, msgs = apply_recipe(df, [{'action': 'drop_nulls', 'column': 'non_existent'}])
    assert any("not found" in m for m in msgs)

    # 4. Advanced Imputation error on non-numeric column
    df_res, msgs = apply_recipe(df, [{'action': 'fill_null', 'column': 'B', 'value': 'knn'}])
    assert any("requires a numeric column" in m for m in msgs)

    # 5. Non-existent column cap_range warning
    df_res, msgs = apply_recipe(df, [{'action': 'cap_range', 'column': 'non_existent', 'min': 0, 'max': 1}])
    assert any("not found" in m for m in msgs)

    # 6. Non-existent column cast_type warning
    df_res, msgs = apply_recipe(df, [{'action': 'cast_type', 'column': 'non_existent', 'dtype': 'int64'}])
    assert any("not found" in m for m in msgs)

    # 7. Non-existent column rename warning
    df_res, msgs = apply_recipe(df, [{'action': 'rename_column', 'column': 'non_existent', 'value': 'new_name'}])
    assert any("not found" in m for m in msgs)

    # 8. Reorder invalid format error
    df_res, msgs = apply_recipe(df, [{'action': 'reorder_columns', 'value': 'not_a_list'}])
    assert any("requires a list of column names" in m for m in msgs)

    # 9. Log transform on non-numeric column error
    df_res, msgs = apply_recipe(df, [{'action': 'log_transform', 'column': 'B'}])
    assert any("requires a numeric column" in m for m in msgs)
    print("Engine error handling tests passed.")

def test_comprehensive_code_generation():
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['  Alice  ', 'Bob', 'Charlie', 'David', 'Eva'],
        'age': [25.0, 30.0, np.nan, 40.0, 50.0],
        'salary': [50000, 60000, 70000, np.nan, 90000],
        'city': ['NYC', 'London', 'Paris', 'NYC', 'Tokyo']
    })

    recipe = [
        # All columns replace
        {'action': 'replace', 'column': 'All', 'find': 'NYC', 'replace': 'New York', 'regex': False},
        # Mode fill null
        {'action': 'fill_null', 'column': 'age', 'value': 'mode'},
        # Constant value fill null
        {'action': 'fill_null', 'column': 'salary', 'value': -1.0},
        # Drop violated using Relational Check constant comparison
        {'action': 'drop_violated', 'rule': {'type': 'Relational Check', 'col_a': 'id', 'op': '>=', 'value': 2, 'target_type': 'Value'}},
        # Text normalization remove punctuation
        {'action': 'normalize_text', 'column': 'name', 'value': 'remove_punctuation'},
        # Text normalization lowercase
        {'action': 'normalize_text', 'column': 'All', 'value': 'lowercase'},
        # Text normalization fuzzy dedupe
        {'action': 'normalize_text', 'column': 'city', 'value': 'fuzzy_dedupe'}
    ]

    df_clean, msgs = apply_recipe(df, recipe)
    code = generate_pipeline_code(recipe)

    # Compile and execute the generated code
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    clean_func = exec_globals['clean_data']
    df_generated = clean_func(df.copy())

    pd.testing.assert_frame_equal(df_clean, df_generated)
    print("Comprehensive code generation compilation and equality tests passed.")

if __name__ == "__main__":
    test_scout_heuristics()
    test_rule_evaluator_edge_cases()
    test_engine_warnings_and_errors()
    test_comprehensive_code_generation()
    print("ALL COMPREHENSIVE SCOUT, RULE, AND ENGINE TESTS PASSED")
