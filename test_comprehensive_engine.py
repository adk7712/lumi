import pandas as pd
import numpy as np
import pytest
from engine import apply_recipe, generate_pipeline_code
from rule_utils import evaluate_rule

def test_all_engine_actions():
    # Setup a comprehensive test dataframe
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['  Alice  ', 'Bob', 'Charlie', '  David', 'Eva'],
        'age': [25.0, 30.0, 35.0, 150.0, np.nan], # Eva (row 4) is null and will be dropped
        'salary': [50000, 60000, np.nan, 80000, 90000],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York'],
        'hire_date': ['2020-01-01', '2021-06-15', '2022-03-10', 'invalid_date', '2023-11-20'],
        'score': [85, 90, 95, 100, 85],
        'class': ['A', 'B', 'A', 'B', 'A']
    })

    recipe = [
        # 1. Strip Whitespace
        {'action': 'strip_whitespace', 'column': 'name'},
        # 2. Drop Column
        {'action': 'drop_column', 'column': 'class'},
        # 3. Cast Type (numeric, nullable int, datetime)
        {'action': 'cast_type', 'column': 'age', 'dtype': 'int64'}, # will cast to Int64
        {'action': 'cast_type', 'column': 'hire_date', 'dtype': 'datetime64[ns]'},
        # 4. Fill Null (mean strategy)
        {'action': 'fill_null', 'column': 'salary', 'value': 'mean'},
        # 5. Cap Range (bounds)
        {'action': 'cap_range', 'column': 'age', 'min': 18.0, 'max': 65.0}, # caps 150 to 65
        # 6. Replace (normal replacement)
        {'action': 'replace', 'column': 'city', 'find': 'New York', 'replace': 'NYC', 'regex': False},
        # 7. Normalize Text (uppercase)
        {'action': 'normalize_text', 'column': 'city', 'value': 'uppercase'},
        # 8. Log Transform
        {'action': 'log_transform', 'column': 'salary'},
        # 9. Drop Nulls (drop rows with null age if any left)
        {'action': 'drop_nulls', 'column': 'age'}
    ]

    # Apply recipe
    df_clean, messages = apply_recipe(df, recipe)
    
    # Assertions
    assert 'class' not in df_clean.columns, "Column 'class' should be dropped"
    assert df_clean.loc[0, 'name'] == 'Alice', "Whitespace should be stripped"
    assert df_clean.loc[3, 'name'] == 'David', "Whitespace should be stripped"
    assert df_clean.loc[3, 'age'] == 65, "Age should be capped to 65"
    assert pd.api.types.is_integer_dtype(df_clean['age']), "Age should be of integer type"
    assert df_clean.loc[0, 'city'] == 'NYC', "NYC replacement should be applied and uppercased"
    assert df_clean.loc[1, 'city'] == 'LONDON', "London should be uppercased"
    assert pd.api.types.is_datetime64_any_dtype(df_clean['hire_date']), "Hire date should be datetime"
    assert pd.isnull(df_clean.loc[3, 'hire_date']), "Invalid date should be coerced to NaT"
    assert df_clean['salary'].isnull().sum() == 0, "Salary nulls should be filled"
    assert np.isclose(df_clean.loc[2, 'salary'], np.log1p(70000.0)), "Salary null should be filled with mean (70000) and log transformed"
    print("Recipe application test passed.")

    # Test standalone python code generation
    code = generate_pipeline_code(recipe)
    assert "def clean_data(df):" in code, "Should contain clean_data function definition"
    
    # Execute the generated python code to verify it compiles and runs correctly
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    clean_func = exec_globals['clean_data']
    df_generated = clean_func(df.copy())
    
    # Assert generated dataframe matches engine output
    pd.testing.assert_frame_equal(df_clean, df_generated)
    print("Code generation and comparison test passed.")

def test_advanced_imputation():
    # Test KNN Imputation and Iterative Imputation
    df = pd.DataFrame({
        'col1': [1.0, 2.0, np.nan, 4.0, 5.0],
        'col2': [2.0, 4.0, 6.0, np.nan, 10.0]
    })
    
    recipe_knn = [{'action': 'fill_null', 'column': 'col1', 'value': 'knn'}]
    df_knn, msgs = apply_recipe(df, recipe_knn)
    assert df_knn['col1'].isnull().sum() == 0, "KNN Imputation should fill nulls"
    
    recipe_iter = [{'action': 'fill_null', 'column': 'col2', 'value': 'iterative'}]
    df_iter, msgs = apply_recipe(df, recipe_iter)
    assert df_iter['col2'].isnull().sum() == 0, "Iterative Imputation should fill nulls"
    print("Advanced imputation tests passed.")

def test_rule_violations():
    # Setup test dataframe for rules
    df = pd.DataFrame({
        'age': [25, 17, 30, np.nan],
        'parent_age': [50, 35, 20, 40],
        'city': ['NYC', 'London', 'Paris', 'NYC']
    })
    
    # 1. Null Check
    rule_null = {'type': 'Null Check', 'col': 'age'}
    mask_null = evaluate_rule(df, rule_null)
    assert list(mask_null) == [False, False, False, True] # Null row violates
    
    # 2. Range Check
    rule_range = {'type': 'Range Check', 'col': 'age', 'min': 18.0, 'max': 60.0}
    mask_range = evaluate_rule(df, rule_range)
    assert list(mask_range) == [False, True, False, True] # Out of range and null rows violate
    
    # 3. Relational Check
    rule_rel = {'type': 'Relational Check', 'col_a': 'parent_age', 'op': '>', 'col_b': 'age', 'target_type': 'Feature'}
    mask_rel = evaluate_rule(df, rule_rel)
    assert list(mask_rel) == [False, False, True, True] # Row 2 violates (20 > 30 is False), Row 3 violates (nan comparison)
    
    # 4. Custom Expression
    rule_custom = {'type': 'Custom Expression', 'query': "city == 'NYC'"}
    mask_custom = evaluate_rule(df, rule_custom)
    assert list(mask_custom) == [False, True, True, False] # Non-NYC rows violate
    
    print("Rule evaluation tests passed.")

def test_rename_and_reorder():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    
    # 1. Test Rename Action
    recipe_rename = [{'action': 'rename_column', 'column': 'A', 'value': 'Alpha'}]
    df_rename, msgs = apply_recipe(df, recipe_rename)
    assert 'Alpha' in df_rename.columns, "Column A should be renamed to Alpha"
    assert 'A' not in df_rename.columns, "Column A should no longer exist"
    assert list(df_rename.columns) == ['Alpha', 'B', 'C'], "Column names list should update correctly"
    
    # Verify rename code gen
    code_rename = generate_pipeline_code(recipe_rename)
    assert "df = df.rename(columns={'A': 'Alpha'})" in code_rename
    
    # 2. Test Reorder Action
    recipe_reorder = [{'action': 'reorder_columns', 'value': ['C', 'B', 'Alpha']}]
    df_reorder, msgs = apply_recipe(df_rename, recipe_reorder)
    assert list(df_reorder.columns) == ['C', 'B', 'Alpha'], "Columns should be in exact requested order"
    
    # Test reorder robustness (handling missing or new columns)
    recipe_reorder_robust = [{'action': 'reorder_columns', 'value': ['C']}]
    df_reorder_robust, msgs = apply_recipe(df_rename, recipe_reorder_robust)
    assert list(df_reorder_robust.columns) == ['C', 'Alpha', 'B'], "Omitted columns should be safely appended to the end"
    
    # Verify reorder code gen
    code_reorder = generate_pipeline_code(recipe_reorder)
    assert "df = df[['C', 'B', 'Alpha']]" in code_reorder
    
    # Run generated code on df
    recipe_combined = recipe_rename + recipe_reorder
    code_combined = generate_pipeline_code(recipe_combined)
    exec_globals = {'pd': pd, 'np': np}
    exec(code_combined, exec_globals)
    clean_func = exec_globals['clean_data']
    df_generated = clean_func(df.copy())
    
    pd.testing.assert_frame_equal(df_reorder, df_generated)
    print("Rename and Reorder engine tests passed.")

if __name__ == "__main__":
    try:
        test_all_engine_actions()
        test_advanced_imputation()
        test_rule_violations()
        test_rename_and_reorder()
        print("ALL COMPREHENSIVE ENGINE TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
