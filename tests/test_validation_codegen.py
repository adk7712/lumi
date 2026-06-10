import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine import generate_pipeline_code

def test_no_rules_codegen():
    recipe = [{'action': 'strip_whitespace', 'column': 'name'}]
    code = generate_pipeline_code(recipe, rules=None)
    
    assert "def validate_data(df):" in code
    assert "No active validation rules defined" in code
    
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    validate_func = exec_globals['validate_data']
    
    df = pd.DataFrame({'name': ['Alice', 'Bob']})
    violations = validate_func(df)
    assert violations == {}
    print("Test no rules codegen passed.")

def test_active_rules_validation_behavior():
    recipe = [
        {'action': 'drop_nulls', 'column': 'age'},
        {'action': 'cap_range', 'column': 'score', 'min': 0.0, 'max': 100.0},
        {'action': 'drop_violated', 'rule': {
            'type': 'Relational Check', 'col_a': 'parent_age', 'op': '>', 'col_b': 'age', 'target_type': 'Feature', 'desc': 'parent_age > age'
        }}
    ]
    
    rules = [
        {'type': 'Null Check', 'col': 'age', 'desc': 'age is NOT NULL', 'enabled': True},
        {'type': 'Range Check', 'col': 'score', 'min': 0.0, 'max': 100.0, 'desc': 'score in [0.0, 100.0]', 'enabled': True},
        {'type': 'Relational Check', 'col_a': 'parent_age', 'op': '>', 'col_b': 'age', 'target_type': 'Feature', 'desc': 'parent_age > age', 'enabled': True},
        {'type': 'Custom Expression', 'query': "city == 'NYC'", 'desc': "Matches: city == 'NYC'", 'enabled': True},
        # This informational rule should be ignored in validation checks
        {'type': 'Informational', 'desc': 'Check city names', 'enabled': True},
        # This rule is disabled, so it should be ignored in validation checks
        {'type': 'Null Check', 'col': 'score', 'desc': 'score is NOT NULL', 'enabled': False}
    ]
    
    code = generate_pipeline_code(recipe, rules)
    
    # Verify both functions exist in generated code
    assert "def clean_data(df):" in code
    assert "def validate_data(df):" in code
    
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    clean_func = exec_globals['clean_data']
    validate_func = exec_globals['validate_data']
    
    # 1. Create dirty dataframe that violates every rule
    df_dirty = pd.DataFrame({
        'age': [25.0, np.nan, 30.0, 40.0],              # Row 1 has null age (Null Check violation)
        'score': [85.0, 95.0, 150.0, 50.0],             # Row 2 has 150.0 score (Range Check violation)
        'parent_age': [50.0, 40.0, 20.0, 60.0],         # Row 2 has parent_age 20 <= age 30 (Relational violation)
        'city': ['NYC', 'NYC', 'NYC', 'London']         # Row 3 has city 'London' (Custom Expression violation)
    })
    
    violations_dirty = validate_func(df_dirty)
    
    # Verify expected violations are caught and counts match
    assert 'age is NOT NULL' in violations_dirty
    assert violations_dirty['age is NOT NULL'] == 1
    
    assert 'score in [0.0, 100.0]' in violations_dirty
    assert violations_dirty['score in [0.0, 100.0]'] == 1 # 150.0 is out of bounds
    
    assert 'parent_age > age' in violations_dirty
    # Row 1: parent_age 40, age is NaN -> comparison fails/violates. Row 2: 20 > 30 is False -> violates. Total 2.
    assert violations_dirty['parent_age > age'] == 2
    
    assert "Matches: city == 'NYC'" in violations_dirty
    assert violations_dirty["Matches: city == 'NYC'"] == 1 # 'London' violates
    
    # Ensure disabled or informational rules are not present in violations
    assert 'Check city names' not in violations_dirty
    assert 'score is NOT NULL' not in violations_dirty
    
    # 2. Run cleaning transformations and check that all violations are resolved
    df_clean = clean_func(df_dirty.copy())
    violations_clean = validate_func(df_clean)
    
    # The Custom Expression city == 'NYC' is still violated on remaining rows if they have London.
    # Let's filter df_clean to see what rows are left.
    # Row 0: age=25, score=85, parent_age=50, city=NYC. (Clean)
    # Row 1: age=NaN -> dropped by drop_nulls
    # Row 2: age=30, score=100 (capped from 150), parent_age=20 -> dropped by relational drop_violated (parent_age > age)
    # Row 3: age=40, score=50, parent_age=60, city=London -> kept, but city is London, so custom query rule fails.
    # Let's verify city == 'NYC' violation count is 1 in the cleaned data since Row 3 is kept.
    assert "Matches: city == 'NYC'" in violations_clean
    assert violations_clean["Matches: city == 'NYC'"] == 1
    
    # But other violations should be resolved (0 counts, so absent from dictionary)
    assert 'age is NOT NULL' not in violations_clean
    assert 'score in [0.0, 100.0]' not in violations_clean
    assert 'parent_age > age' not in violations_clean
    
    print("Test active rules validation behavior passed.")

def test_validation_exception_safety():
    # Test that rules referencing missing columns or having bad custom query expressions
    # return a helpful error string in the violation report instead of crashing the pipeline.
    rules = [
        {'type': 'Null Check', 'col': 'missing_col', 'desc': 'missing_col is NOT NULL', 'enabled': True},
        {'type': 'Custom Expression', 'query': "invalid syntax ++ --", 'desc': 'Bad syntax query', 'enabled': True}
    ]
    
    code = generate_pipeline_code([], rules)
    
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    validate_func = exec_globals['validate_data']
    
    df = pd.DataFrame({'name': ['Alice', 'Bob']})
    violations = validate_func(df)
    
    assert 'missing_col is NOT NULL' in violations
    assert violations['missing_col is NOT NULL'].startswith("Error")
    
    assert 'Bad syntax query' in violations
    assert violations['Bad syntax query'].startswith("Error")
    
    print("Test validation exception safety passed.")

if __name__ == "__main__":
    test_no_rules_codegen()
    test_active_rules_validation_behavior()
    test_validation_exception_safety()
    print("ALL VALIDATION CODEGEN TESTS PASSED")
