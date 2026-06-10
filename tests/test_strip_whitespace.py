import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine import apply_recipe, generate_pipeline_code

def test_strip_whitespace():
    # Setup test data
    data = {
        'name': ['  Alice  ', 'Bob', '  Charlie', 'David  '],
        'city': [' New York ', ' London ', 'Paris', ' Tokyo '],
        'age': [25, 30, 35, 40]
    }
    df = pd.DataFrame(data)

    # 1. Test single column strip
    recipe_single = [{'action': 'strip_whitespace', 'column': 'name'}]
    df_clean_single, messages = apply_recipe(df, recipe_single)
    
    assert all(df_clean_single['name'] == ['Alice', 'Bob', 'Charlie', 'David'])
    assert all(df_clean_single['city'] == [' New York ', ' London ', 'Paris', ' Tokyo ']) # Should be unchanged
    print("Single column strip test passed.")

    # 2. Test All columns strip
    recipe_all = [{'action': 'strip_whitespace', 'column': 'All'}]
    df_clean_all, messages = apply_recipe(df, recipe_all)
    
    assert all(df_clean_all['name'] == ['Alice', 'Bob', 'Charlie', 'David'])
    assert all(df_clean_all['city'] == ['New York', 'London', 'Paris', 'Tokyo'])
    print("All columns strip test passed.")

    # 3. Test code generation
    code = generate_pipeline_code(recipe_all)
    # Execute the generated code
    exec_globals = {'pd': pd, 'np': np}
    exec(code, exec_globals)
    clean_data_func = exec_globals['clean_data']
    df_generated = clean_data_func(df)
    
    assert all(df_generated['name'] == ['Alice', 'Bob', 'Charlie', 'David'])
    assert all(df_generated['city'] == ['New York', 'London', 'Paris', 'Tokyo'])
    print("Code generation test passed.")

if __name__ == "__main__":
    try:
        test_strip_whitespace()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
