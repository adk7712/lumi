import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine_ops import _handle_normalize_text, _handle_fill_null
from scout import generate_proposals
from state_manager import downcast_dtypes

def test_fuzzy_dedupe_cap():
    print("Running test_fuzzy_dedupe_cap...")
    # Create a series with 1050 unique values (above the 1000 threshold)
    unique_strings = [f"value_{i}" for i in range(1050)]
    df = pd.DataFrame({"col": unique_strings})
    
    step = {
        "action": "normalize_text",
        "column": "col",
        "value": "fuzzy_dedupe"
    }
    
    # Run normalize_text
    df_new, messages = _handle_normalize_text(df, step)
    
    # Verify skipped and warning generated
    assert len(messages) == 1, f"Expected 1 warning message, got {len(messages)}"
    assert "skipped" in messages[0]
    assert "too many unique values" in messages[0]
    assert (df_new["col"] == df["col"]).all(), "Data should remain untouched when skipped"
    print("test_fuzzy_dedupe_cap passed.")

def test_imputer_column_restriction():
    print("Running test_imputer_column_restriction...")
    # Create a DataFrame with 15 numeric columns
    data = {}
    for i in range(15):
        # 100 random rows
        data[f"col_{i}"] = np.random.randn(100)
    
    df = pd.DataFrame(data)
    # Introduce NaN in target column col_0
    df.loc[10:20, "col_0"] = np.nan
    
    # Target column col_0, fill_value is knn
    step = {
        "action": "fill_null",
        "column": "col_0",
        "value": "knn"
    }
    
    # Verify it handles the imputer correctly without error
    df_new, messages = _handle_fill_null(df, step)
    assert len(messages) == 0, f"Expected no error messages, got {messages}"
    assert not df_new["col_0"].isnull().any(), "Target column should be imputed"
    print("test_imputer_column_restriction passed.")

def test_scout_downsampling():
    print("Running test_scout_downsampling...")
    # Create a large DataFrame with 12,000 rows
    df = pd.DataFrame({
        "col1": [1.0] * 12000,
        "col2": ["abc"] * 12000
    })
    
    # scanned_columns is empty
    proposals = generate_proposals(df, set())
    # Should complete instantly due to downsampling
    # Proposals should flag constant values for both columns
    types = [p["type"] for p in proposals]
    assert "Constant Value" in types, "Should identify constant values"
    print("test_scout_downsampling passed.")

def test_dtype_downcasting():
    print("Running test_dtype_downcasting...")
    # Create DataFrame with decimal-heavy float64 and large int64
    df = pd.DataFrame({
        "dec_float": [1.123456789012345, 2.987654321098765] * 10,
        "large_int": [100, 200] * 10
    })
    
    # Check original dtypes
    assert df["dec_float"].dtype == np.float64
    assert df["large_int"].dtype == np.int64
    
    # Run downcast
    df_downcast = downcast_dtypes(df)
    
    # Verify dtypes are optimized
    assert df_downcast["dec_float"].dtype == np.float32
    # downcast='integer' will downcast to int8 because values 100/200 fit in int8 or int16
    assert df_downcast["large_int"].dtype in [np.int8, np.int16, np.int32]
    
    # Spot check float32 values precision is within float32 bounds (approx 7 digits of precision)
    # 1.123456789012345 -> 1.1234568
    np.testing.assert_allclose(df_downcast["dec_float"].iloc[0], 1.1234567, rtol=1e-6)
    print("test_dtype_downcasting passed.")

if __name__ == "__main__":
    test_fuzzy_dedupe_cap()
    test_imputer_column_restriction()
    test_scout_downsampling()
    test_dtype_downcasting()
    print("ALL OPTIMIZATION TESTS PASSED")
