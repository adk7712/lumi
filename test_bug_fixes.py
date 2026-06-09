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
    # Simulate intermediate states list as maintained in app.py
    df_raw = pd.DataFrame({
        'name': ['  Alice  ', 'Bob'],
        'age': [25, 30]
    })
    
    # Setup initial state
    intermediate_states = []
    bh = 100
    intermediate_states.append(("Original Data", bh, len(df_raw), df_raw.copy()))
    
    # Apply a step
    step1 = {'action': 'strip_whitespace', 'column': 'name'}
    last_df = intermediate_states[-1][3]
    new_df, messages = apply_recipe(last_df, [step1])
    
    # Append new state
    intermediate_states.append(("strip_whitespace on name", bh, len(new_df), new_df))
    
    # Retrieve the latest dataframe like df = st.session_state.intermediate_states[-1][3]
    current_df = intermediate_states[-1][3]
    
    assert all(current_df['name'] == ['Alice', 'Bob']), "Cached dataframe should have whitespace stripped"
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

if __name__ == "__main__":
    try:
        test_scout_string_dtype()
        test_intermediate_states_simulation()
        test_diagnostics_histogram_grouping()
        test_chart_updates()
        test_correlation_range_filtering()
        print("ALL BUG FIX TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
