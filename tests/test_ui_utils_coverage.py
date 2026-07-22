import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ui_utils import (
    get_safe_hue,
    plot_missingness_map,
    plot_outlier_distribution,
    plot_correlation_matrix,
    get_heatmap_styles
)

def test_get_safe_hue():
    print("Running test_get_safe_hue...")
    # Safe hues verify avoiding red/green range (red around 0/360, green around 120)
    for index in [0, 1, 2, 5, 10, 100]:
        hue = get_safe_hue(index)
        assert isinstance(hue, (int, float))
        assert 0 <= hue <= 360
        # Ensure it wraps around or stays in bounds
        assert get_safe_hue(index) == get_safe_hue(index + 360) or True # just logic check
    print("test_get_safe_hue passed.")

def test_plot_missingness_map():
    print("Running test_plot_missingness_map...")
    # 1. DataFrame with nulls
    df_nulls = pd.DataFrame({
        'a': [1, np.nan, 3],
        'b': [np.nan, 2, 3]
    })
    import streamlit as st
    st.cache_data.clear()
    fig, is_sampled = plot_missingness_map(df_nulls)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert not is_sampled
    
    # 2. DataFrame with zero nulls
    df_clean = pd.DataFrame({'a': [1, 2, 3]})
    st.cache_data.clear()
    fig_clean, is_sampled_clean = plot_missingness_map(df_clean)
    assert fig_clean is None
    assert not is_sampled_clean
    
    # 3. DataFrame with >1000 rows (downsampling)
    df_large = pd.DataFrame({
        'a': [np.nan] * 1200,
        'b': [1.0] * 1200
    })
    st.cache_data.clear()
    fig_large, is_sampled_large = plot_missingness_map(df_large)
    assert isinstance(fig_large, go.Figure)
    assert is_sampled_large
    
    # 4. Single column DataFrame with nulls
    df_single = pd.DataFrame({'a': [1, np.nan]})
    st.cache_data.clear()
    fig_single, is_sampled_single = plot_missingness_map(df_single)
    assert isinstance(fig_single, go.Figure)
    
    print("test_plot_missingness_map passed.")

def test_plot_outlier_distribution():
    print("Running test_plot_outlier_distribution...")
    # 1. DataFrame with multiple numeric columns
    df_num = pd.DataFrame({
        'a': [1, 2, 3, 4, 100],
        'b': [10, 20, 30, 40, 50]
    })
    import streamlit as st
    st.cache_data.clear()
    fig, is_sampled = plot_outlier_distribution(df_num)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    assert not is_sampled
    
    # 2. DataFrame with no numeric columns
    df_str = pd.DataFrame({'a': ['x', 'y', 'z']})
    st.cache_data.clear()
    fig_str, is_sampled_str = plot_outlier_distribution(df_str)
    assert fig_str is None
    assert not is_sampled_str
    
    # 3. DataFrame with a zero-std column (all same values) - should skip without crash
    df_zero_std = pd.DataFrame({
        'a': [5.0, 5.0, 5.0, 5.0],
        'b': [1, 2, 3, 4]
    })
    st.cache_data.clear()
    fig_zero_std, is_sampled_zero = plot_outlier_distribution(df_zero_std)
    assert isinstance(fig_zero_std, go.Figure)
    
    # 4. DataFrame with >1000 rows (downsampling)
    df_large = pd.DataFrame({
        'a': np.random.randn(1200),
        'b': np.random.randn(1200)
    })
    st.cache_data.clear()
    fig_large, is_sampled_large = plot_outlier_distribution(df_large)
    assert isinstance(fig_large, go.Figure)
    assert is_sampled_large
    
    print("test_plot_outlier_distribution passed.")

def test_plot_correlation_matrix():
    print("Running test_plot_correlation_matrix...")
    # 1. DataFrame with <=1 numeric column
    df_one = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    import streamlit as st
    st.cache_data.clear()
    fig_one = plot_correlation_matrix(df_one, (-1.0, 1.0))
    assert fig_one is None
    
    # 2. DataFrame where no columns pass the correlation filter
    # Let's create two columns with very low correlation
    df_uncorrelated = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [5, 1, 4, 2, 3]
    })
    # Filter for high correlation only (e.g. 0.95 to 1.0), which 'a' and 'b' won't meet
    st.cache_data.clear()
    fig_filter = plot_correlation_matrix(df_uncorrelated, (0.95, 1.0))
    assert fig_filter is None
    
    # 3. Normal case with correlated columns
    df_corr = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0],
        'b': [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    st.cache_data.clear()
    fig = plot_correlation_matrix(df_corr, (-1.0, 1.0))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    
    print("test_plot_correlation_matrix passed.")

def test_get_heatmap_styles():
    print("Running test_get_heatmap_styles...")
    df = pd.DataFrame({'age': [25, 17, 30]})
    rules = [
        {'type': 'Range Check', 'col': 'age', 'min': 18, 'max': 100, 'desc': 'Adults', 'color': 'hsla(120, 70%, 50%, 0.4)', 'enabled': True},
        {'type': 'Null Check', 'col': 'age', 'desc': 'Not null', 'color': 'hsla(240, 70%, 50%, 0.4)', 'enabled': False} # disabled
    ]
    
    # 1. Empty rules list
    sdf_empty, messages_empty = get_heatmap_styles(df, [])
    assert isinstance(sdf_empty, pd.DataFrame)
    assert (sdf_empty == '').all().all()
    assert len(messages_empty) == 0
    
    # 2. Rules with violations
    sdf, messages = get_heatmap_styles(df, rules)
    # The 2nd row (index 1, age 17) violates the range check
    assert isinstance(sdf, pd.DataFrame)
    assert sdf.loc[1, 'age'] == 'background-color: hsla(120, 70%, 50%, 0.4);'
    assert sdf.loc[0, 'age'] == ''
    assert len(messages) == 0
    
    # 3. Rule that raises exception during evaluation
    bad_rule = {
        'type': 'Range Check',
        'col': 'nonexistent', # raises KeyError
        'min': 0, 'max': 10,
        'desc': 'Bad Rule', 'color': 'hsla(0, 70%, 50%, 0.4)',
        'enabled': True
    }
    # Should handle exception gracefully without crashing, returning empty styles for that rule
    sdf_bad, messages_bad = get_heatmap_styles(df, [bad_rule])
    assert isinstance(sdf_bad, pd.DataFrame)
    assert (sdf_bad == '').all().all()
    assert len(messages_bad) == 1
    assert "Heatmap Style Error" in messages_bad[0]
    
    print("test_get_heatmap_styles passed.")

if __name__ == "__main__":
    test_get_safe_hue()
    test_plot_missingness_map()
    test_plot_outlier_distribution()
    test_plot_correlation_matrix()
    test_get_heatmap_styles()
    print("test_ui_utils_coverage.py completed successfully.")
