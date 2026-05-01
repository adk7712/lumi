import pandas as pd
import numpy as np
import streamlit as st
from rule_utils import evaluate_rule

def apply_recipe(df, recipe):
    """Applies a sequence of cleaning steps to a dataframe."""
    df_clean = df.copy()
    for step in recipe:
        try:
            action, col = step['action'], step.get('column')
            if action == "drop_column":
                # Remove the specified column from the DataFrame.
                df_clean = df_clean.drop(columns=[col])
            elif action == "drop_nulls":
                # Remove rows where the specified column has null values.
                df_clean = df_clean.dropna(subset=[col])
            elif action == "fill_null":
                # Fill null values in the specified column with a given value (mean, median, mode, or custom).
                val = step['value']
                if val == "mean": fill = df_clean[col].mean()
                elif val == "median": fill = df_clean[col].median()
                elif val == "mode": fill = df_clean[col].mode()[0]
                else: fill = val
                df_clean[col] = df_clean[col].fillna(fill)
            elif action == "cap_range":
                # Cap values in the specified column to be within the defined min and max bounds.
                df_clean.loc[df_clean[col] < step['min'], col] = step['min']
                df_clean.loc[df_clean[col] > step['max'], col] = step['max']
            elif action == "cast_type":
                # Convert the specified column to a target data type, coercing errors to NaN.
                try:
                    target_dtype = step['dtype']
                    # Handle nullable integers (e.g., 'Int64') to preserve NaNs where appropriate,
                    # otherwise default to standard int types.
                    if target_dtype in ['int64', 'int32', 'int']:
                        target_dtype = target_dtype.capitalize() # e.g., 'Int64'

                    if step['dtype'] == "datetime64[ns]":
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    else:
                        numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                        df_clean[col] = numeric_series.astype(target_dtype)
                except Exception as e:
                    st.toast(f"Could not cast '{col}' to {step['dtype']}: {str(e)}", icon="🚨")
                    df_clean[col] = numeric_series # Keep the coerced numeric series
            elif action == "drop_violated":
                # Remove rows from the DataFrame that violate a specific rule.
                rule = step['rule']
                if rule.get('type') == "Informational":
                    continue
                violation_mask = evaluate_rule(df_clean, rule)
                df_clean = df_clean[~violation_mask]
            elif action == "replace":
                # Replace occurrences of a find string with a replace string in a column, or all object columns.
                f, r_val = step['find'], step['replace']
                if col == "All":
                    for c in df_clean.select_dtypes(include=['object']).columns:
                        df_clean[c] = df_clean[c].replace(f, r_val)
                else:
                    df_clean[col] = df_clean[col].replace(f, r_val)
        except Exception as e:
            st.toast(f"Error applying {action} on {col}: {str(e)}", icon="🚨")
            continue
    return df_clean

def generate_pipeline_code(recipe):
    """Generates standalone Python code for the applied recipe."""
    code = ["import pandas as pd\nimport numpy as np\n", "def clean_data(df):"]
    if not recipe:
        code.append("    # No cleaning steps applied")
    else:
        for step in recipe:
            action, col = step['action'], step.get('column')
            if action == "drop_column":
                code.append(f"    # Remove the specified column from the DataFrame.")
                code.append(f"    df = df.drop(columns=['{col}'])")
            elif action == "drop_nulls":
                code.append(f"    # Remove rows where the specified column has null values.")
                code.append(f"    df = df.dropna(subset=['{col}'])")
            elif action == "fill_null":
                v = step['value']
                if v in ["mean", "median", "mode"]:
                    code.append(f"    # Fill null values with mean, median, or mode.")
                    code.append(f"    df['{col}'] = df['{col}'].fillna(df['{col}'].{v + ('()[0]' if v=='mode' else '()')})")
                else:
                    code.append(f"    # Fill null values with a custom constant.")
                    code.append(f"    df['{col}'] = df['{col}'].fillna({repr(v)})")
            elif action == "cap_range":
                code.append(f"    # Cap values in the specified column to be within the defined min and max bounds.")
                code.append(f"    df.loc[df['{col}'] < {step['min']}, '{col}'] = {step['min']}")
                code.append(f"    df.loc[df['{col}'] > {step['max']}, '{col}'] = {step['max']}")
            elif action == "cast_type":
                if step['dtype'] == "datetime64[ns]":
                    code.append(f"    # Convert column to datetime, coercing errors to NaT.")
                    code.append(f"    df['{col}'] = pd.to_datetime(df['{col}'], errors='coerce')")
                else:
                    target_dtype = step['dtype']
                    if target_dtype in ['int64', 'int32', 'int']:
                        target_dtype = target_dtype.capitalize()
                    code.append(f"    # Convert column to target numeric type, coercing errors to NaN.")
                    code.append(f"    df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('{target_dtype}')")
            elif action == "drop_violated":
                code.append(f"    # Drop rows violating rule: {r.get('desc', 'N/A')}")
                r = step['rule']
                if r.get('type') == "Informational":
                    continue
                if r['type'] == "Null Check":
                    code.append(f"    df = df.dropna(subset=['{r['col']}'])")
                elif r['type'] == "Range Check":
                    code.append(f"    df = df[(df['{r['col']}'] >= {r['min']}) & (df['{r['col']}'] <= {r['max']})]")
                elif r['type'] == "Relational Check":
                    val = f"df['{r['col_b']}']" if r.get('target_type') == 'Feature' else repr(r['value'])
                    code.append(f"    df = df[df['{r['col_a']}'] {r['op']} {val}]")
                elif r['type'] == "Custom Expression":
                    # Use repr() to safely embed the query string in the generated code.
                    code.append(f"    df = df.query({repr(r['query'])})")
            elif action == "replace":
                f_repr, r_repr = repr(step['find']), repr(step['replace'])
                if col == "All":
                    code.append(f"    # Replace in all object columns if 'All' is specified.")
                    code.append(f"    for c in df.select_dtypes(include=['object']).columns:")
                    code.append(f"        df[c] = df[c].replace({f_repr}, {r_repr})")
                else:
                    code.append(f"    # Replace in the specified column.")
                    code.append(f"    df['{col}'] = df['{col}'].replace({f_repr}, {r_repr})")
    code.append("    return df")
    return "\n".join(code)
