import pandas as pd
import numpy as np
from rule_utils import evaluate_rule

def apply_recipe(df: pd.DataFrame, recipe: list) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies a sequence of cleaning steps to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to which cleaning steps will be applied.
        recipe (list): A list of dictionaries, where each dictionary represents a cleaning step
                       with 'action' and other relevant parameters (e.g., 'column', 'value').

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing:
            - df_clean (pd.DataFrame): The DataFrame after applying all cleaning steps.
            - messages (list[str]): A list of messages (warnings or errors) encountered during the application of steps.
    """
    df_clean = df.copy()
    messages = []
    for step in recipe:
        try:
            action, col = step['action'], step.get('column')
            if action == "drop_column":
                # Remove the specified column from the DataFrame.
                if col in df_clean.columns:
                    df_clean = df_clean.drop(columns=[col])
                else:
                    messages.append(f"Warning: Column '{col}' not found for drop_column action.")
            elif action == "drop_nulls":
                # Remove rows where the specified column has null values.
                if col in df_clean.columns:
                    df_clean = df_clean.dropna(subset=[col])
                else:
                    messages.append(f"Warning: Column '{col}' not found for drop_nulls action.")
            elif action == "fill_null":
                # Fill null values in the specified column with a given value (mean, median, mode, or custom).
                if col in df_clean.columns:
                    val = step['value']
                    fill_value = None
                    if val == "mean":
                        fill_value = df_clean[col].mean()
                    elif val == "median":
                        fill_value = df_clean[col].median()
                    elif val == "mode":
                        # Mode can return multiple values, take the first
                        mode_result = df_clean[col].mode()
                        if not mode_result.empty:
                            fill_value = mode_result[0]
                    else:
                        fill_value = val
                    
                    if fill_value is not None:
                        df_clean[col] = df_clean[col].fillna(fill_value)
                    else:
                        messages.append(f"Warning: Could not determine fill value for column '{col}' with strategy '{val}'.")
                else:
                    messages.append(f"Warning: Column '{col}' not found for fill_null action.")
            elif action == "cap_range":
                # Cap values in the specified column to be within the defined min and max bounds.
                if col in df_clean.columns:
                    df_clean.loc[df_clean[col] < step['min'], col] = step['min']
                    df_clean.loc[df_clean[col] > step['max'], col] = step['max']
                else:
                    messages.append(f"Warning: Column '{col}' not found for cap_range action.")
            elif action == "cast_type":
                # Convert the specified column to a target data type, coercing errors to NaN.
                if col in df_clean.columns:
                    try:
                        target_dtype = step['dtype']
                        # Handle nullable integers (e.g., 'Int64') to preserve NaNs where appropriate,
                        # otherwise default to standard int types.
                        if target_dtype in ['int64', 'int32', 'int']:
                            target_dtype = "Int64" # Use nullable integer type
                        
                        if step['dtype'] == "datetime64[ns]":
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        else:
                            numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                            df_clean[col] = numeric_series.astype(target_dtype)
                    except Exception as e:
                        messages.append(f"Error: Could not cast '{col}' to {step['dtype']}: {str(e)}")
                        # Keep the coerced numeric series if an error occurs during final astype
                        if 'numeric_series' in locals():
                            df_clean[col] = numeric_series
                else:
                    messages.append(f"Warning: Column '{col}' not found for cast_type action.")
            elif action == "drop_violated":
                # Remove rows from the DataFrame that violate a specific rule.
                rule = step['rule']
                if rule.get('type') == "Informational":
                    continue
                try:
                    violation_mask = evaluate_rule(df_clean, rule)
                    df_clean = df_clean[~violation_mask]
                except Exception as e:
                    messages.append(f"Error: Could not apply drop_violated rule ({rule.get('desc', 'N/A')}): {str(e)}")
            elif action == "replace":
                # Replace occurrences of a find string with a replace string in a column, or all object columns.
                f, r_val = step['find'], step['replace']
                if col == "All":
                    for c in df_clean.select_dtypes(include=['object']).columns:
                        df_clean[c] = df_clean[c].replace(f, r_val)
                elif col in df_clean.columns:
                    df_clean[col] = df_clean[col].replace(f, r_val)
                else:
                    messages.append(f"Warning: Column '{col}' not found for replace action.")
        except (KeyError, ValueError, TypeError) as e:
            messages.append(f"Error applying {action} on {col}: {type(e).__name__} - {str(e)}")
            continue
        except Exception as e:
            messages.append(f"An unexpected error occurred applying {action} on {col}: {type(e).__name__} - {str(e)}")
            continue
    return df_clean, messages

def generate_pipeline_code(recipe: list) -> str:
    """
    Generates standalone Python code for a given cleaning recipe.

    This function iterates through the cleaning steps defined in the recipe
    and translates them into a executable Python code string using pandas operations.

    Args:
        recipe (list): A list of dictionaries, where each dictionary represents a cleaning step.
                       Each step defines an 'action' and associated parameters (e.g., 'column', 'value', 'rule').

    Returns:
        str: A multi-line string containing the generated Python code, starting with
             necessary imports and a `clean_data(df)` function definition.
    """
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
                        target_dtype = "Int64"
                    code.append(f"    # Convert column to target numeric type, coercing errors to NaN.")
                    code.append(f"    df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce').astype('{target_dtype}')")
            elif action == "drop_violated":
                r = step['rule']
                if r.get('type') == "Informational":
                    continue
                code.append(f"    # Drop rows violating rule: {r.get('desc', 'N/A')}")
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
