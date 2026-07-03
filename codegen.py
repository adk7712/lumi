import json

def _generate_clean_data_lines(recipe: list) -> list[str]:
    code = []
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
                    if v == "mode":
                        # Safely handle empty modes in generated code to avoid IndexError
                        code.append(f"    mode_val = df['{col}'].mode()")
                        code.append(f"    df['{col}'] = df['{col}'].fillna(mode_val[0] if not mode_val.empty else np.nan)")
                    else:
                        code.append(f"    df['{col}'] = df['{col}'].fillna(df['{col}'].{v}())")
                elif v in ["knn", "iterative"]:
                    code.append(f"    # Advanced Imputation using scikit-learn.")
                    if v == "knn":
                        code.append(f"    from sklearn.impute import KNNImputer")
                        code.append(f"    imputer = KNNImputer(n_neighbors=5)")
                    else:
                        code.append(f"    from sklearn.experimental import enable_iterative_imputer")
                        code.append(f"    from sklearn.impute import IterativeImputer")
                        code.append(f"    imputer = IterativeImputer(random_state=42)")
                    
                    code.append(f"    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()")
                    code.append(f"    df_imputed = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)")
                    code.append(f"    df['{col}'] = df_imputed['{col}']")
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
                    
                    if target_dtype in ['string', 'object']:
                        code.append(f"    # Convert column to target string/object type.")
                        code.append(f"    df['{col}'] = df['{col}'].astype('{target_dtype}')")
                    else:
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
                regex_param = f", regex={step.get('regex', False)}"
                if col == "All":
                    code.append(f"    # Replace in all object columns if 'All' is specified.")
                    code.append(f"    for c in df.select_dtypes(include=['object']).columns:")
                    code.append(f"        df[c] = df[c].replace({f_repr}, {r_repr}{regex_param})")
                else:
                    code.append(f"    # Replace in the specified column.")
                    code.append(f"    df['{col}'] = df['{col}'].replace({f_repr}, {r_repr}{regex_param})")
            elif action == "strip_whitespace":
                if col == "All":
                    code.append(f"    # Strip whitespace from all object columns, preserving NaNs.")
                    code.append(f"    for c in df.select_dtypes(include=['object']).columns:")
                    code.append(f"        mask = df[c].notnull()")
                    code.append(f"        df.loc[mask, c] = df.loc[mask, c].astype(str).str.strip()")
                else:
                    code.append(f"    # Strip whitespace from the specified column, preserving NaNs.")
                    code.append(f"    mask = df['{col}'].notnull()")
                    code.append(f"    df.loc[mask, '{col}'] = df.loc[mask, '{col}'].astype(str).str.strip()")
            elif action == "normalize_text":
                method = step.get('value', 'lowercase')
                code.append(f"    # Normalize text using {method} method.")
                target = f"df['{col}']" if col != "All" else "df[c]"
                loop_start = [f"    for c in df.select_dtypes(include=['object']).columns:"] if col == "All" else []
                indent = "        " if col == "All" else "    "
                
                if method == "remove_punctuation":
                    code.extend(loop_start)
                    code.append(f"{indent}import string")
                    code.append(f"{indent}{target} = {target}.astype(str).str.replace(f'[{{string.punctuation}}]', '', regex=True)")
                elif method == "fuzzy_dedupe":
                    code.append(f"    from thefuzz import process")
                    code.extend(loop_start)
                    code.append(f"{indent}unique_vals = {target}.dropna().unique()")
                    code.append(f"{indent}mapping = {{}}")
                    code.append(f"{indent}handled = set()")
                    code.append(f"{indent}for v in unique_vals:")
                    code.append(f"{indent}    if v in handled: continue")
                    code.append(f"{indent}    matches = process.extract(v, unique_vals, limit=10)")
                    code.append(f"{indent}    for match, score in matches:")
                    code.append(f"{indent}        if score > 85: mapping[match] = v; handled.add(match)")
                    code.append(f"{indent}{target} = {target}.replace(mapping)")
                else:
                    pandas_method = method
                    if method == "lowercase": pandas_method = "lower"
                    elif method == "uppercase": pandas_method = "upper"
                    elif method == "titlecase": pandas_method = "title"
                    code.extend(loop_start)
                    code.append(f"{indent}{target} = {target}.astype(str).str.{pandas_method}()")
            elif action == "log_transform":
                code.append(f"    # Apply log(1+x) transformation to handle outliers.")
                code.append(f"    df['{col}'] = np.log1p(df['{col}'].clip(lower=0))")
            elif action == "rename_column":
                new_name = step['value']
                code.append(f"    # Rename column '{col}' to '{new_name}'.")
                code.append(f"    df = df.rename(columns={{'{col}': '{new_name}'}})")
            elif action == "reorder_columns":
                new_order = step['value']
                code.append(f"    # Reorder columns to the specified layout.")
                code.append(f"    df = df[{repr(new_order)}]")
            elif action == "extract_datetime":
                new_col = step['new_column']
                component = step['component']
                accessor = f"dt.{component}" if component != "day_of_week" else "dt.day_name()"
                code.append(f"    # Extract {component} component from '{col}' into '{new_col}'.")
                code.append(f"    df['{new_col}'] = pd.to_datetime(df['{col}']).{accessor}")
            elif action == "drop_duplicates":
                code.append("    # Remove duplicate rows from the DataFrame.")
                code.append("    df = df.drop_duplicates()")
            elif action == "drop_empty_columns":
                code.append("    # Drop columns that are completely empty (all nulls).")
                code.append("    empty_cols = [c for c in df.columns if df[c].isnull().all()]")
                code.append("    df = df.drop(columns=empty_cols)")
            elif action == "drop_empty_rows":
                code.append("    # Remove rows that are completely empty (all nulls).")
                code.append("    df = df.dropna(how='all')")
    code.append("    return df")
    return code


def _generate_validate_data_lines(rules: list) -> list[str]:
    code = []
    code.append("    \"\"\"")
    code.append("    Validates the DataFrame against the rules defined in the Lumi Rulebook.")
    code.append("    Returns a dictionary of {rule_description: violation_count_or_error}.")
    code.append("    \"\"\"")
    code.append("    violations = {}")

    active_rules = []
    if rules:
        active_rules = [r for r in rules if r.get('enabled', True) and r.get('type') != 'Informational']

    if not active_rules:
        code.append("    # No active validation rules defined")
    else:
        for r in active_rules:
            desc = r.get('desc', 'N/A')
            desc_repr = repr(desc)
            code.append(f"    # Rule: {r['type']}")
            code.append(f"    # Description: {desc}")
            code.append("    try:")
            if r['type'] == "Null Check":
                code.append(f"        mask = df['{r['col']}'].isnull()")
            elif r['type'] == "Range Check":
                code.append(f"        col_numeric = pd.to_numeric(df['{r['col']}'], errors='coerce')")
                code.append(f"        mask = (col_numeric < {r['min']}) | (col_numeric > {r['max']}) | col_numeric.isnull()")
            elif r['type'] == "Relational Check":
                val = f"df['{r['col_b']}']" if r.get('target_type') == 'Feature' else repr(r['value'])
                code.append(f"        valid = (df['{r['col_a']}'] {r['op']} {val})")
                code.append(f"        mask = ~valid")
            elif r['type'] == "Custom Expression":
                code.append(f"        valid_indices = df.query({repr(r['query'])}).index")
                code.append(f"        mask = ~df.index.isin(valid_indices)")
            else:
                code.append("        mask = pd.Series(False, index=df.index)")

            code.append(f"        count = mask.sum()")
            code.append(f"        if count > 0:")
            code.append(f"            violations[{desc_repr}] = int(count)")
            code.append("    except Exception as e:")
            code.append(f"        violations[{desc_repr}] = f\"Error: {{type(e).__name__}} - {{str(e)}}\"")
            code.append("") # Spacer between rules

    code.append("    return violations")
    return code


def generate_pipeline_code(recipe: list, rules: list = None) -> str:
    """
    Generates standalone Python code for a given cleaning recipe and optional validation rules.
    """
    code = ["import pandas as pd\nimport numpy as np\n", "def clean_data(df):"]
    code.extend(_generate_clean_data_lines(recipe))
    code.append("\n\ndef validate_data(df):")
    code.extend(_generate_validate_data_lines(rules))

    # Add example runner block
    code.append("\nif __name__ == \"__main__\":")
    code.append("    # Example usage:")
    code.append("    # df = pd.read_csv(\"your_data.csv\")")
    code.append("    # df_cleaned = clean_data(df)")
    code.append("    # report = validate_data(df_cleaned)")
    code.append("    # if report:")
    code.append("    #     print(\"Validation failed with issues:\", report)")
    code.append("    # else:")
    code.append("    #     print(\"Validation passed!\")")
    code.append("    pass")

    return "\n".join(code)


def generate_notebook_code(recipe: list, rules: list = None) -> str:
    """
    Generates a Jupyter Notebook (.ipynb) as a JSON-encoded string for the given recipe and rules.
    """
    clean_lines = ["def clean_data(df):\n"] + [f"{line}\n" for line in _generate_clean_data_lines(recipe)]
    validate_lines = ["def validate_data(df):\n"] + [f"{line}\n" for line in _generate_validate_data_lines(rules)]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Lumi Data Cleaning & Validation Pipeline\n",
                "\n",
                "This notebook contains the data cleaning and validation pipeline automatically generated by **Lumi**.\n",
                "\n",
                "### Notebook structure:\n",
                "1. **Imports cell**: Loads `pandas` and `numpy` dependencies.\n",
                "2. **`clean_data(df)` function**: Applies the sequential data cleaning recipe.\n",
                "3. **`validate_data(df)` function**: Runs assertions and checks active validation rules.\n",
                "4. **Example Runner**: Demonstrates how to run the pipeline on your datasets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": clean_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": validate_lines
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Example Usage:\n",
                "# df = pd.read_csv(\"your_data.csv\")\n",
                "# df_cleaned = clean_data(df)\n",
                "# violations = validate_data(df_cleaned)\n",
                "# if violations:\n",
                "#     print(\"Validation failed with issues:\", violations)\n",
                "# else:\n",
                "#     print(\"Validation passed!\")\n"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    return json.dumps(notebook, indent=2)
