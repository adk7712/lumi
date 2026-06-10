import pandas as pd
import numpy as np
from typing import List, Tuple
from engine_ops import TRANSFORM_REGISTRY, CleaningStep
from codegen import generate_pipeline_code

def apply_recipe(df: pd.DataFrame, recipe: List[CleaningStep]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Applies a sequence of cleaning steps to a DataFrame using a dispatcher pattern.
    """
    df_clean = df.copy()
    messages = []
    for step in recipe:
        action = step.get('action')
        handler = TRANSFORM_REGISTRY.get(action)
        if handler:
            try:
                df_clean, step_messages = handler(df_clean, step)
                messages.extend(step_messages)
            except Exception as e:
                messages.append(f"An unexpected error occurred applying {action}: {type(e).__name__} - {str(e)}")
        else:
            messages.append(f"Warning: Unknown action '{action}' encountered in recipe.")
    return df_clean, messages
