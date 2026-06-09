import pandas as pd
from streamlit.testing.v1 import AppTest

def test_app_initialization():
    # Load app in memory
    at = AppTest.from_file("app.py")
    at.run()
    
    # Assert no exceptions occurred during execution
    assert not at.exception, "App should run without exceptions"
    
    # Verify session state values are correctly initialized
    assert at.session_state.raw_data is not None, "Raw data should be loaded"
    assert isinstance(at.session_state.raw_data, pd.DataFrame), "Raw data should be a DataFrame"
    assert len(at.session_state.intermediate_states) == 1, "Should have 1 state initially"
    assert at.session_state.intermediate_states[0][0] == "Original Data"
    print("Test App Initialization passed.")

def test_multiselect_diagnostics():
    at = AppTest.from_file("app.py")
    at.run()
    
    # Retrieve the multiselect widget by its key
    multiselect = at.multiselect(key="active_features")
    assert multiselect is not None
    
    # Verify columns are available for selection
    cols = multiselect.options
    assert len(cols) > 0, "Should have columns to select"
    
    # Select the first column and run the app
    multiselect.select(cols[0]).run()
    
    # Verify state updated and no exceptions raised
    assert not at.exception
    assert cols[0] in at.session_state.active_features, "Selected column should be in active_features"
    print("Test Multiselect Diagnostics passed.")

def test_rule_addition_and_clear():
    at = AppTest.from_file("app.py")
    at.run()
    
    # Initial rules list should be empty
    assert len(at.session_state.rules) == 0
    
    # Change rule type to Informational
    rule_type = at.selectbox(key="rule_type_select")
    rule_type.select("Informational").run()
    
    # Fill in the text area note
    note_input = at.text_area(key="info_note_input")
    note_input.input("Test info note").run()
    
    # Click the Add Rule button
    add_btn = at.button(key="btn_add_info")
    add_btn.click().run()
    
    # Verify rule was successfully added
    assert len(at.session_state.rules) == 1
    assert at.session_state.rules[0]["type"] == "Informational"
    assert at.session_state.rules[0]["desc"] == "Test info note"
    
    # Click Clear All rules button
    clear_btn = at.button(key="clear_all_rules_btn")
    clear_btn.click().run()
    
    # Verify rules and recipe were cleared
    assert len(at.session_state.rules) == 0
    print("Test Rule Addition and Clear passed.")

def test_undo_and_reset():
    at = AppTest.from_file("app.py")
    at.run()
    
    # Change manual transformation type to Strip Whitespace
    trans_select = at.selectbox(key="trans_type_select")
    trans_select.select("Strip Whitespace").run()
    
    # Click Add Step button
    add_btn = at.button(key="btn_strip")
    add_btn.click().run()
    
    # Verify recipe and cache state have updated
    assert len(at.session_state.cleaning_recipe) == 1
    assert len(at.session_state.intermediate_states) == 2
    
    # Click Undo button
    undo_btn = at.button(key="undo_btn")
    assert not undo_btn.disabled
    undo_btn.click().run()
    
    # Verify recipe is reverted
    assert len(at.session_state.cleaning_recipe) == 0
    assert len(at.session_state.intermediate_states) == 1
    
    # Click Reset
    reset_btn = at.button(key="reset_all")
    reset_btn.click().run()
    assert len(at.session_state.cleaning_recipe) == 0
    print("Test Undo and Reset passed.")

def test_rename_and_reorder_ui():
    at = AppTest.from_file("app.py")
    at.run()
    
    # 1. Add a dummy rule first to test renaming synchronization
    rule_type = at.selectbox(key="rule_type_select")
    rule_type.select("Null Check").run()
    target_col = at.session_state.intermediate_states[-1][3].columns[0]
    target_col_select = at.selectbox(key="rule_target_col")
    target_col_select.select(target_col).run()
    add_rule_btn = at.button(key="btn_add_null")
    add_rule_btn.click().run()
    
    assert len(at.session_state.rules) == 1
    assert at.session_state.rules[0]["col"] == target_col
    
    # Select Rename Column manual transformation
    trans_select = at.selectbox(key="trans_type_select")
    trans_select.select("Rename Column").run()
    
    # Input renaming arguments
    rename_target = at.selectbox(key="rename_target_col")
    rename_target.select(target_col).run()
    
    new_name_input = at.text_input(key="rename_new_name_input")
    new_name_val = f"{target_col}_renamed"
    new_name_input.input(new_name_val).run()
    
    # Click Apply Rename
    apply_rename_btn = at.button(key="btn_rename")
    apply_rename_btn.click().run()
    
    # Assertions for Rename
    assert not at.exception
    assert len(at.session_state.cleaning_recipe) == 1
    assert at.session_state.cleaning_recipe[0]["action"] == "rename_column"
    assert at.session_state.cleaning_recipe[0]["column"] == target_col
    assert at.session_state.cleaning_recipe[0]["value"] == new_name_val
    
    # Verify rule synced
    assert at.session_state.rules[0]["col"] == new_name_val
    assert new_name_val in at.session_state.rules[0]["desc"]
    
    # 2. Reorder columns UI simulation
    trans_select = at.selectbox(key="trans_type_select")
    trans_select.select("Reorder Columns").run()
    
    # Verify temp order initialized
    assert len(at.session_state.temp_col_order) > 1
    first_col = at.session_state.temp_col_order[0]
    second_col = at.session_state.temp_col_order[1]
    
    # Simulate drag-and-drop order swap in session state
    at.session_state.temp_col_order[0], at.session_state.temp_col_order[1] = second_col, first_col
    at.run()
    
    # Verify swapped in temp state
    assert at.session_state.temp_col_order[0] == second_col
    assert at.session_state.temp_col_order[1] == first_col
    
    # Click apply
    apply_reorder_btn = at.button(key="btn_apply_reorder")
    apply_reorder_btn.click().run()
    
    # Verify step registered in recipe
    assert len(at.session_state.cleaning_recipe) == 2
    assert at.session_state.cleaning_recipe[1]["action"] == "reorder_columns"
    assert at.session_state.cleaning_recipe[1]["value"][0] == second_col
    assert at.session_state.cleaning_recipe[1]["value"][1] == first_col
    
    print("Rename and Reorder UI integration tests passed.")

def test_visual_insights_tab_ui():
    at = AppTest.from_file("app.py")
    at.run()
    
    slider = at.slider(key="corr_range_val")
    assert slider is not None, "Slider for correlation range should be present"
    assert slider.value == (-1.0, 1.0)
    
    slider.set_value((0.3, 0.8)).run()
    assert not at.exception
    print("Visual Insights UI integration test passed.")

def test_audit_log_remove_step():
    at = AppTest.from_file("app.py")
    at.run()
    
    # 1. Add three distinct transformations to the recipe
    trans_select = at.selectbox(key="trans_type_select")
    trans_select.select("Strip Whitespace").run()
    at.button(key="btn_strip").click().run()
    
    trans_select.select("Cast Data Type").run()
    sb = at.selectbox(key="cast_target_col")
    sb.select("Id").run()
    at.selectbox(key="cast_dtype_select").select("string").run()
    at.button(key="btn_cast").click().run()
    
    trans_select.select("Drop Column").run()
    at.selectbox(key="drop_target_col").select("Alley").run()
    at.button(key="btn_drop").click().run()
    
    assert len(at.session_state.cleaning_recipe) == 3
    assert len(at.session_state.intermediate_states) == 4
    
    # 2. Simulate clicking "Remove" on Step 2 (i=2)
    rm_btn = at.button(key="rm_step_2")
    assert rm_btn is not None
    rm_btn.click().run()
    
    # Verify Step 2 is removed, leaving 2 steps
    assert len(at.session_state.cleaning_recipe) == 2
    assert len(at.session_state.intermediate_states) == 3
    assert at.session_state.cleaning_recipe[0]["action"] == "strip_whitespace"
    assert at.session_state.cleaning_recipe[1]["action"] == "drop_column"
    assert "Alley" not in at.session_state.intermediate_states[-1][3].columns
    print("Audit Log Remove step integration test passed.")

if __name__ == "__main__":
    try:
        test_app_initialization()
        test_multiselect_diagnostics()
        test_rule_addition_and_clear()
        test_undo_and_reset()
        test_rename_and_reorder_ui()
        test_visual_insights_tab_ui()
        test_audit_log_remove_step()
        print("ALL APP FLOW TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
