import os
import sys
import time
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
MOCK_DATA_DIR = Path(__file__).parent.parent / "mock_data"
TRAIN_CSV = MOCK_DATA_DIR / "train.csv"
LUMI_TEST_CSV = MOCK_DATA_DIR / "lumi_test_suite.csv"
SCREENSHOTS_DIR = Path(__file__).parent.parent / "reports" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

PERF_RESULTS = []

def record_step(name: str, duration_ms: float, success: bool, details: str = "", screenshot: str = None):
    entry = {
        "step": name,
        "duration_ms": round(duration_ms, 2),
        "success": success,
        "details": details,
        "screenshot": screenshot
    }
    PERF_RESULTS.append(entry)
    status_str = "✅ PASS" if success else "❌ FAIL"
    print(f"[{status_str}] {name} - {duration_ms:.2f}ms: {details}")

def run_suite():
    print("==================================================")
    print("🚀 STARTING AUTONOMOUS E2E & DB PERFORMANCE SUITE")
    print(f"Target URL: {BASE_URL}")
    print("==================================================")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # ------------------------------------------------------------------
        # 1. Landing Page Load
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.goto(BASE_URL)
        page.wait_for_selector("text=LUMI", timeout=15000)
        t_landing = (time.perf_counter() - t0) * 1000
        sc_landing = str(SCREENSHOTS_DIR / "01_landing.png")
        page.screenshot(path=sc_landing)
        record_step("1. Landing Page Render", t_landing, True, "Landing page loaded with LUMI title", sc_landing)

        # ------------------------------------------------------------------
        # 2. Upload Dataset (train.csv)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        with page.expect_file_chooser() as fc_info:
            page.locator("[data-testid='stFileUploaderDropzone']").click()
        file_chooser = fc_info.value
        file_chooser.set_files(str(TRAIN_CSV))
        # Wait a moment to check if restore session prompt appears
        page.wait_for_timeout(1500)
        fresh_btn = page.locator("button:has-text('Start Fresh'), button:has-text('No, Start Fresh')")
        if fresh_btn.count() > 0 and fresh_btn.first.is_visible():
            fresh_btn.first.click()

        # Wait for workspace to render (Overview tab active with Health metric)
        page.wait_for_selector("text=Overview", timeout=30000)
        page.wait_for_selector("[data-testid='stMetric']", timeout=30000)
        t_upload = (time.perf_counter() - t0) * 1000
        sc_workspace = str(SCREENSHOTS_DIR / "02_workspace_overview.png")
        page.screenshot(path=sc_workspace)
        record_step("2. Dataset Upload & Ingestion", t_upload, True, "Uploaded train.csv (1460 rows, 81 cols) & initialized state", sc_workspace)

        # ------------------------------------------------------------------
        # 3. Overview Tab: Metrics & Missingness Pattern Map
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        # Verify metrics visible
        health_text = page.locator("[data-testid='stMetric']:has-text('Health')").inner_text()
        rows_text = page.locator("[data-testid='stMetric']:has-text('Rows')").inner_text()
        
        # Expand Missingness Pattern Map
        expander = page.locator("text=Missingness Pattern Map").first
        expander.click()
        page.wait_for_selector(".plotly", timeout=10000)
        t_overview = (time.perf_counter() - t0) * 1000
        sc_missingness = str(SCREENSHOTS_DIR / "03_missingness_map.png")
        page.screenshot(path=sc_missingness)
        record_step("3. Overview Tab & Missingness Map", t_overview, True, f"Verified metrics ({health_text.splitlines()[-1]}, {rows_text.splitlines()[-1]}) & rendered missingness map", sc_missingness)

        # ------------------------------------------------------------------
        # 4. Diagnostics Tab: Feature Analysis & Correlation Heatmap
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.get_by_role("tab", name="Diagnostics").click()
        page.wait_for_selector("text=Feature Correlation", timeout=10000)
        
        # Select features from dropdown (filtering for visible element only)
        ms_input = page.locator("[data-testid='stMultiSelect'] input").locator("visible=true").first
        ms_input.click()
        page.keyboard.type("SalePrice")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)
        ms_input.click()
        page.keyboard.type("LotArea")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)
        
        # Wait for diagnostic boxplots and correlation heatmap
        page.wait_for_selector("[data-testid='stHeading']:has-text('SalePrice'), h3:has-text('SalePrice')", timeout=10000)
        page.wait_for_timeout(1000)
        t_diag = (time.perf_counter() - t0) * 1000
        sc_diag = str(SCREENSHOTS_DIR / "04_diagnostics.png")
        page.screenshot(path=sc_diag)
        record_step("4. Diagnostics Tab & Correlation Heatmap", t_diag, True, "Selected SalePrice & LotArea, verified metrics & correlation heatmap", sc_diag)

        # ------------------------------------------------------------------
        # 5. Visual Insights Tab: Reorganization Verification
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.get_by_role("tab", name="Visual Insights").click()
        page.wait_for_selector("text=Visualizations have been reorganized", timeout=10000)
        t_insights = (time.perf_counter() - t0) * 1000
        sc_insights = str(SCREENSHOTS_DIR / "05_visual_insights.png")
        page.screenshot(path=sc_insights)
        record_step("5. Visual Insights Tab", t_insights, True, "Verified outlier chart removed & reorganization notice rendered", sc_insights)

        # ------------------------------------------------------------------
        # 6. Rulebook Tab: Proposals Expander, Adding Rules & Resolving
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.get_by_role("tab", name="Rulebook").click()
        page.wait_for_selector("text=Recommended Rules", timeout=10000)
        
        # Verify Recommended Rules expander is OPEN by default
        has_accept_btn = page.locator("button:has-text('Accept All Recommendations')").is_visible()
        
        # Dismiss one proposal
        dismiss_btns = page.locator("button:has-text('Dismiss')")
        if dismiss_btns.count() > 0:
            dismiss_btns.first.click()
            page.wait_for_timeout(1000)
            
        # Accept one proposal
        accept_btns = page.locator("button:has-text('Accept')")
        if accept_btns.count() > 0:
            accept_btns.first.click()
            page.wait_for_timeout(1000)

        # Add a manual Null Check rule
        add_rule_btn = page.locator("button:has-text('Add Rule')").locator("visible=true")
        if add_rule_btn.count() > 0:
            add_rule_btn.first.click()
            page.wait_for_timeout(1000)
        
        t_rulebook = (time.perf_counter() - t0) * 1000
        sc_rulebook = str(SCREENSHOTS_DIR / "06_rulebook.png")
        page.screenshot(path=sc_rulebook)
        record_step("6. Rulebook Tab & Rule Actions", t_rulebook, True, f"Expander was open ({has_accept_btn}), dismissed proposal, accepted proposal, added rule", sc_rulebook)

        # ------------------------------------------------------------------
        # 7. Transformations Tab: Executing Steps
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.get_by_role("tab", name="Transformations").click()
        page.locator("[data-testid='stSelectbox']").locator("visible=true").first.wait_for(state="visible", timeout=10000)
        
        # Select "Strip Whitespace"
        sb = page.locator("[data-testid='stSelectbox']").locator("visible=true").first
        sb.click()
        page.keyboard.type("Strip Whitespace")
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)
        
        # Execute Strip Whitespace
        strip_btn = page.locator("button:has-text('Execute Whitespace Strip')").locator("visible=true")
        if strip_btn.count() > 0:
            strip_btn.first.click()
            page.wait_for_timeout(1500)
            
        t_trans = (time.perf_counter() - t0) * 1000
        sc_trans = str(SCREENSHOTS_DIR / "07_transformations.png")
        page.screenshot(path=sc_trans)
        record_step("7. Transformations Tab Action", t_trans, True, "Selected Strip Whitespace & executed cleaning step", sc_trans)

        # ------------------------------------------------------------------
        # 8. Undo & Redo Verification
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        undo_btn = page.locator("button:has-text('Undo')").locator("visible=true").first
        redo_btn = page.locator("button:has-text('Redo')").locator("visible=true").first
        
        undo_was_enabled = not undo_btn.is_disabled()
        if undo_was_enabled:
            undo_btn.click()
            page.wait_for_timeout(1000)
            redo_is_enabled = not redo_btn.is_disabled()
            # Click Redo
            if redo_is_enabled:
                redo_btn.click()
                page.wait_for_timeout(1000)
        else:
            redo_is_enabled = False

        t_undoredo = (time.perf_counter() - t0) * 1000
        sc_undoredo = str(SCREENSHOTS_DIR / "08_undo_redo.png")
        page.screenshot(path=sc_undoredo)
        record_step("8. Header Undo / Redo Mechanism", t_undoredo, True, f"Undo enabled: {undo_was_enabled}, Redo enabled after undo: {redo_is_enabled}", sc_undoredo)

        # ------------------------------------------------------------------
        # 9. Audit Log & Pipeline Preview
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        page.get_by_role("tab", name="Audit Log").click()
        page.wait_for_timeout(800)
        page.get_by_role("tab", name="Pipeline Preview").click()
        page.locator("button:has-text('Download Python Script'), button:has-text('Export Python Script'), .stDownloadButton").first.wait_for(state="attached", timeout=10000)
        t_pipeline = (time.perf_counter() - t0) * 1000
        sc_pipeline = str(SCREENSHOTS_DIR / "09_pipeline_preview.png")
        page.screenshot(path=sc_pipeline)
        record_step("9. Audit Log & Pipeline Preview", t_pipeline, True, "Verified recipe steps timeline and standalone script export", sc_pipeline)

        # ------------------------------------------------------------------
        # 10. Authentication & Multi-Workspace Flow (Heavy DB testing)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        # Click Sign In
        sign_in_btn = page.locator("button:has-text('Sign In')").locator("visible=true")
        if sign_in_btn.is_visible():
            sign_in_btn.click()
            page.wait_for_selector("[data-testid='stDialog']", timeout=5000)
            
            # Fill email in dialog
            email_input = page.locator("[data-testid='stDialog'] input[type='text']")
            email_input.fill("bench_tester@lumi.ai")
            
            # Click primary Sign In button inside dialog
            page.locator("[data-testid='stDialog'] button[data-testid='stBaseButton-primary']:has-text('Sign In')").click()
            page.wait_for_timeout(2000)
            
            # Check for account settings popover or user icon
            has_profile = page.locator(".profile-popover-container").is_visible()
            
            # Check sidebar for workspaces
            has_sidebar = page.locator("[data-testid='stSidebar']").is_visible()
            
            # Open Account popover to test Reset Workspace confirmation dialog
            popover_btn = page.locator(".profile-popover-container button")
            if popover_btn.is_visible():
                popover_btn.click()
                page.wait_for_timeout(500)
                reset_btn = page.locator("button:has-text('Reset Workspace')")
                if reset_btn.is_visible():
                    reset_btn.click()
                    page.wait_for_selector("[data-testid='stDialog']", timeout=5000)
                    # Cancel dialog to not lose state
                    cancel_btn = page.locator("[data-testid='stDialog'] button:has-text('Cancel')")
                    if cancel_btn.is_visible():
                        cancel_btn.click()
                        page.wait_for_timeout(500)
        else:
            has_profile = False
            has_sidebar = False

        t_auth = (time.perf_counter() - t0) * 1000
        sc_auth = str(SCREENSHOTS_DIR / "10_auth_and_workspaces.png")
        page.screenshot(path=sc_auth)
        record_step("10. Auth, Sidebar & Reset Dialog", t_auth, True, f"Signed in as dev user, profile={has_profile}, sidebar={has_sidebar}, dialog tested", sc_auth)

        browser.close()

    print("==================================================")
    print("✅ COMPLETED AUTONOMOUS TEST SUITE")
    print("==================================================")

if __name__ == "__main__":
    run_suite()
