# Project Lumi

## Project Overview

This project, codenamed "Lumi," is an interactive web application for data cleaning and validation. Built with Python using the Streamlit framework, it allows users to upload a dataset (CSV/XLSX), diagnose data quality issues, apply a series of cleaning transformations, and export the entire workflow as a reusable Python script.

The core technologies used are:
- **Streamlit:** For building the interactive web user interface.
- **Pandas:** For all data manipulation and analysis tasks.
- **Plotly:** For generating interactive data visualizations (pie charts, correlation heatmaps, etc.).

## Building and Running the Project

To get the project running locally, follow these steps:

1.  **Install Dependencies:**
    It's recommended to use a virtual environment. Once activated, install the required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application:**
    Start the Streamlit server. The application will open in your default web browser.
    ```bash
    streamlit run app.py
    ```
    The application will load with a default sample dataset located in the `mock_data/` directory.

## Development Conventions

The application is architecturally simple and follows common practices for Streamlit applications.

-   **State Management:** All application state (the raw dataframe, active rules, cleaning recipe, etc.) is managed within Streamlit's `st.session_state` object.
-   **Modular Structure:** The code is organized into several key modules:
    -   `app.py`: The main entry point and UI layout. It handles user interactions and orchestrates the other modules.
    -   `engine.py`: Contains the core data processing logic. `apply_recipe` executes the cleaning steps, and `generate_pipeline_code` creates the exportable Python script.
    -   `scout.py`: Implements the automated data quality "scout" (`generate_proposals`) that scans the dataframe to proactively identify issues and suggest fixes.
    -   `ui_utils.py`: A collection of helper functions for styling the UI, such as injecting custom CSS.
-   **Data Flow:**
    1.  A dataset is loaded into `st.session_state.raw_data`.
    2.  `scout.py` generates "proposals" for cleaning.
    3.  Users accept proposals or create manual rules, which are added to the `st.session_state.cleaning_recipe`.
    4.  `engine.py` applies this recipe to the raw data to produce a clean dataframe.
    5.  The final recipe can be exported as a standalone Python function.
