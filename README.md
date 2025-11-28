# Assignment
# Code Folder Introduction
### 1. `.vscode/`

This folder contains specific configurations for the Visual Studio Code editor. It stores files for debugging (`launch.json`), C/C++ properties
(`c_cpp_properties.json`), and workspace settings (`settings.json`) to ensure a consistent development environment.

### 2. `advisor/artifacts/`

This folder is used to store a pre-trained machine learning model and all its auxiliary components. Our AI model is based on it.

### 3. `advisor/`

The folder constitutes a typical Python machine learning project.
# (1) data_preprocessor.py: Convert the "raw" and possibly disorganized data into a clean and standardized format so that machine learning models can understand it.
# (2) model_trainer.py: Train machine learning models.
# (3) analysis.py: Conduct exploratory data analysis on the data.
# (4) app.py: Web server script.
# (5) requirements.txt: All the Python libraries required for the project and their specific versions.

### 4. `data_cleaning/`

Contains the Jupyter Notebook for data cleaning and preprocessing. The raw data is processed here to ensure data quality and correct formatting before any analysis is performed.

### 5. `dataset/raw/`

Stores all the raw data files required for the project.

### 6. `dataset/`

Dataset for AI job market analysis:
(1) ai_job_market_unified.csv: The final dataset that has been thoroughly cleaned and prepared after merging multiple sources.
(2) ai_job_market_cleaned.csv: Dataset have undergone initial cleaning but have not yet been uniformly processed.
(3) clustered_jobs.csv: The core output of cluster analysis.
(4) cluster_summary_report.csv: Cluster summary report. Grouped by cluster_id, macro statistics for each cluster are provided.
(5) cluster_features.csv: Explain the "characteristics" of each cluster.
(6) job_title_statistics.csv: Ananlysis that is grouped by "job title".
(7) opportunity_metrics.csv: Ananlysis that is grouped by "opportunity_metrics".

### 7. `feature_engineering/`

Contains the Jupyter Notebook used for feature engineering and analysis. This stage involves creating new, meaningful features from the existing data to improve the results of subsequent modeling and analysis (like clustering and time-series analysis).

### 8. `interactive interface/assets/`

Contains all the pictures for our interactive interface.

### 9. `interactive interface/`

Used to create and run a Python Web application:
(1) dashboard.py: Create and run an interactive Web dashboard.
(2) List all the Python libraries required to run the dashboard.py script and their version numbers.

### 10. `outputs/`

Used to store output files generated during the analysis, primarily charts and visualizations (as `.png` images). It is organized into subfolders (e.g., `Member2_figure`) to hold images produced by different analysis modules.

### 11. `requirements/`

Contains requirement documents and specifications (as `.pdf` files) related to the project. These files define the project's goals, scope, and deliverables, such as the project proposal and final project requirements.

### 12. `Time_Series_Analysis&Visualization_Report/`

Contains the Jupyter Notebook ( `.ipynb`) for analysis and related data ( `.csv`) used for time-series analysis and visualization. This part of the analysis focuses on studying the trends of AI job roles and skill demands over time.
