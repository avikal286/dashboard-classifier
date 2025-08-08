# dashboard-classifier
Built a Streamlit web app for CSV upload, automated cleaning, EDA, and visualization.  Added interactive filtering, summary statistics, and real-time classification predictions.
# CSV Analyzer Web App

This project is a **Streamlit** web application that allows users to:
- Upload CSV files
- Perform automated data cleaning
- Conduct exploratory data analysis (EDA)
- Generate interactive visualizations
- Apply interactive filtering
- View summary statistics
- Make real-time classification predictions

## Features
- **CSV Upload** – Easily upload CSV files for analysis.
- **Automated Cleaning** – Handles missing values, duplicates, and data type conversions.
- **EDA** – Automatically generates summaries, distributions, and correlations.
- **Interactive Visualizations** – Histograms, boxplots, scatter plots, and heatmaps.
- **Filtering** – Filter data interactively to refine results.
- **Summary Statistics** – Key metrics for numerical and categorical columns.
- **Classification Predictions** – Train or use a pre-trained model for real-time predictions.

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn / Plotly

## Installation
```bash
# Clone the repository
git clone <repo-url>
cd <repo-folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
