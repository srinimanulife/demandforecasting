# Demand Forecasting Analysis

This project provides a comprehensive suite of tools for demand forecasting, featuring advanced data analysis and a variety of predictive models. It includes notebooks for feature engineering, promotion detection, and experimentation with multiple forecasting techniques tailored to different types of sales data.

## Features

- **Promotion Detection:** A custom algorithm to identify promotional periods from sales data based on significant increases in purchase volume. Includes an explainability feature to detail how each promotion was detected.
- **Advanced Feature Engineering:** Techniques for creating insightful features from time series data, such as temporal encodings, price sensitivity, and lag effects.
- **Diverse Forecasting Models:** A collection of models to handle various data patterns:
    - **Sparse Data:** Croston's Method, Zero-Inflated Models.
    - **Volatile Data:** LightGBM, Prophet, Random Forest.
    - **Stable Data:** ARIMA, Exponential Smoothing (ETS).
- **Experimentation Framework:** A dedicated notebook to test and compare the performance of different models on synthetic data that mimics sparse, volatile, and stable sales patterns.

## File Structure

- **`sales.csv`, `revenue.csv`**: Raw data files containing sales and purchase information.
- **`Promotion_Detection.ipynb`**: A notebook that implements and explains the logic for detecting promotional events from the sales data.
- **`Advanced_Feature_Engineering.ipynb`**: Integrates the promotion detection feature into a larger feature engineering pipeline and trains a forecasting model.
- **`Advanced_Forecasting_Techniques.ipynb`**: A standalone notebook for experimenting with a wide range of forecasting models on different types of synthetic data (sparse, volatile, stable).
- **`forecasting_models.py`**: A Python script containing the implementation of all forecasting models used in the experimentation notebook.
- **`test_promotion_detection.py`**: Unit tests for the promotion detection algorithm.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/srinimanulife/demandforecasting.git
cd demandforecasting
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies

This project requires several Python libraries. The `Advanced_Forecasting_Techniques.ipynb` notebook requires additional libraries, including `tensorflow`.

```bash
# Install core libraries
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels

# Install libraries for advanced forecasting notebook
pip install prophet lightgbm tensorflow
```

### 4. Run the Notebooks

You can now launch Jupyter and run the notebooks to explore the project's features.

```bash
# Launch Jupyter Lab or Jupyter Notebook
jupyter lab
```

- **To understand promotion detection:** Start with `Promotion_Detection.ipynb`.
- **To see feature engineering in action:** Open `Advanced_Feature_Engineering.ipynb`.
- **To experiment with different models:** Run the `Advanced_Forecasting_Techniques.ipynb` notebook.