# Global Weather Trend Forecasting

Tech Assessment for Data Scientist / Analyst — submitted to **PM Accelerator**.

> **PM Accelerator Mission:** *(paste the official mission statement here, copied
> verbatim from [pmaccelerator.io](https://www.pmaccelerator.io/) or the
> [PM Accelerator LinkedIn page](https://www.linkedin.com/company/product-manager-accelerator/) —
> required for submission.)*

## Project Overview

This project analyzes the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
dataset — daily weather and air-quality readings for cities worldwide — to:

- Clean and preprocess raw weather data (missing values, outliers, normalization)
- Explore trends, correlations, and patterns via EDA
- Build and evaluate forecasting models, from a Linear Regression baseline up
  through SARIMAX, XGBoost, Random Forest, Prophet, and an ensemble
- Run advanced analyses: anomaly detection, climate patterns, air-quality
  correlation, feature importance, and spatial/geographical visualization

Both the **Basic** and **Advanced** assessment tracks are covered in a single
notebook: `weather_forecasting_assessment.ipynb`.

## Repository Structure

```
.
├── weather_forecasting_assessment.ipynb   # Main notebook (run this in Kaggle/Jupyter)
├── requirements.txt                       # Python dependencies
├── model_comparison_results.csv           # Generated after running the notebook
├── country_climate_summary.csv            # Generated after running the notebook
└── README.md
```

## How to Run

1. **On Kaggle (recommended):**
   - Create a new notebook and attach the
     [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset.
   - Upload/import `weather_forecasting_assessment.ipynb`.
   - (Optional) Turn on **Internet** in notebook Settings to enable Prophet, SHAP,
     and pycountry_convert — the notebook runs fine without them too.
   - Run all cells top to bottom.

2. **Locally:**
   ```bash
   pip install -r requirements.txt
   jupyter notebook weather_forecasting_assessment.ipynb
   ```
   Download the dataset CSV from Kaggle and place it at
   `/kaggle/input/global-weather-repository/GlobalWeatherRepository.csv`, or edit
   the `DATA_PATH` variable in the notebook to point at your local copy.

## Methodology Summary

| Stage | What was done |
|---|---|
| **Data Cleaning** | Median/mode imputation for numeric/categorical columns, duplicate removal, IQR-based outlier capping, Min-Max scaling |
| **EDA** | Correlation heatmap, temperature & precipitation distributions, daily trend lines, country-level boxplots |
| **Basic Forecasting** | Calendar + lag features built from `last_updated`; chronological train/test split; Linear Regression baseline; MAE/RMSE/MAPE/R² |
| **Anomaly Detection** | Z-score (univariate) and Isolation Forest (multivariate) |
| **Multi-Model Forecasting** | SARIMAX, XGBoost, Random Forest, Prophet (optional), and an averaging ensemble |
| **Unique Analyses** | Seasonal climate patterns by country, air-quality vs weather correlation, three feature-importance methods (impurity-based, permutation, SHAP), geo-scatter and choropleth maps, country/continent comparison |

## Results

See `model_comparison_results.csv` for the full metric table after running the
notebook. *(Fill in your top-line takeaways here once you have real results —
e.g. best model, RMSE achieved, strongest predictive feature.)*

## Limitations

- The dataset is a snapshot-style daily series rather than a multi-year history,
  which limits how much long-term seasonality a model can learn.
- Air-quality fields are point estimates and may have uneven location coverage.

## Demo Video

https://drive.google.com/file/d/1-Kf7yHnW4IEonmIODqPW6fb0JALqedSM/view?usp=drive_link
# weather_forecasting
