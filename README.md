# Regression Model Comparison Project

This repository demonstrates a **comprehensive comparison of multiple regression models** using a dataset from [Huggingface Datasets](https://huggingface.co/datasets/ketan0/test_regression_preds).  
It includes linear, ridge, and polynomial regression models, along with proper data preprocessing, model evaluation, and visualization.

---

## Project Overview

The goal of this project is to:

1. Load and explore a regression dataset.
2. Build different regression models:
   - Linear Regression  
   - Ridge Regression (α = 0.1, 1.0, 10.0)  
   - Polynomial Regression (Degree 2, 3)  
   - Ridge with Polynomial Features (Degree 2)
3. Compare model performances using key metrics:
   - **Mean Squared Error (MSE)**  
   - **Root Mean Squared Error (RMSE)**  
   - **Mean Absolute Error (MAE)**  
   - **R² Score**
4. Visualize results with plots and identify the best-performing model.
5. Perform **k-fold cross-validation** on the best model for robustness.

---

## Key Features

- Automated **data loading** and **inspection**  
- Proper **train-validation-test split** (64% / 16% / 20%)  
- **Pipeline-based preprocessing** with `StandardScaler` and `PolynomialFeatures`  
- **Comprehensive model comparison** and evaluation  
- **Matplotlib visualizations** for performance comparison  
- **Automatic best model selection** based on validation metrics  
- **Cross-validation** for performance stability  


