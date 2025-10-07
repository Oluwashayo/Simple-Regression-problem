# Regression Model Comparison

This repository compares multiple regression approaches using a dataset from [Huggingface Datasets](https://huggingface.co/datasets/ketan0/test_regression_preds).  
It includes classical models (Linear, Ridge, Polynomial), **a Feedforward Neural Network (FNN)** implemented in Keras, evaluation, visualizations.


## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Training & Evaluation Workflow](#training--evaluation-workflow)

## Project Overview

The main goals are to:

1. Load and inspect the dataset from Hugging Face.
2. Train and compare several regression algorithms:
   - Linear Regression
   - Ridge Regression (α = 0.1, 1.0, 10.0)
   - Polynomial Regression (degree 2, 3)
   - Ridge + Polynomial features
   - **Feedforward Neural Network (FNN)** (Keras)
3. Evaluate models using MSE, RMSE, MAE, and R².
4. Visualize results and automatically select the best model (by validation MSE).
5. Perform k-fold cross-validation for the chosen model.

## Key Features

- Pipeline-based preprocessing (scaling, polynomial features)
- Train / validation / test split (64% / 16% / 20%)
- Automatic best-model selection (by Val MSE)
- 5-fold cross-validation on best model
- Keras FNN with callbacks (EarlyStopping, ModelCheckpoint)
- Model & artifact saving (joblib / Keras `model.save`)

## Training & Evaluation Workflow

1. Inspect dataset (shape, missing values, statistics)
2. Split data → Train, Validation, Test
3. Build multiple regression models
4. Train & Evaluate each model
5. Compare metrics (MSE, RMSE, MAE, R²)
6. Select the best model (lowest validation MSE)
7. Cross-validate the best model (5-fold)
