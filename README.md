# Regression Model Comparison & Hyperparameter Tuning

A comprehensive regression analysis pipeline that compares multiple models (Linear, Ridge, Polynomial, Neural Network) on a dataset from [Huggingface Datasets](https://huggingface.co/datasets/ketan0/test_regression_preds), performs hyperparameter tuning, and follows best practices for model evaluation with proper train-validation-test splits.

## Project Overview

This project demonstrates a **machine learning workflow** for regression tasks. It started as an exploration of different regression approaches and evolved into a full pipeline that includes:

- Synthetic data generation
- Initial model comparison across multiple regression techniques
- Ridge regression hyperparameter tuning
- Proper train/validation/test splitting methodology
- Model serialization for deployment
- Comprehensive performance visualization

**Developer Note**: I built this to showcase the importance of proper data splitting and avoiding data leakage. The code intentionally uses a professional 60/20/20 train/val/test split and only touches the test set once at the very end for unbiased evaluation.

## Dataset

The project uses **synthetic data** to ensure reproducibility and controlled experimentation:

- **890 samples** with 2 features
- Target generated using: `y = 4 + 3*x₁ + 5*x₂ + noise`
- Features are uniformly distributed in the range [0, 2]
- Random noise added to simulate real-world variability

The synthetic nature allows for easy validation of model behavior and serves as a template for real-world datasets.

## Model & Approach

### Models Evaluated

1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization with alpha tuning
3. **Polynomial Regression** (degrees 2, 4, 10) - Capturing non-linear relationships
4. **Neural Network (MLP)** - 3-layer architecture (100, 50, 25 neurons)

### Two-Phase Analysis

#### Phase 1: Initial Model Comparison
- Quick 80/20 train/validation split
- All models trained and compared on validation metrics (MSE, RMSE, R²)
- Overfitting detection based on train-val performance gap
- 5-fold cross-validation on best model

#### Phase 2: Evaluation
- **60% training** / **20% validation** / **20% test** split
- Ridge hyperparameter tuning (alpha: 0.01 to 100)
- Final model retrained on combined train+val data
- **Test set used only once** for final unbiased evaluation

## Key Features

- **Proper data splitting** to prevent leakage
- **Feature scaling** with StandardScaler (fitted on training data only)
- **Pipeline architecture** for polynomial models
- **Hyperparameter tuning** with visual comparison
- **Model persistence** using joblib
- **Comprehensive visualizations** for all metrics
- **Overfitting detection** with configurable thresholds
- **Cross-validation** for robust performance estimates

## Results

The analysis provides:

- Side-by-side comparison of all models across MSE, RMSE, and R² metrics
- Identification of overfitting in high-degree polynomial models
- Optimal Ridge alpha parameter through systematic tuning
- Final test set performance proving model generalization
