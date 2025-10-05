# Machine Learning Regression Analysis

This project demonstrates how to apply and compare **classical regression models** and a **neural network** to a real-world dataset.  
It covers the full machine learning workflow — from dataset loading and preprocessing to model evaluation, cross-validation, and performance comparison.

---

## Project Overview

In this project, we explored the regression dataset from [Hugging Face Datasets](https://huggingface.co/datasets/ketan0/test_regression_preds) and carried out the following tasks:

1. **Implemented classical regression models:**
   - Linear Regression  
   - Ridge Regression  
   - Polynomial Regression (degrees 2, 4, and 10)

2. **Compared training and validation performance** using Mean Squared Error (MSE).

3. **Performed 5-fold Cross-Validation** on the best model (Ridge Regression).

4. **Built a Feedforward Neural Network (FNN)** to solve the same regression task using TensorFlow/Keras.

5. **Compared model accuracies** and interpreted results in terms of bias, variance, and generalization.

---

## Dataset Description

- **Type:** Supervised regression dataset
- **Features (X):** Independent variable(s) used as input  
- **Target (y):** Continuous value to predict  

The dataset was split into:
- **Training Set:** 80%
- **Validation Set:** 20%

This ensures that model evaluation reflects real-world performance on unseen data.
