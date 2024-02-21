# Heart Disease Classification Project

## Overview

This project focuses on building a classification model to predict heart disease based on various health-related features. The analysis involves data preprocessing, visualization, handling missing values, normalization, outlier detection, and the creation of new features. Several classification algorithms such as k-Nearest Neighbors (kNN), Support Vector Machine (SVM), Decision Tree, Random Forest, and XGBoost are implemented and evaluated. The project includes model comparison, selection, and performance assessment on test and validation datasets.

## Code Structure

- `data_processing.R`: Contains the R code for loading and preprocessing the heart disease dataset, including handling missing values, normalization, outlier detection, and feature engineering.
- `classification_models.R`: Implements various classification models (kNN, SVM, Decision Tree, Random Forest, XGBoost), performs model evaluation, and compares their performance.
- `model_selection.R`: Selects the best-performing model based on accuracy and other metrics.
- `new_data_prediction.R`: Demonstrates how to use the selected model for making predictions on new data.

## Prerequisites

- R (version 4.1.2)
- R libraries: ggplot2, caret, pROC, PRROC, randomForest, xgboost (install using `install.packages("package_name")`)
- kalp.csv

## Usage

1. Open R or RStudio.
2. Run the scripts in the order specified above (`data_processing.R`, `classification_models.R`, `model_selection.R`, `new_data_prediction.R`).

## Results and Visualization

The project includes visualization of model performance using ROC and Precision-Recall curves. The best model is selected based on accuracy and other evaluation metrics.

## Contributors

- niluse