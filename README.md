# Titanic Survival Prediction using PyCaret

This project uses the **Titanic dataset** to predict the survival of passengers based on their attributes such as age, gender, class, and others, using **PyCaret** for automated machine learning.

The model compares different classification models and selects the best one, tunes hyperparameters, evaluates performance, and saves the final model for later use.

## Overview

The Titanic dataset contains information about passengers on the Titanic ship. The goal is to predict whether a passenger survived based on features such as:
- Age
- Gender
- Class
- Embarked
- Fare
- and more...

This project demonstrates the use of **PyCaret** for setting up an end-to-end machine learning pipeline for classification tasks with minimal code.

## Technologies

- **PyCaret**: A low-code machine learning library for fast experimentation.
- **Seaborn**: A Python visualization library that also provides easy access to datasets.
- **Pandas**: For data manipulation.
- **Scikit-learn**: For model evaluation and performance metrics.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/titanic-survival-pycaret.git
    cd titanic-survival-pycaret
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Python script to train and evaluate the model:
    ```bash
    python titanic_classification.py
    ```

2. The script will:
    - Load the Titanic dataset.
    - Preprocess the data using **PyCaret**.
    - Compare different classification models.
    - Select the best model based on performance metrics.
    - Tune the model and evaluate it.
    - Save the final trained model as `titanic_survival_model.pkl`.

3. The trained model can be used later for making predictions:
    ```bash
    python make_predictions.py
    ```

## Steps

1. **Data Preprocessing**: Using PyCaret's `setup()` function, the dataset is preprocessed, which includes handling missing values, encoding categorical variables, and removing outliers.
2. **Model Comparison**: The `compare_models()` function compares several classification models and selects the best one based on performance metrics like accuracy and AUC.
3. **Hyperparameter Tuning**: Optionally, the best model can be fine-tuned using `tune_model()` to improve performance.
4. **Model Finalization**: The model is finalized with `finalize_model()` and saved for future use.
5. **Prediction**: The trained model can be used to make predictions on new or unseen data.

## Results

After training the model, the evaluation plots (such as confusion matrix and ROC curve) are displayed, and the final model's performance is assessed. The tuned model is saved as `tuned_titanic_model.pkl`.

