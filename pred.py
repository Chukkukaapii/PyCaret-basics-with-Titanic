import pandas as pd
from pycaret.classification import *
import seaborn as sns

# Step 1: Load the Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())

# Step 2: Initialize PyCaret and set up the experiment
exp1 = setup(data=titanic, target='survived', session_id=123, 
             categorical_features=['sex', 'embarked', 'class', 'who', 'deck', 'embark_town', 'alive', 'alone'], 
             numeric_features=['age', 'fare'], 
             ignore_features=['age'], 
             preprocess=True, 
             remove_outliers=True, 
             handle_unknown_categorical=True)

# Step 3: Compare different models and select the best one
best_model = compare_models()

# Step 4: Finalize the best model (i.e., train it on the entire dataset)
final_model = finalize_model(best_model)

# Step 5: Save the trained model to a file
save_model(final_model, 'titanic_survival_model')

# Step 6: Load the saved model
model = load_model('titanic_survival_model')

# Step 7: Make predictions on the dataset (for demonstration)
predictions = predict_model(model, data=titanic)

# Display predictions (true label vs predicted label)
print(predictions[['survived', 'Label']])

# Step 8: Evaluate the model using PyCaret’s evaluation plots
evaluate_model(final_model)

# Step 9: Tune hyperparameters of the best model (optional, for better performance)
tuned_model = tune_model(final_model)

# Step 10: Evaluate the tuned model
evaluate_model(tuned_model)

# Step 11: Save the tuned model (optional)
save_model(tuned_model, 'tuned_titanic_model')

# Step 12: Reload the tuned model and make predictions (if needed)
tuned_model = load_model('tuned_titanic_model')
tuned_predictions = predict_model(tuned_model, data=titanic)

# Display tuned predictions
print(tuned_predictions[['survived', 'Label']])

# Optional: Print the final model details
print(final_model)
