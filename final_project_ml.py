import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Loading data
train_data = pd.read_csv(
    "/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv"
)
test_data = pd.read_csv(
    "/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv"
)

# Separation into features and target variable
X = train_data.drop("y", axis=1)
y = train_data["y"]

# Determination of numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Create a pipeline for numeric variables
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Creating a pipeline for categorical variables
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Merge pipelines to process all variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Creating a pipeline with feature processing, SMOTE and RandomForest classifier
rf = RandomForestClassifier(random_state=42)

model = ImbPipeline(
    [
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", rf),
    ]
)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="balanced_accuracy")
grid_search.fit(X, y)

# Cross-validation results
cv_scores = grid_search.best_score_
best_params = grid_search.best_params_
print(f"Best cross-validation score: {cv_scores:.3f}")
print(f"Best parameters: {best_params}")

# Training the best model on all training data
best_model = grid_search.best_estimator_
best_model.fit(X, y)

# Predictions for the test set
test_predictions = best_model.predict(test_data)

# Create file to send results
submission = pd.DataFrame({"index": test_data.index, "y": test_predictions})
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("The submission.csv file has been created successfully.")
