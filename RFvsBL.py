import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
import statsmodels.api as sm

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ---------------------------
# 1. Data Loading and Preprocessing
# ---------------------------

# Load the dataset
df = pd.read_excel('/content/New.xlsx')  # Ensure the correct path

# Define features
numeric_features = [
    'Travel_Time_minutes', 'Travel_Schedule', 'Reliable_Travel_Time',
    'Safety', 'Comfort', 'Cost_Effective'
]
ordinal_features = [
    'Gender', 'Income', 'Station_Closer_Origin', 'Station_Closer_Destination',
    'Age', 'Purpose', 'Frequency_of_Travel', 'Time_of_Travel', 'Waiting_Time'
]

# Define preprocessing for numeric columns (imputation and scaling)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for ordinal columns (ordinal encoding)
ordinal_transformer = OrdinalEncoder(categories=[
    ['Female', 'Male'],  # Gender
    ['<6000', '6000-8000', '8000-10000', '>10000'],  # Income
    ['No', 'Yes'],  # Station_Closer_Origin
    ['No', 'Yes'],  # Station_Closer_Destination
    ['18-40', '40-60', '>60'],  # Age
    ['Work/Business', 'Academic Purpose', 'Tourism', 'Visiting Family/Friends'],  # Purpose
    ['<=1', '2-3', '4-5', '>=6'],  # Frequency_of_Travel
    ['Morning', 'Afternoon', 'Evening', 'Night'],  # Time_of_Travel
    ['<=5 minutes', '6-10 minutes', '11-15 minutes', '>=16 minutes']  # Waiting_Time
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features)
    ]
)

# Split the dataset into features (X) and target variable (y)
X = df.drop('Preferred_Mode', axis=1)  # Ensure 'Preferred_Mode' is the correct target column
y = df['Preferred_Mode']

# ---------------------------
# 2. Multicollinearity Check using VIF
# ---------------------------

# Apply preprocessing to compute VIF
X_processed = pd.DataFrame(preprocessor.fit_transform(X))

# Add constant for VIF calculation
X_processed_with_const = sm.add_constant(X_processed)

# Calculate VIF for each feature
vif_data = pd.Series(
    [sm.OLS(X_processed_with_const.iloc[:, i], X_processed_with_const.drop(X_processed_with_const.columns[i], axis=1)).fit().rsquared
     for i in range(X_processed_with_const.shape[1])],
    index=X_processed_with_const.columns
)
vif = 1 / (1 - vif_data)
print("Variance Inflation Factor (VIF) for each feature:\n", vif)

# Identify features with VIF > 10
high_vif_columns = vif[vif > 10].index.tolist()
if 'const' in high_vif_columns:
    high_vif_columns.remove('const')  # Remove 'const' as it's not an original feature
print(f"Features with VIF > 10: {high_vif_columns}")

# Remove highly collinear features from X
X_filtered = X.drop(columns=high_vif_columns)
print(f"Shape after removing high VIF features: {X_filtered.shape}")

# Update preprocessing after removing high VIF features
# Re-define numeric and ordinal features based on X_filtered
numeric_features = [feature for feature in numeric_features if feature not in high_vif_columns]
ordinal_features = [feature for feature in ordinal_features if feature not in high_vif_columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features)
    ]
)

# ---------------------------
# 3. Define Cross-Validation Strategy
# ---------------------------

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------
# 4. Random Forest Pipeline
# ---------------------------

# Define the Random Forest model
rf_model = RandomForestClassifier(oob_score=True, random_state=42)

# Hyperparameter tuning for Random Forest
rf_param_distributions = {
    'classifier__n_estimators': [50, 100, 200, 500, 1000],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20, 30, 50],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

rf_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', rf_model)
])

rf_random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=rf_param_distributions,
    n_iter=50,
    cv=kfold,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

# Fit Random Forest model
print("Training Random Forest...")
rf_random_search.fit(X_filtered, y)
print("Random Forest training completed.")

# Best Random Forest model
best_rf_model = rf_random_search.best_estimator_
print(f"Best Random Forest parameters: {rf_random_search.best_params_}")

# OOB accuracy score
oob_score = best_rf_model.named_steps['classifier'].oob_score_
print(f"Random Forest Out-of-Bag (OOB) Accuracy: {oob_score:.2f}")

# Testing accuracy score using cross_val_predict
rf_y_pred = cross_val_predict(best_rf_model, X_filtered, y, cv=kfold)
rf_test_accuracy = accuracy_score(y, rf_y_pred)
rf_f1 = f1_score(y, rf_y_pred, average='weighted')
print(f"Random Forest Testing Accuracy: {rf_test_accuracy:.2f}")
print(f"Random Forest F1-Score: {rf_f1:.2f}")

# Check for overfitting
if abs(oob_score - rf_test_accuracy) > 0.05:
    print("Random Forest model might be overfitting.")
else:
    print("Random Forest model does not seem to be overfitting.")

# Confusion Matrix for Random Forest
rf_conf_matrix = confusion_matrix(y, rf_y_pred)

# Plot Confusion Matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report for Random Forest
print("Classification Report for Random Forest:\n")
print(classification_report(y, rf_y_pred))

# Feature Importance for Random Forest
feature_importances_rf = best_rf_model.named_steps['classifier'].feature_importances_
processed_features = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
feature_names = processed_features  # Assuming no one-hot encoding; adjust if necessary

# Create a DataFrame for feature importances
rf_importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importances for Random Forest
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=rf_importances_df)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Logistic Regression Pipeline
# ---------------------------

# Encode the target variable if it's binary
if y.nunique() == 2:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
else:
    y_encoded = y.copy()  # For multiclass, LabelEncoder can still be used if needed

# Define the Logistic Regression model
logit_model = LogisticRegression(solver='saga', max_iter=1000, random_state=42, class_weight='balanced')

# Hyperparameter tuning for Logistic Regression
logit_param_distributions = {
    'classifier__penalty': ['l2', 'elasticnet'],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    'classifier__l1_ratio': [0, 0.5, 1]  # Only used if penalty='elasticnet'
}

logit_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('sampler', SMOTE(random_state=42)),
    ('classifier', logit_model)
])

logit_random_search = RandomizedSearchCV(
    logit_pipeline,
    param_distributions=logit_param_distributions,
    n_iter=20,  # Reduced n_iter for quicker execution; adjust as needed
    cv=kfold,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

# Fit Logistic Regression model
print("Training Logistic Regression...")
logit_random_search.fit(X_filtered, y_encoded)
print("Logistic Regression training completed.")

# Best Logistic Regression model
best_logit_model = logit_random_search.best_estimator_
print(f"Best Logistic Regression parameters: {logit_random_search.best_params_}")

# Testing accuracy score using cross_val_predict
logit_y_pred = cross_val_predict(best_logit_model, X_filtered, y_encoded, cv=kfold)
logit_test_accuracy = accuracy_score(y_encoded, logit_y_pred)
logit_f1 = f1_score(y_encoded, logit_y_pred, average='weighted')
print(f"Logistic Regression Testing Accuracy: {logit_test_accuracy:.2f}")
print(f"Logistic Regression F1-Score: {logit_f1:.2f}")

# Confusion Matrix for Logistic Regression
logit_conf_matrix = confusion_matrix(y_encoded, logit_y_pred)

# Plot Confusion Matrix for Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(logit_conf_matrix, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report for Logistic Regression
print("Classification Report for Logistic Regression:\n")
print(classification_report(y_encoded, logit_y_pred))

# Feature Coefficients for Logistic Regression
if hasattr(best_logit_model.named_steps['classifier'], 'coef_'):
    coef = best_logit_model.named_steps['classifier'].coef_[0]
    logit_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef
    }).sort_values(by='Coefficient', key=lambda x: x.abs(), ascending=False)

    # Plot Feature Coefficients for Logistic Regression
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=logit_importances_df)
    plt.title('Feature Coefficients - Logistic Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
else:
    print("The Logistic Regression model does not have coefficients to display.")

# ---------------------------
# 6. Model Comparison
# ---------------------------

print(f"Random Forest - Accuracy: {rf_test_accuracy:.2f}, F1-Score: {rf_f1:.2f}")
print(f"Logistic Regression - Accuracy: {logit_test_accuracy:.2f}, F1-Score: {logit_f1:.2f}")

if rf_f1 > logit_f1:
    print("Random Forest performs better based on F1-Score.")
elif rf_f1 < logit_f1:
    print("Logistic Regression performs better based on F1-Score.")
else:
    print("Both models have the same F1-Score.")

# ---------------------------
# 7. ROC Curve and AUC (if Binary Classification)
# ---------------------------

if y.nunique() == 2:
    # ROC for Random Forest
    rf_y_probs = cross_val_predict(best_rf_model, X_filtered, y, cv=kfold, method='predict_proba')[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_encoded, rf_y_probs)
    rf_roc_auc = auc(rf_fpr, rf_tpr)

    # ROC for Logistic Regression
    logit_y_probs = cross_val_predict(best_logit_model, X_filtered, y_encoded, cv=kfold, method='predict_proba')[:, 1]
    logit_fpr, logit_tpr, _ = roc_curve(y_encoded, logit_y_probs)
    logit_roc_auc = auc(logit_fpr, logit_tpr)

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, color='darkorange', lw=2, label=f'Random Forest ROC curve (area = {rf_roc_auc:.2f})')
    plt.plot(logit_fpr, logit_tpr, color='blue', lw=2, label=f'Logistic Regression ROC curve (area = {logit_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Print AUC scores
    print(f"Random Forest AUC: {rf_roc_auc:.2f}")
    print(f"Logistic Regression AUC: {logit_roc_auc:.2f}")
else:
    print("ROC Curve is only applicable for binary classification.")
