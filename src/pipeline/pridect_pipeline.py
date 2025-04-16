import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Define features and target
features = ['fico', 'int.rate', 'dti', 'revol.util', 'inq.last.6mths', 'credit.policy']
target = 'not.fully.paid'

# Load data (replace with your dataset path)
# df = pd.read_csv('your_loan_data.csv')
# For demonstration, create a synthetic dataset
def create_synthetic_data(n_samples=10000):
    data = {
        'fico': np.random.normal(700, 50, n_samples).clip(300, 850),
        'int.rate': np.random.uniform(0.05, 0.25, n_samples),
        'dti': np.random.normal(20, 10, n_samples).clip(0, 50),
        'revol.util': np.random.uniform(0, 100, n_samples),
        'inq.last.6mths': np.random.poisson(1, n_samples).clip(0, 10),
        'credit.policy': np.random.binomial(1, 0.8, n_samples),
        'not.fully.paid': np.random.binomial(1, 0.15, n_samples)  # ~15% defaults
    }
    return pd.DataFrame(data)

df = create_synthetic_data()

# Step 1: Data Preprocessing
def preprocess_data(df, features, target):
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])
    
    # Split features and target
    X = df[features]
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer

X_train, X_test, y_train, y_test, scaler, imputer = preprocess_data(df, features, target)

# Step 2: Model Training
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        'XGBoost': XGBClassifier(random_state=42, scale_pos_weight=sum(y_train == 0) / sum(y_train == 1))
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained.")
    
    return trained_models

models = train_models(X_train, y_train)

# Step 3: Model Evaluation
def evaluate_models(models, X_test, y_test):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'AUC-ROC': auc,
            'Precision': precision,
            'Recall': recall
        })
        
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    return pd.DataFrame(results)

results_df = evaluate_models(models, X_test, y_test)
print("\nModel Performance Summary:")
print(results_df)

# Step 4: Select and Save Best Model
best_model_name = results_df.loc[results_df['AUC-ROC'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Save model, scaler, and imputer
joblib.dump(best_model, 'best_loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
print("Best model, scaler, and imputer saved.")

# Step 5: Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'XGBoost']:
    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(best_model.feature_importances_, index=features)
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.title(f'Feature Importance ({best_model_name})')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig('feature_importance.png')
    plt.close()

# Step 6: Function for New Predictions
def predict_new_data(new_data, model, scaler, imputer):
    # Ensure new_data has the same features
    new_data = new_data[features]
    
    # Handle missing values
    new_data = imputer.transform(new_data)
    
    # Scale features
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)[:, 1]
    
    return predictions, probabilities

# Example: Predict on a new sample
new_sample = pd.DataFrame({
    'fico': [650],
    'int.rate': [0.12],
    'dti': [25],
    'revol.util': [60],
    'inq.last.6mths': [2],
    'credit.policy': [1]
})
pred, prob = predict_new_data(new_sample, best_model, scaler, imputer)
print(f"\nNew Sample Prediction: {'Default' if pred[0] == 1 else 'No Default'} (Probability: {prob[0]:.2f})")