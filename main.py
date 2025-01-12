import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'credit_risk_dataset.csv'  # Adjust the file path
data = pd.read_csv(file_path)

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Data Overview
print("\nData Summary:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())
print("\nStatistical Summary:")
print(data.describe())

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing values
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical columns
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=False, cmap="coolwarm")  # Removed annotations for better performance
plt.title("Correlation Matrix")
plt.show()

# Pairplot for relationships
sns.pairplot(data[['person_income', 'loan_amnt', 'loan_int_rate', 'loan_status']], diag_kind='hist', hue='loan_status')
plt.title("Pairplot of Features")
plt.show()

# Histograms for Numeric Features
numeric_features = ['person_income', 'loan_amnt', 'loan_int_rate']
for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=data, x=feature, kde=False, hue='loan_status', bins=20, palette='coolwarm')
    plt.title(f"Distribution of {feature} by Loan Status")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

# Barplots for Categorical Features
categorical_features = ['loan_intent', 'loan_grade']
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=feature, hue='loan_status', palette='viridis')
    plt.title(f"{feature} vs Loan Status")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()

# Violin Plot for Loan Amount
plt.figure(figsize=(8, 6))
sns.violinplot(x='loan_status', y='loan_amnt', data=data, palette='muted')
plt.title("Loan Amount Distribution by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.show()

# Default on File vs Loan Status
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='cb_person_default_on_file', hue='loan_status', palette='plasma')
plt.title("Default on File vs Loan Status")
plt.xlabel("Default on File (Credit Bureau)")
plt.ylabel("Count")
plt.show()

# Boxplots for Loan Percent Income and Credit History Length
boxplot_features = ['loan_percent_income', 'cb_person_cred_hist_length']
for feature in boxplot_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='loan_status', y=feature, data=data, palette='Set2')
    plt.title(f"{feature} vs Loan Status")
    plt.xlabel("Loan Status")
    plt.ylabel(feature)
    plt.show()

# Prepare Data for Modeling
target = 'loan_status'
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
roc_auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance")
plt.show()
