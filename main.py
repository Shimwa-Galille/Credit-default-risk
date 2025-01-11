import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'credit_risk_dataset.csv'  # Adjust the file path
data = pd.read_csv(file_path)

# Separate numeric and categorical columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Fill missing values
# For numeric columns, fill with the mean
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# For categorical columns, fill with the mode
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical columns for further processing
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col])

# Compute correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Data Summary
print("\nData Summary:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values or drop rows/columns as needed
# Example: Filling missing numerical values with median
data.fillna(data.median(numeric_only=True), inplace=True)

# Exploratory Data Analysis (EDA)
print("\nStatistical Summary:")
print(data.describe())

# Visualize correlations
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Identify target and features
# Assuming 'default' is the target variable and the rest are predictors
target = 'default'  # Replace with the actual target column name
features = [col for col in data.columns if col != target]

X = data[features]
y = data[target]

# Convert categorical variables to numeric (if any)
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a simple classification model (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()
