import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Mock dataset
data = pd.DataFrame({
    'time_spent': [30, 5, 12, 50, 3, 40, 8, 60],
    'pages_visited': [5, 1, 2, 7, 1, 6, 2, 9],
    'country': ['US', 'IN', 'US', 'UK', 'IN', 'US', 'UK', 'IN'],
    'is_returning_user': [1, 0, 1, 1, 0, 1, 0, 1],
    'purchase': [1, 0, 0, 1, 0, 1, 0, 1]
})

# Backup original country for display
data['original_country'] = data['country']

# One-hot encoding (drop US to avoid multicollinearity)
data = pd.get_dummies(data, columns=['country'], drop_first=True)

# Feature engineering
data['engagement_score'] = data['time_spent'] * data['pages_visited']

# Features and label
X = data.drop(['purchase', 'original_country'], axis=1)
y = data['purchase']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and Scaler saved as model.pkl & scaler.pkl")