import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load data
df = pd.read_csv("BostonHousing.csv")

# Separate features and target
# The dataset has 'medv' as the target variable.
X = df.drop(columns=['medv'])
y = df['medv']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
regression = LinearRegression()
regression.fit(X_train, y_train)

# Save
pickle.dump(regression, open('regmodel.pkl', 'wb'))
pickle.dump(scaler, open('scaling.pkl', 'wb'))

print("Model and scaler saved successfully with 13 features.")
