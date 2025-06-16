import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("car_data_set.csv")

# Encode categorical variables
le_brand = LabelEncoder()
le_fuel = LabelEncoder()

df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])

# Define features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save models and encoders
with open("model_lr.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("model_rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("brand_encoder.pkl", "wb") as f:
    pickle.dump(le_brand, f)

with open("fuel_encoder.pkl", "wb") as f:
    pickle.dump(le_fuel, f)
