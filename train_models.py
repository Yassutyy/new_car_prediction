import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("car_data_set.csv")

# Create car age
df["Car_Age"] = 2025 - df["Year"]
df.drop("Year", axis=1, inplace=True)

# Encode categorical variables
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Fuel"] = le_fuel.fit_transform(df["Fuel"])

# Features and Target
X = df[["Brand", "Car_Age", "KM_Driven", "Fuel"]]
y = df["Selling_Price"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Train Random Forest
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)

# Save models and encoders
pickle.dump(model_lr, open("model_lr.pkl", "wb"))
pickle.dump(model_rf, open("model_rf.pkl", "wb"))
pickle.dump(le_brand, open("brand_encoder.pkl", "wb"))
pickle.dump(le_fuel, open("fuel_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
