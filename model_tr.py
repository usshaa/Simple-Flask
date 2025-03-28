import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a simple dataset
np.random.seed(42)
n_samples = 200

data = {
    "Age": np.random.randint(20, 60, n_samples),  # Age in years
    "Experience": np.random.randint(1, 40, n_samples),  # Years of experience
    "Salary": np.random.randint(30000, 120000, n_samples),  # Annual salary in dollars
    "House_Price": np.random.randint(100000, 500000, n_samples)  # Target: House price in dollars
}

df = pd.DataFrame(data)

df.to_csv("house_price_data.csv", index=False)  # Save the dataset to a CSV file

# Step 2: Data Cleaning
# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()

# Step 3: Exploratory Data Analysis (EDA)
print("Dataset Overview:")
print(df.describe())

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df)
plt.show()

# Step 4: Feature Engineering
# Add a new feature: Age-to-Experience ratio
df["Age_Experience_Ratio"] = df["Age"] / (df["Experience"] + 1)

# Step 5: Data Transformation
# Separate features and target
X = df.drop(columns=["House_Price"])
y = df["House_Price"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Step 10: Save the trained model as a .pkl file
model_filename = "linear_regression_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")

# Save the scaler
scaler_filename = "scaler.pkl"
with open(scaler_filename, "wb") as file:
    pickle.dump(scaler, file)

print(f"Scaler saved as {scaler_filename}")

