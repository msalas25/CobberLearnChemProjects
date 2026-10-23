
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Step 1: Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Step 2: Convert the dataset into a DataFrame
df = housing.frame

# Step 3: Look at the feature descriptions
print("--- FEATURE DESCRIPTIONS ---")
print(housing.DESCR)

# Step 4: Display the first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Step 5: Display basic statistics
print("\n--- BASIC STATISTICS ---")
print(df.describe())

# Step 6: Check the dataset size
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Step 7: Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\nDataset exploration completed!")



# STEP 3: TRAIN MULTIPLE REGRESSION MODELS

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Separate the input features and target variable
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the four regression models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Support Vector Regression": make_pipeline(
        StandardScaler(), SVR()
    )
}

# Store the results
results = {}

print("\n--- REGRESSION MODEL RESULTS ---")

# Train and test each model
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    results[name] = r2

    print(f"{name} R2 Score: {r2:.4f}")

# Find the best-performing model
best_model_name = max(results, key=results.get)

print("\n--- BEST MODEL ---")
print(f"Best Model: {best_model_name}")
print(f"R2 Score: {results[best_model_name]:.4f}")



# STEP 4: DATA VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Save graphs in the CaliforniaHousing folder
output_folder = Path(__file__).resolve().parent

# Create histograms for the eight input features
X.hist(figsize=(14, 10), bins=30, edgecolor="black")

plt.suptitle("California Housing Feature Distributions")
plt.tight_layout()

plt.savefig(
    output_folder / "housing_histograms.png",
    dpi=300
)

plt.close()

print("Histograms saved successfully!")

# Create a correlation heatmap
plt.figure(figsize=(11, 8))

correlation = df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("California Housing Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    output_folder / "housing_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("Correlation heatmap saved successfully!")
print("Step 4 completed!")



# STEP 5: PREDICT HOUSING PRICES

# Use the best-performing model from Step 3
best_model = models[best_model_name]

# Create hypothetical California neighborhoods
new_houses = pd.DataFrame([
    {
        "MedInc": 3.0,
        "HouseAge": 20,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.24
    },
    {
        "MedInc": 6.0,
        "HouseAge": 20,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.24
    },
    {
        "MedInc": 6.0,
        "HouseAge": 40,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 34.05,
        "Longitude": -118.24
    },
    {
        "MedInc": 6.0,
        "HouseAge": 40,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1200,
        "AveOccup": 3.0,
        "Latitude": 37.77,
        "Longitude": -122.42
    }
])

# Ensure the feature order matches the training data
new_houses = new_houses[X.columns]

# Predict using the best model
predicted_prices = best_model.predict(new_houses)

print("\n--- HOUSING PRICE PREDICTIONS ---")

for i, price in enumerate(predicted_prices):
    print(f"Example {i + 1}: ${price * 100000:,.2f}")

print("\nStep 5 completed!")



# STEP 6: FEATURE IMPORTANCE AND MODEL EVALUATION

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

# Feature importance from the Random Forest model
importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importances)

# Plot feature importance
plt.figure(figsize=(10, 6))
importances.plot(kind="bar")

plt.title("Random Forest Feature Importance")
plt.ylabel("Importance Score")
plt.xlabel("Housing Features")
plt.tight_layout()

plt.savefig(
    output_folder / "housing_feature_importance.png",
    dpi=300
)
plt.close()

print("Feature importance graph saved!")

# Compare actual and predicted housing values
y_pred = best_model.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Housing Value ($100,000)")
plt.ylabel("Predicted Housing Value ($100,000)")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()

plt.savefig(
    output_folder / "housing_actual_vs_predicted.png",
    dpi=300
)
plt.close()

print("Actual vs predicted graph saved!")
print("Step 6 completed!")



# STEP 7: INTERACTIVE HOUSING PRICE PREDICTIONS

print("\n--- CUSTOM HOUSING PRICE PREDICTOR ---")

while True:
    choice = input("\nWould you like to predict a housing price? (yes/no): ").strip().lower()

    if choice in ("no", "n"):
        print("Thank you for using the housing price predictor!")
        break

    if choice not in ("yes", "y"):
        print("Please enter yes or no.")
        continue

    try:
        med_inc = float(input("Median income (in $10,000 units): "))
        house_age = float(input("Median house age: "))
        avg_rooms = float(input("Average rooms per household: "))
        avg_bedrooms = float(input("Average bedrooms per household: "))
        population = float(input("Block group population: "))
        avg_occupancy = float(input("Average household occupancy: "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))

        custom_house = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": avg_rooms,
            "AveBedrms": avg_bedrooms,
            "Population": population,
            "AveOccup": avg_occupancy,
            "Latitude": latitude,
            "Longitude": longitude
        }])

        custom_house = custom_house[X.columns]

        prediction = best_model.predict(custom_house)[0]

        print(f"\nPredicted Median Housing Value: ${prediction * 100000:,.2f}")

    except ValueError:
        print("Invalid input. Please enter numerical values.")
