
import tkinter as tk
from tkinter import messagebox
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# STEP 1: Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# STEP 2: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 3: Train the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")
model.fit(X_train, y_train)
print("Model ready!")

# STEP 4: Create the GUI window
window = tk.Tk()
window.title("California Housing Price Predictor")
window.geometry("480x570")

title = tk.Label(
    window,
    text="California Housing Price Predictor",
    font=("Arial", 16, "bold")
)
title.pack(pady=15)

description = tk.Label(
    window,
    text="Enter neighborhood information to predict housing prices."
)
description.pack(pady=5)

# STEP 5: Create input fields
features = {
    "MedInc": ("Median Income ($10,000 units)", "5.0"),
    "HouseAge": ("House Age (years)", "25"),
    "AveRooms": ("Average Rooms", "5"),
    "AveBedrms": ("Average Bedrooms", "1"),
    "Population": ("Population", "1500"),
    "AveOccup": ("Average Occupancy", "3"),
    "Latitude": ("Latitude", "34.05"),
    "Longitude": ("Longitude", "-118.24")
}

entries = {}

form = tk.Frame(window)
form.pack(pady=10)

for row, (feature, (label_text, default)) in enumerate(features.items()):
    label = tk.Label(form, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=7, sticky="w")

    entry = tk.Entry(form, width=18)
    entry.grid(row=row, column=1, padx=10, pady=7)
    entry.insert(0, default)

    entries[feature] = entry

# STEP 6: Predict housing price
def predict_price():
    try:
        values = {}

        for feature, entry in entries.items():
            values[feature] = float(entry.get())

        # Check for invalid or unrealistic inputs
        if (
            values["MedInc"] <= 0
            or values["HouseAge"] <= 0
            or values["AveRooms"] <= 0
            or values["AveBedrms"] <= 0
            or values["Population"] <= 0
            or values["AveOccup"] <= 0
            or not 32 <= values["Latitude"] <= 42.1
            or not -125 <= values["Longitude"] <= -114
        ):
            messagebox.showerror(
                "Invalid Input",
                "Please enter positive housing values and valid California coordinates."
            )
            return

        # Convert inputs into a DataFrame
        input_data = pd.DataFrame([values], columns=X.columns)

        # Predict the median housing value
        prediction = model.predict(input_data)[0] * 100000

        result_label.config(
            text=f"Predicted Housing Value: ${prediction:,.2f}"
        )

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter numbers only in all fields."
        )

# STEP 7: Reset all input fields
def clear_fields():
    for feature, entry in entries.items():
        entry.delete(0, tk.END)
        entry.insert(0, features[feature][1])

    result_label.config(text="Predicted Housing Value: ")

# STEP 8: Add buttons
predict_button = tk.Button(
    window,
    text="Predict Price",
    command=predict_price,
    width=20
)
predict_button.pack(pady=8)

clear_button = tk.Button(
    window,
    text="Reset Fields",
    command=clear_fields,
    width=20
)
clear_button.pack(pady=5)

# Display predicted price
result_label = tk.Label(
    window,
    text="Predicted Housing Value: ",
    font=("Arial", 12, "bold")
)
result_label.pack(pady=15)

# Keep the GUI window open
window.mainloop()
