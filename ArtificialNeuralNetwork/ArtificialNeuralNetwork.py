
# Chapter 13: Artificial Neural Network
# Step 1: Build a Small Chemistry Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create the chemistry dataset
data = {
    "Compound": [
        "Methane", "Water", "Propane", "Ethanol",
        "Formic Acid", "Acetic Acid", "Butane",
        "Acetone", "Benzene", "Toluene", "Octane"
    ],

    "MW": [
        16, 18, 44, 46, 46, 60,
        58, 58, 78, 92, 114
    ],

    "BoilingPoint": [
        -161, 100, -42, 78, 101, 118,
        -1, 56, 80, 111, 125
    ]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Define input (X) and target (y)
X = df[["MW"]]
y = df["BoilingPoint"]

# Display the dataset
print("--- CHEMISTRY DATASET ---")
print(df)

print("\n--- DATASET INFORMATION ---")
print("Total Compounds:", len(df))
print("Input Feature:", X.columns.tolist())
print("Target Variable:", y.name)

print("\nDataset created successfully!")



# STEP 2: TRAIN A LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Save graphs in the ArtificialNeuralNetwork folder
folder = Path(__file__).resolve().parent

# Create and train the model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predict boiling points
y_pred = linear_model.predict(X)

# Calculate evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\n--- LINEAR REGRESSION RESULTS ---")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nSlope:", linear_model.coef_[0])
print("Intercept:", linear_model.intercept_)


# GRAPH 1: LINEAR REGRESSION

# Sort molecular weights for a smooth fitted line
X_sorted = np.sort(X["MW"].to_numpy()).reshape(-1, 1)
y_sorted = linear_model.predict(
    pd.DataFrame(X_sorted, columns=["MW"])
)

plt.figure(figsize=(10, 6))

plt.scatter(
    X["MW"], y,
    color="blue",
    label="Actual Boiling Points"
)

plt.plot(
    X_sorted, y_sorted,
    color="red",
    linewidth=2,
    label="Linear Regression"
)

plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression: Molecular Weight vs Boiling Point")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(folder / "LinearRegression_BoilingPoint.png", dpi=300)
plt.close()

print("\nLinear regression graph saved successfully!")


# GRAPH 2: RESIDUAL PLOT

# Residual = Actual - Predicted
residuals = y - y_pred

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred, residuals,
    color="purple"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Boiling Point (°C)")
plt.ylabel("Residual (°C)")
plt.title("Linear Regression Residual Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(folder / "LinearRegression_Residuals.png", dpi=300)
plt.close()

print("Residual plot saved successfully!")



# STEP 3: TRAIN AN ARTIFICIAL NEURAL NETWORK

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

# Scale the input data to help neural network training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create the neural network
neural_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=5000,
    early_stopping=False,
    tol=0.0,
    n_iter_no_change=5001,
    random_state=42
)

# Train the neural network
print("\n--- NEURAL NETWORK TRAINING ---")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    neural_model.fit(X_scaled, y)

# Make predictions
nn_predictions = neural_model.predict(X_scaled)

# Calculate evaluation metrics
nn_mae = mean_absolute_error(y, nn_predictions)
nn_mse = mean_squared_error(y, nn_predictions)
nn_r2 = r2_score(y, nn_predictions)

print("\n--- NEURAL NETWORK RESULTS ---")
print(f"MAE: {nn_mae:.4f}")
print(f"MSE: {nn_mse:.4f}")
print(f"R² Score: {nn_r2:.4f}")
print("Training Epochs:", neural_model.n_iter_)

# Compare with linear regression
print("\n--- MODEL COMPARISON ---")
print(f"Linear Regression R²: {r2:.4f}")
print(f"Neural Network R²: {nn_r2:.4f}")

print("\nNeural network training completed!")



# STEP 4: TEST MODELS ON AN UNSEEN COMPOUND

# Remove Butane from the training dataset
train_df = df[df["Compound"] != "Butane"].copy()
test_df = df[df["Compound"] == "Butane"].copy()

X_train = train_df[["MW"]]
y_train = train_df["BoilingPoint"]

X_test = test_df[["MW"]]
actual_bp = test_df["BoilingPoint"].iloc[0]

# Train a new linear regression model
linear_test_model = LinearRegression()
linear_test_model.fit(X_train, y_train)

# Predict Butane using linear regression
linear_prediction = linear_test_model.predict(X_test)[0]

# Scale data for the neural network
test_scaler = StandardScaler()
X_train_scaled = test_scaler.fit_transform(X_train)
X_test_scaled = test_scaler.transform(X_test)

# Train a new neural network
neural_test_model = MLPRegressor(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=5000,
    early_stopping=False,
    tol=0.0,
    n_iter_no_change=5001,
    random_state=42
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    neural_test_model.fit(X_train_scaled, y_train)

# Predict Butane using the neural network
neural_prediction = neural_test_model.predict(X_test_scaled)[0]

# Calculate prediction errors
linear_error = linear_prediction - actual_bp
neural_error = neural_prediction - actual_bp

print("\n--- BUTANE PREDICTION RESULTS ---")
print(f"Actual Boiling Point: {actual_bp:.2f} °C")

print(f"\nLinear Regression Prediction: {linear_prediction:.2f} °C")
print(f"Linear Regression Error: {linear_error:.2f} °C")

print(f"\nNeural Network Prediction: {neural_prediction:.2f} °C")
print(f"Neural Network Error: {neural_error:.2f} °C")

# Determine which model was closer
if abs(linear_error) < abs(neural_error):
    print("\nLinear Regression predicted Butane more accurately.")
elif abs(neural_error) < abs(linear_error):
    print("\nNeural Network predicted Butane more accurately.")
else:
    print("\nBoth models had the same prediction error.")


# GRAPH 3: COMPARE BUTANE PREDICTIONS

model_names = [
    "Actual Butane",
    "Linear Regression",
    "Neural Network"
]

boiling_points = [
    actual_bp,
    linear_prediction,
    neural_prediction
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    model_names,
    boiling_points,
    color=["green", "blue", "purple"]
)

plt.axhline(y=0, color="black", linewidth=0.8)

plt.ylabel("Boiling Point (°C)")
plt.title("Butane: Actual vs Predicted Boiling Points")

for bar, value in zip(bars, boiling_points):
    plt.annotate(
        f"{value:.2f} °C",
        (bar.get_x() + bar.get_width() / 2, value),
        xytext=(0, 5 if value >= 0 else -15),
        textcoords="offset points",
        ha="center"
    )

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    folder / "Butane_Prediction_Comparison.png",
    dpi=300
)
plt.close()

print("\nButane prediction comparison graph saved successfully!")



# STEP 5: COMPARE BOTH MODELS

print("\n--- FINAL MODEL COMPARISON ---")

print("Linear Regression:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\nNeural Network:")
print(f"MAE: {nn_mae:.4f}")
print(f"MSE: {nn_mse:.4f}")
print(f"R²: {nn_r2:.4f}")
print(f"Training Epochs: {neural_model.n_iter_}")


# GRAPH 4: LINEAR REGRESSION VS NEURAL NETWORK

# Create evenly spaced molecular weights for smooth lines
mw_range = np.linspace(
    X["MW"].min(),
    X["MW"].max(),
    200
).reshape(-1, 1)

mw_df = pd.DataFrame(mw_range, columns=["MW"])

# Predict using both models
linear_curve = linear_model.predict(mw_df)

mw_scaled = scaler.transform(mw_df)
neural_curve = neural_model.predict(mw_scaled)

plt.figure(figsize=(10, 6))

plt.scatter(
    X["MW"], y,
    color="black",
    label="Actual Boiling Points",
    zorder=3
)

plt.plot(
    mw_range,
    linear_curve,
    color="blue",
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    mw_range,
    neural_curve,
    color="purple",
    linewidth=2,
    label="Neural Network"
)

plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression vs Neural Network")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    folder / "Linear_vs_NeuralNetwork.png",
    dpi=300
)
plt.close()

print("\nModel comparison graph saved successfully!")


# GRAPH 5: ReLU ACTIVATION FUNCTION

# ReLU = max(0, x)
mw_values = X["MW"].to_numpy()
relu_values = np.maximum(0, mw_values)

plt.figure(figsize=(10, 6))

plt.scatter(
    mw_values,
    relu_values,
    color="purple",
    s=70,
    label="Molecular Weight After ReLU"
)

plt.plot(
    [0, max(mw_values)],
    [0, max(mw_values)],
    color="blue",
    linestyle="--",
    label="ReLU(x) = max(0, x)"
)

plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("ReLU Output")
plt.title("ReLU Activation Applied to Molecular Weight")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    folder / "ReLU_Activation.png",
    dpi=300
)
plt.close()

print("ReLU activation graph saved successfully!")
print("\nStep 5 completed!")



# STEP 6: EXPERIMENT WITH NEURAL NETWORK ARCHITECTURES

print("\n--- CUSTOM NEURAL NETWORK EXPERIMENT ---")

experiment_number = 0

while True:

    # Ask for number of hidden layers
    while True:
        try:
            num_layers = int(
                input("\nHow many hidden layers (1-4)? ")
            )

            if 1 <= num_layers <= 4:
                break

            print("Please enter a number between 1 and 4.")

        except ValueError:
            print("Please enter a valid whole number.")

    # Ask for neurons in each hidden layer
    neurons = []

    for layer in range(num_layers):

        while True:
            try:
                num_neurons = int(
                    input(
                        f"How many neurons in layer {layer + 1} (1-10)? "
                    )
                )

                if 1 <= num_neurons <= 10:
                    neurons.append(num_neurons)
                    break

                print("Please enter a number between 1 and 10.")

            except ValueError:
                print("Please enter a valid whole number.")

    architecture = tuple(neurons)
    experiment_number += 1

    print("\nTraining custom neural network...")
    print("Architecture:", architecture)

    # Build custom neural network

    custom_model = MLPRegressor(
        hidden_layer_sizes=architecture,
        activation="relu",
        solver="lbfgs",
        max_iter=5000,
        max_fun=50000,
        random_state=42
    )

    # Train using the scaled molecular weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        custom_model.fit(X_scaled, y)

    # Predict boiling points
    custom_predictions = custom_model.predict(X_scaled)

    # Calculate metrics
    custom_mae = mean_absolute_error(y, custom_predictions)
    custom_mse = mean_squared_error(y, custom_predictions)
    custom_r2 = r2_score(y, custom_predictions)

    print("\n--- CUSTOM MODEL RESULTS ---")
    print("Architecture:", architecture)
    print(f"MAE: {custom_mae:.4f}")
    print(f"MSE: {custom_mse:.4f}")
    print(f"R² Score: {custom_r2:.4f}")
    print("Training Epochs:", custom_model.n_iter_)

    # Compare against original neural network
    print("\n--- ORIGINAL MODEL RESULTS ---")
    print("Architecture: (10, 10)")
    print(f"MAE: {nn_mae:.4f}")
    print(f"MSE: {nn_mse:.4f}")
    print(f"R² Score: {nn_r2:.4f}")

    # Predict smooth curves for graph
    custom_curve = custom_model.predict(mw_scaled)

    # Create comparison graph
    plt.figure(figsize=(10, 6))

    plt.scatter(
        X["MW"], y,
        color="black",
        label="Actual Boiling Points",
        zorder=3
    )

    plt.plot(
        mw_range,
        neural_curve,
        color="blue",
        linewidth=2,
        label="Original NN (10, 10)"
    )

    plt.plot(
        mw_range,
        custom_curve,
        color="purple",
        linewidth=2,
        label=f"Custom NN {architecture}"
    )

    plt.xlabel("Molecular Weight (g/mol)")
    plt.ylabel("Boiling Point (°C)")
    plt.title("Neural Network Architecture Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save graph with a unique name
    graph_name = (
        f"NeuralNetwork_Experiment_{experiment_number}.png"
    )

    plt.savefig(folder / graph_name, dpi=300)
    plt.close()

    print("\nComparison graph saved successfully!")
    print("File:", graph_name)

    # Ask whether to run another experiment
    again = input(
        "\nWould you like to try another architecture? (yes/no): "
    ).strip().lower()

    if again not in ["yes", "y"]:
        print("\nNeural network experiments completed!")
        break
