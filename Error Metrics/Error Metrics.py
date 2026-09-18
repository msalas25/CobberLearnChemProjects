import numpy as np

# 1. Create the actual and predicted NumPy arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# 2. Calculate the residuals
residuals = predicted - actual

# 3. Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(residuals))

# 4. Calculate Mean Squared Error (MSE)
mse = np.mean(residuals ** 2)

# 5. Calculate R-squared (R²)
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# 6. Print the results
print("Actual:", actual)
print("Predicted:", predicted)
print("Residuals:", residuals)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r_squared)

import numpy as np
import matplotlib.pyplot as plt

# Import error metrics from scikit-learn
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Actual and predicted values
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Calculate residuals
residuals = predicted - actual

# Calculate metrics using scikit-learn
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r_squared = r2_score(actual, predicted)

# Print results
print("Actual:", actual)
print("Predicted:", predicted)
print("Residuals:", residuals)

print("\nScikit-learn metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r_squared)


# -----------------------------------
# Predicted vs. Actual Scatter Plot
# -----------------------------------

plt.figure()
plt.scatter(actual, predicted)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs. Actual Values")
plt.grid(True)

# Save the plot
plt.savefig("predicted_vs_actual.png", dpi=300, bbox_inches="tight")

plt.show()


# -----------------------------------
# Residual Plot
# -----------------------------------

plt.figure()
plt.scatter(predicted, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)

# Save the plot
plt.savefig("residual_plot.png", dpi=300, bbox_inches="tight")

plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path


# -----------------------------
# Data
# -----------------------------

actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Residuals = predicted - actual
residuals = predicted - actual


# -----------------------------
# Error Metrics
# -----------------------------

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r_squared = r2_score(actual, predicted)

print("ERROR METRICS")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R²): {r_squared:.3f}")


# -----------------------------
# Display Values in Columns
# -----------------------------

print("\nDATA")
print("-" * 40)
print(f"{'Actual':>10} {'Predicted':>12} {'Residual':>12}")
print("-" * 40)

for a, p, r in zip(actual, predicted, residuals):
    print(f"{a:>10.1f} {p:>12.1f} {r:>12.1f}")


# -----------------------------
# Find Worst Prediction
# -----------------------------

worst_index = np.argmax(np.abs(residuals))

print("\nWorst prediction:")
print(f"Actual: {actual[worst_index]}")
print(f"Predicted: {predicted[worst_index]}")
print(f"Residual: {residuals[worst_index]}")


# -----------------------------
# Create ErrorMetrics Folder Path
# -----------------------------

output_folder = Path(__file__).parent


# -----------------------------
# Predicted vs. Actual Plot
# -----------------------------

plt.figure(figsize=(8, 6))

# Regular prediction points
plt.scatter(
    actual,
    predicted,
    s=100,
    label="Predictions"
)

# Highlight worst prediction
plt.scatter(
    actual[worst_index],
    predicted[worst_index],
    s=180,
    color="red",
    label="Worst prediction",
    edgecolor="black"
)

# Perfect prediction line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual Values", fontsize=12)
plt.ylabel("Predicted Values", fontsize=12)
plt.title("Predicted vs. Actual Values", fontsize=15)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plt.savefig(
    output_folder / "predicted_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------
# Residual Plot
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    residuals,
    s=100
)

# Highlight worst residual
plt.scatter(
    predicted[worst_index],
    residuals[worst_index],
    s=180,
    color="red",
    label="Largest error",
    edgecolor="black"
)

# Zero-error reference line
plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.title("Residual Plot", fontsize=15)
plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
plt.savefig(
    output_folder / "residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()