import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# Create arrays
actual = np.array([2, 4, 5, 4, 5, 7, 9])
predicted = np.array([2.5, 3.5, 4, 5, 6, 8, 8])

# Calculate residuals
residuals = predicted - actual

# Calculate error metrics
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
r_squared = r2_score(actual, predicted)

# Print results
print("ERROR METRICS")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"R-squared (R²): {r_squared:.3f}")

# Display actual, predicted, and residual values
print("\nDATA")
print("-" * 40)
print(f"{'Actual':>10} {'Predicted':>12} {'Residual':>12}")
print("-" * 40)

for a, p, r in zip(actual, predicted, residuals):
    print(f"{a:>10.1f} {p:>12.1f} {r:>12.1f}")

# Find the largest prediction error
worst_index = np.argmax(np.abs(residuals))

print("\nWorst prediction:")
print(f"Actual: {actual[worst_index]}")
print(f"Predicted: {predicted[worst_index]}")
print(f"Residual: {residuals[worst_index]}")

# Save plots in the same folder as this Python file
output_folder = Path(__file__).parent

# Predicted vs. Actual plot
plt.figure(figsize=(8, 6))

plt.scatter(actual, predicted, s=100, label="Predictions")

# Highlight worst prediction
plt.scatter(
    actual[worst_index],
    predicted[worst_index],
    s=180,
    color="red",
    edgecolor="black",
    label="Worst prediction"
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

plt.savefig(
    output_folder / "predicted_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Residual plot
plt.figure(figsize=(8, 6))

plt.scatter(
    predicted,
    residuals,
    s=100
)

# Highlight largest error
plt.scatter(
    predicted[worst_index],
    residuals[worst_index],
    s=180,
    color="red",
    edgecolor="black",
    label="Largest error"
)

# Zero-error line
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

plt.savefig(
    output_folder / "residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()