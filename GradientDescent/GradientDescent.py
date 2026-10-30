
# Chapter 11: Gradient Descent
# Step 2: Generate Noisy Data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create x values from 0 to 10
x = np.linspace(0, 10, 50)

# Define the true line
true_y = 2 * x + 5

# Add random noise to simulate real-world data
np.random.seed(42)
noise = np.random.normal(0, 2, size=x.shape)
noisy_y = true_y + noise

# Plot the noisy data and true line
plt.figure(figsize=(8, 5))

plt.scatter(x, noisy_y, label="Noisy Data")
plt.plot(x, true_y, color="red", label="True Line: y = 2x + 5")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Noisy Data and True Linear Relationship")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("GradientDescent_noisy_data.png", dpi=300)
plt.close()

print("Data generated successfully!")
print("Number of data points:", len(x))


# STEP 3: TRAIN THE LINEAR REGRESSION MODEL

# Reshape x values for scikit-learn
X = x.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(X, noisy_y)

# Predict the best-fit line
predicted_y = model.predict(X)

# STEP 4: PRINT SLOPE AND INTERCEPT
print("\n--- LINEAR REGRESSION RESULTS ---")
print("True Slope: 2")
print("Predicted Slope:", model.coef_[0])
print("True Intercept: 5")
print("Predicted Intercept:", model.intercept_)

# Plot the results
plt.figure(figsize=(8, 5))

plt.scatter(x, noisy_y, label="Noisy Data")
plt.plot(x, true_y, color="red", label="True Line")
plt.plot(x, predicted_y, color="green", label="Best-Fit Line")

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear Regression: True vs Best-Fit Line")
plt.legend()
plt.grid(True, alpha=0.3)

# Save the graph for GitHub
plt.savefig("GradientDescent_regression.png", dpi=300)

plt.close()

print("Regression graph saved successfully!")



# STEP 6: CALCULATE MEAN SQUARED ERROR

def calculate_mse(slope, intercept):
    predicted = slope * x + intercept
    mse = np.mean((noisy_y - predicted) ** 2)
    return mse


# Test different slope and intercept values
print("\n--- MEAN SQUARED ERROR ---")

print("True Line MSE:", calculate_mse(2, 5))

print("Best-Fit Line MSE:", calculate_mse(
    model.coef_[0], model.intercept_
))

print("Random Line MSE:", calculate_mse(4, 10))



# STEP 7: CREATE THE LOSS LANDSCAPE

# Create a range of slopes and intercepts
slopes = np.linspace(0, 4, 100)
intercepts = np.linspace(0, 10, 100)

# Create a grid
slope_grid, intercept_grid = np.meshgrid(
    slopes, intercepts
)

# Calculate MSE for each combination
loss_grid = np.zeros_like(slope_grid)

for i in range(slope_grid.shape[0]):
    for j in range(slope_grid.shape[1]):
        loss_grid[i, j] = calculate_mse(
            slope_grid[i, j],
            intercept_grid[i, j]
        )

# Create the loss landscape
plt.figure(figsize=(9, 6))

plt.contourf(
    slope_grid,
    intercept_grid,
    loss_grid,
    levels=50,
    cmap="viridis_r"
)

plt.colorbar(label="Mean Squared Error")

# Mark the best-fit parameters
plt.scatter(
    model.coef_[0],
    model.intercept_,
    color="red",
    marker="X",
    s=120,
    label="Best-Fit Parameters"
)

plt.xlabel("Slope")
plt.ylabel("Intercept")
plt.title("Loss Landscape: Mean Squared Error")
plt.legend()
plt.tight_layout()

# Save without opening a graph window
plt.savefig("GradientDescent_loss_landscape.png", dpi=300)
plt.close()

print("\nLoss landscape saved successfully!")
