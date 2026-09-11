import matplotlib.pyplot as plt

# Number of carbons in the first 10 linear alkanes
carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Boiling points in degrees Celsius
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1,
                  68.7, 98.4, 125.6, 150.8, 174.1]

# Create the scatterplot
plt.scatter(carbons, boiling_points)

# Add title and labels
plt.title("Boiling Point vs. Number of Carbons")
plt.xlabel("Number of Carbons")
plt.ylabel("Boiling Point (°C)")

# Display the plot
plt.show()