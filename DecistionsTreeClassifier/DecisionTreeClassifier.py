
import pandas as pd

# Chemical properties of 12 molecules
data = {
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Name each molecule
df.index = [f"Molecule {i}" for i in range(1, 13)]

# Display the dataset
print("--- CHEMICAL DATASET ---")
print(df)

print("\nTotal molecules:", len(df))

# Import the Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

# Separate the chemical features from water solubility
X = df[["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]]
y = df["Water Solubility"]

# Create the decision tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# Train the model using the chemical dataset
model.fit(X, y)

# Display the results
print("\n--- DECISION TREE CLASSIFIER ---")
print("Model trained successfully!")
print("Number of molecules:", len(X))
print("Features used:", list(X.columns))
print("Target: Water Solubility")


# --- VISUALIZE THE DECISION TREE ---

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pathlib import Path

# Save the graph in the same folder as this Python file
output_folder = Path(__file__).resolve().parent

# Create the figure
plt.figure(figsize=(16, 9))

# Draw the trained decision tree
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Water-Soluble", "Water-Soluble"],
    filled=True,
    rounded=True,
    fontsize=12
)

plt.title("Decision Tree: Predicting Water Solubility")

# Save the image
output_file = output_folder / "DecisionTree_WaterSolubility.png"
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("\n--- DECISION TREE VISUALIZATION ---")
print("Graph saved successfully!")
print("Location:", output_file)

# Display the graph
plt.show()
