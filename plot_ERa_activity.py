
import pandas as pd
import matplotlib.pyplot as plt

# Read the descriptive dataset you already created
input_file = (
    r"C:\Users\Maryl\PycharmProjects"
    r"\CobberLearnChemProjects\data"
    r"\ERa_descriptive_statistics.csv"
)

# Save the graph in your main project folder
output_file = (
    r"C:\Users\Maryl\PycharmProjects"
    r"\CobberLearnChemProjects"
    r"\ERa_activity_graph.png"
)

df = pd.read_csv(input_file)

# Count the activity classifications
counts = df["activity_label"].value_counts().reindex(
    ["Active", "Inactive", "Unable to fit"],
    fill_value=0
)

# Create the bar graph
plt.figure(figsize=(9, 6))

bars = plt.bar(
    counts.index,
    counts.values,
    color=["seagreen", "steelblue", "darkorange"]
)

plt.title("ERα Agonist Assay Results", fontsize=16)
plt.xlabel("Activity Classification")
plt.ylabel("Number of Assay Records")

plt.bar_label(bars, fmt="%.0f", padding=4)

plt.tight_layout()

# Save a high-resolution image for GitHub
plt.savefig(output_file, dpi=300, bbox_inches="tight")

print("Graph saved successfully!")
print("Location:", output_file)

plt.show()