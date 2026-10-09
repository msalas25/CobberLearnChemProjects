
import pandas as pd

input_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_BLA_Agonist_ratio_extracted.csv"

output_file = r"C:\Users\Maryl\PycharmProjects\CobberLearnChemProjects\data\ERa_descriptive_summary.csv"

df = pd.read_csv(input_file)

# Standard invitrodb v4.x activity threshold
df["activity_label"] = df["hitc"].apply(
    lambda x: "Active" if x >= 0.90 else "Inactive"
)

print("--- ER-ALPHA AGONIST DESCRIPTIVES ---")
print("Total assay records:", len(df))

print("\nActivity counts:")
print(df["activity_label"].value_counts())

print("\nActivity percentages:")
print(
    (df["activity_label"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nModel counts:")
print(df["model"].value_counts())

print("\nFit status counts:")
print(df["fitc"].value_counts().sort_index())

# Save a separate file
df.to_csv(output_file, index=False)

print("\nSaved descriptive dataset to:")
print(output_file)